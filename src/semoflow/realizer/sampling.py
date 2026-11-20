from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from semoflow.realizer.flow_matching import (
    euler_step,
    heun_step,
    velocity_from_x0,
)
from semoflow.realizer.model import (
    LatentSuffixConditioning,
    PrefixTimeSuffixLatentFlowRealizer,
)
from semoflow.realizer.objective import (
    denormalize_tail,
    normalize_tail,
    suffix_tail_level_embeddings,
)
from semoflow.tokenization.decoding.codebooks import (
    codes_to_embeddings_from_codebooks,
)

_CATALOG_DECODE_QUERY_CHUNK_SIZE = 4096
_CATALOG_DECODE_MAX_CANDIDATE_PAIRS = 65536
_CATALOG_FEATURE_CHUNK_SIZE = 16384


def _packed_code_keys(
    codes: torch.Tensor, *, codebook_size: int
) -> torch.LongTensor:
    width = codes.shape[-1]
    powers = torch.tensor(
        [codebook_size**power for power in range(width - 1, -1, -1)],
        dtype=torch.long,
        device=codes.device,
    )
    return (codes.long() * powers).sum(dim=-1)


def _catalog_query_ranges(
    catalog: PrefixTailCatalog,
    active_prefix: torch.Tensor,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    query_keys = catalog._prefix_keys(active_prefix)
    left = torch.searchsorted(catalog.prefix_keys, query_keys, side="left")
    right = torch.searchsorted(catalog.prefix_keys, query_keys, side="right")
    missing = left == right
    if missing.any():
        row = int(torch.where(missing)[0][0])
        prefix = tuple(int(value) for value in active_prefix[row])
        raise ValueError(
            f"catalog has no legal SID continuation for prefix {prefix}"
        )
    return left, right


def _candidate_index_block(
    left: torch.LongTensor,
    right: torch.LongTensor,
    query_rows: torch.LongTensor,
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    starts = left[query_rows]
    widths = right[query_rows] - starts
    offsets = torch.arange(
        int(widths.max()), device=left.device, dtype=torch.long
    )
    legal = offsets[None, :] < widths[:, None]
    indices = starts[:, None] + offsets[None, :]
    indices = torch.where(legal, indices, starts[:, None])
    return indices, legal


def _catalog_decode_query_chunks(
    query_order: torch.LongTensor,
    widths: torch.LongTensor,
) -> list[torch.LongTensor]:
    chunks: list[torch.LongTensor] = []
    start = 0
    while start < query_order.numel():
        max_width = int(widths[query_order[start]].item())
        pair_limited_rows = max(
            1, _CATALOG_DECODE_MAX_CANDIDATE_PAIRS // max_width
        )
        count = min(
            _CATALOG_DECODE_QUERY_CHUNK_SIZE,
            pair_limited_rows,
            query_order.numel() - start,
        )
        chunks.append(query_order[start : start + count])
        start += count
    return chunks


def _precompute_catalog_candidate_features(
    model: PrefixTimeSuffixLatentFlowRealizer,
    normalized_catalog_tail: torch.Tensor,
    catalog_levels: torch.Tensor,
) -> torch.Tensor:
    chunks = []
    for start in range(
        0, normalized_catalog_tail.size(0), _CATALOG_FEATURE_CHUNK_SIZE
    ):
        end = min(
            start + _CATALOG_FEATURE_CHUNK_SIZE,
            normalized_catalog_tail.size(0),
        )
        chunks.append(
            model.catalog_candidate_features(
                normalized_catalog_tail[start:end], catalog_levels[start:end]
            )
        )
    return torch.cat(chunks, dim=0)


def _padded_output(
    valid_mask: torch.Tensor,
    prefix_codes: torch.Tensor,
    *,
    width: int,
    pad_id: int,
    prefix_depth: int,
) -> torch.Tensor:
    """Pad-filled [B, L, width] codes, with valid prefixes copied in."""
    output = torch.full(
        (*valid_mask.shape, width),
        pad_id,
        dtype=torch.long,
        device=prefix_codes.device,
    )
    output[..., :prefix_depth] = torch.where(
        valid_mask[..., None],
        prefix_codes.long(),
        torch.full_like(prefix_codes, pad_id),
    )
    return output


class _CityConstraint:
    """Keep in-city candidates, falling back to every legal candidate.

    Counts how many queries were constrained and how many fell back.
    """

    def __init__(
        self,
        candidate_city: torch.Tensor | None,
        row_city: torch.Tensor | None,
        valid_mask: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.candidate_city = candidate_city
        self.active_city = (
            None
            if row_city is None or candidate_city is None
            else row_city.to(valid_mask.device)
            .long()[:, None]
            .expand_as(valid_mask)[valid_mask]
        )
        self.constrained = torch.zeros((), dtype=torch.long, device=device)
        self.fallback = torch.zeros((), dtype=torch.long, device=device)

    def keep(
        self,
        candidate_indices: torch.Tensor,
        query_rows: torch.Tensor,
        legal: torch.Tensor,
    ) -> torch.Tensor:
        if self.active_city is None or self.candidate_city is None:
            return legal
        in_city = (
            self.candidate_city[candidate_indices].eq(
                self.active_city[query_rows, None]
            )
            & legal
        )
        has_in_city = in_city.any(dim=-1)
        self.constrained += has_in_city.sum()
        self.fallback += (~has_in_city).sum()
        return torch.where(has_in_city[:, None], in_city, legal)

    def record(self, stats: dict[str, int] | None) -> None:
        if stats is None:
            return
        for key, count in (
            ("city_constrained_events", self.constrained),
            ("city_constraint_fallback_events", self.fallback),
        ):
            stats[key] = stats.get(key, 0) + int(count.item())


@dataclass(frozen=True, slots=True)
class PrefixTailCatalog:
    codes: torch.LongTensor
    tail_embeddings: torch.Tensor
    prefix_keys: torch.LongTensor
    codebook_size: int
    prefix_depth: int
    codebooks: torch.Tensor
    log_prior: torch.Tensor
    candidate_city: torch.LongTensor | None = None

    @classmethod
    def from_codes(
        cls,
        catalog_codes: torch.Tensor,
        codebooks: torch.Tensor,
        *,
        prefix_depth: int,
        observed_codes: torch.Tensor | None = None,
        candidate_city: torch.Tensor | None = None,
    ) -> PrefixTailCatalog:
        codebook_size = int(codebooks.shape[1])
        codes, unique_inverse = torch.unique(
            catalog_codes.long(), dim=0, return_inverse=True
        )
        codes = codes.contiguous()
        if candidate_city is not None:
            city_source = candidate_city.long().cpu()
            mapped = torch.full((codes.shape[0],), -1, dtype=torch.long)
            reversed_index = torch.arange(
                city_source.shape[0] - 1, -1, -1, dtype=torch.long
            )
            mapped[unique_inverse.cpu()[reversed_index]] = city_source[
                reversed_index
            ]
            candidate_city = mapped
        tail = codes_to_embeddings_from_codebooks(
            codes[:, prefix_depth:],
            codebooks[prefix_depth:],
        ).float()
        prefix = codes[:, :prefix_depth]
        powers = torch.tensor(
            [
                codebook_size**power
                for power in range(prefix_depth - 1, -1, -1)
            ],
            dtype=torch.long,
            device=codes.device,
        )
        keys = (prefix * powers).sum(dim=-1)
        order = torch.argsort(keys, stable=True)
        log_prior = torch.zeros(codes.size(0), dtype=torch.float32)
        if observed_codes is not None:
            observed_keys = _packed_code_keys(
                observed_codes.long(), codebook_size=codebook_size
            )
            observed_unique, counts = torch.unique(
                observed_keys, sorted=True, return_counts=True
            )
            if observed_unique.numel():
                code_keys = _packed_code_keys(
                    codes, codebook_size=codebook_size
                ).to(observed_unique.device)
                positions = torch.searchsorted(observed_unique, code_keys)
                in_range = positions < observed_unique.numel()
                safe_positions = positions.clamp_max(
                    observed_unique.numel() - 1
                )
                matched = in_range & (
                    observed_unique[safe_positions] == code_keys
                )
                code_counts = torch.zeros_like(code_keys)
                code_counts[matched] = counts[safe_positions[matched]]
                log_prior = torch.log1p(code_counts.float()).to(codes.device)
        return cls(
            codes=codes[order],
            tail_embeddings=tail[order],
            prefix_keys=keys[order],
            codebook_size=codebook_size,
            prefix_depth=int(prefix_depth),
            codebooks=codebooks.float().cpu().contiguous(),
            log_prior=log_prior[order],
            candidate_city=(
                None
                if candidate_city is None
                else candidate_city.long().cpu()[order].contiguous()
            ),
        )

    def to(self, device: torch.device | str) -> PrefixTailCatalog:
        return dataclasses.replace(
            self,
            codes=self.codes.to(device),
            tail_embeddings=self.tail_embeddings.to(device),
            prefix_keys=self.prefix_keys.to(device),
            codebooks=self.codebooks.to(device),
            log_prior=self.log_prior.to(device),
            candidate_city=(
                None
                if self.candidate_city is None
                else self.candidate_city.to(device)
            ),
        )

    def _prefix_keys(self, prefixes: torch.Tensor) -> torch.LongTensor:
        powers = torch.tensor(
            [
                self.codebook_size**power
                for power in range(self.prefix_depth - 1, -1, -1)
            ],
            dtype=torch.long,
            device=prefixes.device,
        )
        return (prefixes.long() * powers).sum(dim=-1)


@dataclass(frozen=True, slots=True)
class PrefixTimeSuffixLatentFlowSample:
    full_sids: torch.LongTensor
    tail_embeddings: torch.Tensor
    valid_mask: torch.BoolTensor
    city_constrained_events: int = 0
    city_constraint_fallback_events: int = 0


CATALOG_NOISE_COUPLINGS = ("independent", "trajectory_prefix")
_UINT32_MASK = 0xFFFFFFFF


def _hashed_uniform(
    keys: torch.Tensor, candidate_indices: torch.Tensor
) -> torch.Tensor:
    """Deterministic uniform(0, 1) noise per (key, catalog candidate) pair.

    Every multiplier is below 2**31 and operands are masked to 32 bits first,
    so no int64 product overflows on CPU or CUDA.
    """
    h = (keys.long() * 0x5BD1E995 + candidate_indices.long()) & _UINT32_MASK
    for multiplier in (0x7FEB352D, 0x2C1B3C6D, 0x297A2D39):
        h = h ^ (h >> 16)
        h = (h * multiplier) & _UINT32_MASK
    h = h ^ (h >> 15)
    return (h.double() + 0.5) / float(2**32)


def _trajectory_prefix_coupling_keys(
    valid_mask: torch.BoolTensor,
    active_prefix: torch.Tensor,
    generator: torch.Generator,
) -> torch.LongTensor:
    """One random key per (trajectory row, planned prefix), per active event.

    Events are ordered as ``valid_mask`` boolean indexing orders them.
    """
    rows = valid_mask.nonzero()[:, 0]
    _, groups = torch.unique(
        torch.cat([rows[:, None], active_prefix.long()], dim=1),
        dim=0,
        return_inverse=True,
    )
    keys = torch.randint(
        0,
        2**31 - 1,
        (int(groups.max().item()) + 1,),
        generator=generator,
        device=active_prefix.device,
    )
    return keys[groups]


def _sample_catalog_candidates(
    logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    catalog_temperature: float,
    generator: torch.Generator,
    coupling_keys: torch.Tensor | None,
) -> torch.LongTensor:
    """Draw one candidate slot per query from ``softmax(logits / T)``.

    Without coupling keys this is an independent multinomial draw.  With keys,
    Gumbel-max noise is shared by every query carrying the same key for the
    same catalog candidate: each query keeps exactly its softmax marginal, while
    same-key queries with similar logits tend to choose the same candidate.
    """
    if coupling_keys is None:
        probabilities = torch.softmax(logits / catalog_temperature, dim=-1)
        return torch.multinomial(
            probabilities, 1, generator=generator
        ).squeeze(-1)
    uniform = _hashed_uniform(coupling_keys[:, None], candidate_indices)
    gumbel = -torch.log(-torch.log(uniform))
    scores = logits.double() / catalog_temperature + gumbel
    return scores.argmax(dim=-1)


def _decode_calibrated_catalog(
    model: PrefixTimeSuffixLatentFlowRealizer,
    catalog: PrefixTailCatalog,
    prefix_codes: torch.Tensor,
    normalized_tail: torch.Tensor,
    conditioning: LatentSuffixConditioning,
    valid_mask: torch.BoolTensor,
    tail_mean: torch.Tensor,
    tail_whitener: torch.Tensor,
    *,
    catalog_prior_weight: float,
    catalog_temperature: float,
    generator: torch.Generator,
    row_city: torch.Tensor | None = None,
    stats: dict[str, int] | None = None,
    catalog_noise_coupling: str = "independent",
) -> torch.LongTensor:
    """Sample from a learned posterior over same-prefix catalog candidates.

    ``catalog_noise_coupling="trajectory_prefix"`` shares the sampling noise
    among a trajectory's events with the same planned prefix (see
    ``_sample_catalog_candidates``); ``"independent"`` keeps per-event draws.
    """
    output = _padded_output(
        valid_mask,
        prefix_codes,
        width=catalog.codes.shape[-1],
        pad_id=model.pad_id,
        prefix_depth=model.prefix_depth,
    )
    if not valid_mask.any():
        return output

    active_tail = normalized_tail[valid_mask].float()
    coupling_keys = (
        _trajectory_prefix_coupling_keys(
            valid_mask, prefix_codes[valid_mask].long(), generator
        )
        if catalog_noise_coupling == "trajectory_prefix"
        else None
    )
    active_codes = output[valid_mask].clone()
    city = _CityConstraint(
        catalog.candidate_city, row_city, valid_mask, active_tail.device
    )
    normalized_catalog_tail = normalize_tail(
        catalog.tail_embeddings, tail_mean, tail_whitener
    )
    catalog_levels = suffix_tail_level_embeddings(
        catalog.codes.unsqueeze(1),
        catalog.codebooks,
        prefix_depth=model.prefix_depth,
    ).squeeze(1)
    catalog_features = _precompute_catalog_candidate_features(
        model, normalized_catalog_tail, catalog_levels
    )
    del catalog_levels
    left, right = _catalog_query_ranges(
        catalog, prefix_codes[valid_mask].long()
    )
    widths = right - left
    query_chunks = _catalog_decode_query_chunks(
        torch.argsort(widths, stable=True), widths
    )
    active_plan = conditioning.plan[valid_mask]
    for query_rows in query_chunks:
        candidate_indices, legal = _candidate_index_block(
            left, right, query_rows
        )
        candidate_codes = catalog.codes[candidate_indices]
        candidate_tail_batch = normalized_catalog_tail[candidate_indices]
        candidate_feature_batch = catalog_features[candidate_indices]
        group_plan = active_plan[query_rows].unsqueeze(1)
        group_conditioning = LatentSuffixConditioning(
            plan=group_plan,
            valid_mask=torch.ones(
                group_plan.shape[:2],
                dtype=torch.bool,
                device=group_plan.device,
            ),
        )
        group_tail = active_tail[query_rows]
        query_features = model.catalog_query_features(
            group_conditioning, group_tail, group_conditioning.valid_mask
        )
        compatibility = torch.einsum(
            "nd,nkd->nk", query_features, candidate_feature_batch
        ).div(model.model_dim**0.5)
        distance = (
            (group_tail[:, None, :] - candidate_tail_batch)
            .square()
            .mean(dim=-1)
        )
        logits = compatibility - 0.5 * distance
        logits = (
            logits
            + catalog_prior_weight * catalog.log_prior[candidate_indices]
        )
        keep = city.keep(candidate_indices, query_rows, legal)
        logits = logits.masked_fill(~keep, float("-inf"))
        sampled = _sample_catalog_candidates(
            logits,
            candidate_indices,
            catalog_temperature=catalog_temperature,
            generator=generator,
            coupling_keys=(
                None if coupling_keys is None else coupling_keys[query_rows]
            ),
        )
        active_codes[query_rows] = candidate_codes[
            torch.arange(query_rows.numel(), device=query_rows.device), sampled
        ]
    output[valid_mask] = active_codes
    city.record(stats)
    return output


def _integrate_flow(
    model: PrefixTimeSuffixLatentFlowRealizer,
    state: torch.Tensor,
    conditioning: LatentSuffixConditioning,
    valid: torch.Tensor,
    times_grid: torch.Tensor,
    *,
    solver: str,
) -> torch.Tensor:
    """Solve the flow ODE from noise (t=1) to data (t=0)."""
    padding = ~valid[..., None]
    rows = state.shape[0]
    num_steps = times_grid.numel() - 1

    def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0 = model(x, t, conditioning, valid_mask=valid)
        return velocity_from_x0(x, x0, t).masked_fill(padding, 0.0)

    for step in range(num_steps):
        t = times_grid[step].expand(rows)
        v1 = velocity(state, t)
        dt = times_grid[step + 1] - times_grid[step]
        if solver == "heun" and step < num_steps - 1:
            predicted = euler_step(state, v1, dt).masked_fill(padding, 0.0)
            v2 = velocity(predicted, times_grid[step + 1].expand(rows))
            state = heun_step(state, v1, v2, dt)
        else:
            state = euler_step(state, v1, dt)
        state = state.masked_fill(padding, 0.0)
    return state


def _integrate_chunks(
    model: PrefixTimeSuffixLatentFlowRealizer,
    noise: torch.Tensor,
    prefix: torch.Tensor,
    times: torch.Tensor,
    valid: torch.Tensor,
    times_grid: torch.Tensor,
    *,
    chunk_size: int,
    solver: str,
    city_id: torch.LongTensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate the flow from ``noise`` in row chunks, in eval mode.

    Returns the normalized endpoints and the conditioning plans.
    """
    batch_size = valid.size(0)
    normalized_chunks: list[torch.Tensor] = []
    conditioning_chunks: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, batch_size, chunk_size):
            rows = slice(start, min(start + chunk_size, batch_size))
            conditioning = model.encode_conditioning(
                prefix[rows],
                times[rows],
                valid[rows],
                city_id=None if city_id is None else city_id[rows],
            )
            conditioning_chunks.append(conditioning.plan.detach())
            normalized_chunks.append(
                _integrate_flow(
                    model,
                    noise[rows].clone(),
                    conditioning,
                    valid[rows],
                    times_grid,
                    solver=solver,
                )
            )
    finally:
        model.train(was_training)
    return torch.cat(normalized_chunks, dim=0), torch.cat(
        conditioning_chunks, dim=0
    )


@torch.inference_mode()
def sample_prefix_time_suffix_latent_flow(
    model: PrefixTimeSuffixLatentFlowRealizer,
    prefix_codes: torch.Tensor,
    time_tokens: torch.Tensor,
    valid_mask: torch.BoolTensor,
    *,
    catalog: PrefixTailCatalog,
    tail_mean: torch.Tensor,
    tail_whitener: torch.Tensor,
    tail_dewhitener: torch.Tensor,
    num_steps: int = 32,
    solver: str = "heun",
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    max_batch_size: int | None = None,
    catalog_prior_weight: float = 0.25,
    catalog_temperature: float = 1.0,
    city_id: torch.LongTensor | None = None,
    catalog_noise_coupling: str = "independent",
) -> PrefixTimeSuffixLatentFlowSample:
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be 'euler' or 'heun'")

    model_device = next(model.parameters()).device
    resolved_device = model_device if device is None else torch.device(device)
    prefix = prefix_codes.to(resolved_device, dtype=torch.long)
    times = time_tokens.to(resolved_device, dtype=torch.long)
    valid = valid_mask.to(resolved_device, dtype=torch.bool)
    mean = tail_mean.to(resolved_device).float()
    tail_whitener = tail_whitener.to(resolved_device).float()
    tail_dewhitener = tail_dewhitener.to(resolved_device).float()
    catalog = catalog.to(resolved_device)

    if generator is None:
        generator = torch.Generator(device=resolved_device)
        generator.manual_seed(torch.seed())

    batch_size, seq_len = valid.shape
    chunk_size = batch_size if max_batch_size is None else max_batch_size
    times_grid = torch.linspace(
        1.0, 0.0, num_steps + 1, device=resolved_device
    )
    initial_noise = torch.randn(
        batch_size,
        seq_len,
        model.latent_dim,
        device=resolved_device,
        generator=generator,
    ).masked_fill(~valid[..., None], 0.0)
    normalized, plan = _integrate_chunks(
        model,
        initial_noise,
        prefix,
        times,
        valid,
        times_grid,
        chunk_size=chunk_size,
        solver=solver,
        city_id=city_id,
    )
    tail = denormalize_tail(normalized, mean, tail_dewhitener).masked_fill(
        ~valid[..., None], 0.0
    )
    decode_stats: dict[str, int] = {}
    full_sids = _decode_calibrated_catalog(
        model,
        catalog,
        prefix,
        normalized,
        LatentSuffixConditioning(plan=plan, valid_mask=valid),
        valid,
        mean,
        tail_whitener,
        catalog_prior_weight=catalog_prior_weight,
        catalog_temperature=catalog_temperature,
        generator=generator,
        row_city=city_id,
        stats=decode_stats,
        catalog_noise_coupling=catalog_noise_coupling,
    )
    return PrefixTimeSuffixLatentFlowSample(
        full_sids=full_sids,
        tail_embeddings=tail,
        valid_mask=valid,
        city_constrained_events=decode_stats.get("city_constrained_events", 0),
        city_constraint_fallback_events=decode_stats.get(
            "city_constraint_fallback_events", 0
        ),
    )


__all__ = [
    "CATALOG_NOISE_COUPLINGS",
    "PrefixTailCatalog",
    "PrefixTimeSuffixLatentFlowSample",
    "sample_prefix_time_suffix_latent_flow",
]
