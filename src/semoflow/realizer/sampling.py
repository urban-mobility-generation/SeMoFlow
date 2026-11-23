from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from semoflow.realizer.model import (
    LatentSuffixConditioning,
    PrefixTimeSuffixLatentFlowRealizer,
)
from semoflow.realizer.objective import (
    count_catalog_occurrences,
    normalize_tail,
    suffix_tail_level_embeddings,
    velocity_from_x0,
)
from semoflow.tokenization.sids import (
    codes_to_embeddings,
    find_sorted,
    packed_code_keys,
)

_CATALOG_DECODE_QUERY_CHUNK_SIZE = 4096
_CATALOG_DECODE_MAX_CANDIDATE_PAIRS = 65536
_CATALOG_FEATURE_CHUNK_SIZE = 16384


def _catalog_query_ranges(
    catalog: PrefixTailCatalog,
    active_prefix: torch.Tensor,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    query_keys = catalog.prefix_keys_of(active_prefix)
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
        observed_codes: torch.Tensor,
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
        tail = codes_to_embeddings(
            codes[:, prefix_depth:],
            codebooks[prefix_depth:],
        ).float()
        keys = packed_code_keys(
            codes[:, :prefix_depth], codebook_size=codebook_size
        )
        order = torch.argsort(keys, stable=True)
        counts = count_catalog_occurrences(
            packed_code_keys(codes, codebook_size=codebook_size),
            observed_codes,
            codebook_size=codebook_size,
        )
        log_prior = torch.log1p(counts.float()).to(codes.device)
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

    def prefix_keys_of(self, prefixes: torch.Tensor) -> torch.LongTensor:
        return packed_code_keys(
            prefixes[..., : self.prefix_depth],
            codebook_size=self.codebook_size,
        )

    def supports(self, prefixes: torch.Tensor) -> torch.BoolTensor:
        keys = self.prefix_keys_of(prefixes.to(self.prefix_keys.device))
        return find_sorted(self.prefix_keys, keys)[1]


@dataclass(frozen=True, slots=True)
class PrefixTimeSuffixLatentFlowSample:
    full_sids: torch.LongTensor
    city_constrained_events: int = 0
    city_constraint_fallback_events: int = 0


_UINT32_MASK = 0xFFFFFFFF


def _hashed_uniform(
    keys: torch.Tensor, candidate_indices: torch.Tensor
) -> torch.Tensor:
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
    catalog_noise_coupling: str = "independent",
) -> tuple[torch.LongTensor, _CityConstraint | None]:
    output = _padded_output(
        valid_mask,
        prefix_codes,
        width=catalog.codes.shape[-1],
        pad_id=model.pad_id,
        prefix_depth=model.prefix_depth,
    )
    if not valid_mask.any():
        return output, None

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
        catalog.codes, catalog.codebooks, prefix_depth=model.prefix_depth
    )
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
    return output, city


def _integrate_flow(
    model: PrefixTimeSuffixLatentFlowRealizer,
    state: torch.Tensor,
    conditioning: LatentSuffixConditioning,
    valid: torch.Tensor,
    times_grid: torch.Tensor,
    *,
    solver: str,
) -> torch.Tensor:
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
            predicted = (state + dt * v1).masked_fill(padding, 0.0)
            v2 = velocity(predicted, times_grid[step + 1].expand(rows))
            state = state + dt * (v1 + v2) * 0.5
        else:
            state = state + dt * v1
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
    batch_size = valid.size(0)
    normalized_chunks: list[torch.Tensor] = []
    conditioning_chunks: list[torch.Tensor] = []
    for start in range(0, batch_size, chunk_size):
        rows = slice(start, min(start + chunk_size, batch_size))
        conditioning = model.encode_conditioning(
            prefix[rows],
            times[rows],
            valid[rows],
            city_id=None if city_id is None else city_id[rows],
        )
        conditioning_chunks.append(conditioning.plan)
        normalized_chunks.append(
            _integrate_flow(
                model,
                noise[rows],
                conditioning,
                valid[rows],
                times_grid,
                solver=solver,
            )
        )
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
    num_steps: int,
    solver: str,
    generator: torch.Generator,
    max_batch_size: int,
    catalog_prior_weight: float,
    catalog_temperature: float,
    city_id: torch.LongTensor | None,
    catalog_noise_coupling: str,
) -> PrefixTimeSuffixLatentFlowSample:
    resolved_device = next(model.parameters()).device
    prefix = prefix_codes.to(resolved_device, dtype=torch.long)
    times = time_tokens.to(resolved_device, dtype=torch.long)
    valid = valid_mask.to(resolved_device, dtype=torch.bool)
    mean = tail_mean.to(resolved_device).float()
    tail_whitener = tail_whitener.to(resolved_device).float()
    catalog = catalog.to(resolved_device)

    batch_size, seq_len = valid.shape
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
        chunk_size=max_batch_size,
        solver=solver,
        city_id=city_id,
    )
    full_sids, city = _decode_calibrated_catalog(
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
        catalog_noise_coupling=catalog_noise_coupling,
    )
    return PrefixTimeSuffixLatentFlowSample(
        full_sids=full_sids,
        city_constrained_events=0 if city is None else int(city.constrained),
        city_constraint_fallback_events=(
            0 if city is None else int(city.fallback)
        ),
    )


__all__ = [
    "PrefixTailCatalog",
    "PrefixTimeSuffixLatentFlowSample",
    "sample_prefix_time_suffix_latent_flow",
]
