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
    sid_level_embeddings,
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
    if codebook_size**width > torch.iinfo(torch.long).max:
        raise ValueError(
            "SID width and codebook size exceed int64 key capacity"
        )
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
        if max_width < 1:
            raise ValueError("catalog query candidate width must be positive")
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


@dataclass(frozen=True, slots=True)
class PrefixTailCatalog:
    codes: torch.LongTensor
    tail_embeddings: torch.Tensor
    prefix_keys: torch.LongTensor
    codebook_size: int
    prefix_depth: int
    target_level_start: int
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
        target_level_start: int | None = None,
    ) -> PrefixTailCatalog:
        if catalog_codes.ndim != 2:
            raise ValueError("catalog_codes must have shape [N, M]")
        if catalog_codes.dtype.is_floating_point:
            raise TypeError("catalog_codes must contain integer SID codes")
        if codebooks.ndim != 3:
            raise ValueError("codebooks must have shape [M, C, D]")
        if catalog_codes.shape[1] != codebooks.shape[0]:
            raise ValueError(
                "catalog_codes and codebooks have different widths"
            )
        if not 0 < prefix_depth < catalog_codes.shape[1]:
            raise ValueError("prefix_depth must be within the SID width")
        if catalog_codes.numel() == 0:
            raise ValueError("catalog_codes cannot be empty")
        codebook_size = int(codebooks.shape[1])
        if catalog_codes.min() < 0 or catalog_codes.max() >= codebook_size:
            raise ValueError("catalog_codes contain invalid code values")
        codes, unique_inverse = torch.unique(
            catalog_codes.long(), dim=0, return_inverse=True
        )
        codes = codes.contiguous()
        if candidate_city is not None:
            city_source = candidate_city.long().cpu()
            if city_source.shape[0] != catalog_codes.shape[0]:
                raise ValueError(
                    "candidate_city must align with catalog_codes rows"
                )
            mapped = torch.full((codes.shape[0],), -1, dtype=torch.long)
            reversed_index = torch.arange(
                city_source.shape[0] - 1, -1, -1, dtype=torch.long
            )
            mapped[unique_inverse.cpu()[reversed_index]] = city_source[
                reversed_index
            ]
            candidate_city = mapped
        level_start = (
            prefix_depth if target_level_start is None else target_level_start
        )
        if not 0 <= level_start < catalog_codes.shape[1]:
            raise ValueError("target_level_start must be within the SID width")
        tail = codes_to_embeddings_from_codebooks(
            codes[:, level_start:],
            codebooks[level_start:],
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
            if (
                observed_codes.ndim != 2
                or observed_codes.shape[1] != codes.shape[1]
            ):
                raise ValueError("observed_codes must have shape [N, M]")
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
            target_level_start=int(level_start),
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

    def decode(
        self,
        prefix_codes: torch.Tensor,
        tail_embeddings: torch.Tensor,
        valid_mask: torch.BoolTensor,
        *,
        pad_id: int,
        row_city: torch.Tensor | None = None,
        stats: dict[str, int] | None = None,
    ) -> torch.LongTensor:
        if prefix_codes.ndim != 3:
            raise ValueError("prefix_codes must have shape [B, L, P]")
        if tail_embeddings.ndim != 3:
            raise ValueError("tail_embeddings must have shape [B, L, D]")
        if valid_mask.shape != prefix_codes.shape[:2]:
            raise ValueError("valid_mask must align with prefix_codes")
        if prefix_codes.shape[-1] != self.prefix_depth:
            raise ValueError("prefix_codes width does not match the catalog")
        if tail_embeddings.shape[:2] != valid_mask.shape:
            raise ValueError("tail_embeddings and valid_mask must align")
        if not valid_mask.any():
            return torch.full(
                (*valid_mask.shape, self.codes.shape[-1]),
                pad_id,
                dtype=torch.long,
                device=prefix_codes.device,
            )

        output = torch.full(
            (*valid_mask.shape, self.codes.shape[-1]),
            pad_id,
            dtype=torch.long,
            device=prefix_codes.device,
        )
        output[..., : self.prefix_depth] = torch.where(
            valid_mask[..., None],
            prefix_codes.long(),
            torch.full_like(prefix_codes, pad_id),
        )
        active_prefix = prefix_codes[valid_mask].long()
        active_tail = tail_embeddings[valid_mask].float()
        candidate_city = self.candidate_city
        active_city = None
        if row_city is not None and candidate_city is not None:
            if row_city.ndim != 1 or row_city.shape[0] != valid_mask.shape[0]:
                raise ValueError("row_city must have shape [B]")
            active_city = (
                row_city.to(valid_mask.device)
                .long()[:, None]
                .expand_as(valid_mask)[valid_mask]
            )
        left, right = _catalog_query_ranges(self, active_prefix)
        widths = right - left
        query_order = torch.argsort(widths, descending=True, stable=True)
        flat_output = output[valid_mask]
        active_codes = flat_output.clone()
        constrained = torch.zeros((), dtype=torch.long, device=left.device)
        fallback = torch.zeros((), dtype=torch.long, device=left.device)
        for query_rows in _catalog_decode_query_chunks(query_order, widths):
            candidate_indices, legal = _candidate_index_block(
                left, right, query_rows
            )
            candidates = self.tail_embeddings[candidate_indices].float()
            distances = (
                (active_tail[query_rows, None, :] - candidates)
                .square()
                .sum(dim=-1)
            )
            keep = legal
            if active_city is not None and candidate_city is not None:
                in_city = (
                    candidate_city[candidate_indices].eq(
                        active_city[query_rows, None]
                    )
                    & legal
                )
                has_in_city = in_city.any(dim=-1)
                constrained += has_in_city.sum()
                fallback += (~has_in_city).sum()
                keep = torch.where(has_in_city[:, None], in_city, legal)
            distances = distances.masked_fill(~keep, float("inf"))
            nearest = distances.argmin(dim=-1)
            selected = candidate_indices.gather(1, nearest[:, None]).squeeze(1)
            active_codes[query_rows] = self.codes[selected]
        if stats is not None:
            stats["city_constrained_events"] = stats.get(
                "city_constrained_events", 0
            ) + int(constrained.item())
            stats["city_constraint_fallback_events"] = stats.get(
                "city_constraint_fallback_events", 0
            ) + int(fallback.item())
        output[valid_mask] = active_codes
        return output


@dataclass(frozen=True, slots=True)
class PrefixTimeSuffixLatentFlowSample:
    full_sids: torch.LongTensor
    tail_embeddings: torch.Tensor
    valid_mask: torch.BoolTensor
    tail_level_embeddings: torch.Tensor | None = None
    city_constrained_events: int = 0
    city_constraint_fallback_events: int = 0
    normalized_flow_path: torch.Tensor | None = None
    flow_times: torch.Tensor | None = None


def _global_catalog_neighbors(
    catalog_embeddings: torch.Tensor,
    queries: torch.Tensor,
    *,
    count: int,
) -> torch.LongTensor:
    if count < 1:
        raise ValueError("global candidate count must be positive")
    count = min(count, int(catalog_embeddings.shape[0]))
    if catalog_embeddings.ndim != 2 or queries.ndim != 2:
        raise ValueError("catalog embeddings and queries must be matrices")
    if catalog_embeddings.shape[1] != queries.shape[1]:
        raise ValueError("catalog embeddings and queries must share width")
    catalog = catalog_embeddings.to(queries.device).float()
    query = queries.float()
    catalog_norm = catalog.square().sum(dim=-1)
    output: list[torch.Tensor] = []
    query_chunk_size = 512
    catalog_chunk_size = 16384
    for query_start in range(0, query.size(0), query_chunk_size):
        query_chunk = query[query_start : query_start + query_chunk_size]
        query_norm = query_chunk.square().sum(dim=-1, keepdim=True)
        best_distance = torch.full(
            (query_chunk.size(0), count),
            float("inf"),
            device=query.device,
        )
        best_index = torch.zeros(
            (query_chunk.size(0), count),
            dtype=torch.long,
            device=query.device,
        )
        for catalog_start in range(0, catalog.size(0), catalog_chunk_size):
            catalog_end = min(
                catalog_start + catalog_chunk_size, catalog.size(0)
            )
            catalog_chunk = catalog[catalog_start:catalog_end]
            distance = (
                query_norm
                + catalog_norm[catalog_start:catalog_end][None, :]
                - 2.0 * query_chunk @ catalog_chunk.transpose(0, 1)
            ).clamp_min_(0.0)
            local_count = min(count, catalog_chunk.size(0))
            local_distance, local_index = distance.topk(
                local_count, dim=-1, largest=False
            )
            local_index = local_index + catalog_start
            merged_distance = torch.cat(
                (best_distance, local_distance), dim=-1
            )
            merged_index = torch.cat((best_index, local_index), dim=-1)
            best_distance, selection = merged_distance.topk(
                count, dim=-1, largest=False
            )
            best_index = merged_index.gather(-1, selection)
        output.append(best_index)
    return torch.cat(output, dim=0)


def _decode_global_nearest_catalog(
    catalog: PrefixTailCatalog,
    target_embeddings: torch.Tensor,
    valid_mask: torch.BoolTensor,
    *,
    pad_id: int,
) -> torch.LongTensor:
    output = torch.full(
        (*valid_mask.shape, catalog.codes.shape[-1]),
        pad_id,
        dtype=torch.long,
        device=valid_mask.device,
    )
    if valid_mask.any():
        nearest = _global_catalog_neighbors(
            catalog.tail_embeddings.float(),
            target_embeddings[valid_mask].float(),
            count=1,
        ).squeeze(-1)
        output[valid_mask] = catalog.codes[nearest]
    return output


def _decode_calibrated_catalog(
    model: PrefixTimeSuffixLatentFlowRealizer,
    catalog: PrefixTailCatalog,
    prefix_codes: torch.Tensor,
    normalized_tail: torch.Tensor,
    conditioning: LatentSuffixConditioning,
    valid_mask: torch.BoolTensor,
    tail_mean: torch.Tensor,
    tail_std: torch.Tensor,
    *,
    tail_whitener: torch.Tensor | None,
    catalog_prior_weight: float,
    catalog_temperature: float,
    global_candidate_count: int,
    generator: torch.Generator,
    row_city: torch.Tensor | None = None,
    stats: dict[str, int] | None = None,
) -> torch.LongTensor:
    if model.catalog_decoder_mode != "calibrated":
        raise ValueError(
            "calibrated catalog decoding requires the calibrated model"
        )
    if catalog_temperature <= 0.0 or catalog_prior_weight < 0.0:
        raise ValueError("catalog calibration parameters are invalid")
    output = torch.full(
        (*valid_mask.shape, catalog.codes.shape[-1]),
        model.pad_id,
        dtype=torch.long,
        device=prefix_codes.device,
    )
    if model.latent_target_scope == "suffix":
        output[..., : model.prefix_depth] = torch.where(
            valid_mask[..., None],
            prefix_codes.long(),
            torch.full_like(prefix_codes, model.pad_id),
        )
    if not valid_mask.any():
        return output

    active_prefix = prefix_codes[valid_mask].long()
    active_tail = normalized_tail[valid_mask].float()
    left = right = None
    active_codes = output[valid_mask].clone()
    active_city = None
    candidate_city = catalog.candidate_city
    if row_city is not None and candidate_city is not None:
        active_city = (
            row_city.to(valid_mask.device)
            .long()[:, None]
            .expand_as(valid_mask)[valid_mask]
        )
    normalized_catalog_tail = normalize_tail(
        catalog.tail_embeddings,
        tail_mean,
        tail_std,
        tail_whitener=tail_whitener,
    )
    catalog_levels = sid_level_embeddings(
        catalog.codes.unsqueeze(1),
        catalog.codebooks,
        level_start=model.target_level_start,
    ).squeeze(1)
    catalog_features = _precompute_catalog_candidate_features(
        model, normalized_catalog_tail, catalog_levels
    )
    del catalog_levels
    if model.latent_target_scope == "full_sid":
        if (
            active_city is not None
            and candidate_city is not None
            and torch.unique(candidate_city).numel() > 1
        ):
            raise ValueError(
                "global full-SID decoding currently requires a single-city "
                "catalog"
            )
        global_indices = _global_catalog_neighbors(
            normalized_catalog_tail,
            active_tail,
            count=global_candidate_count,
        )
        query_order = torch.arange(
            active_tail.size(0), device=active_tail.device
        )
    else:
        left, right = _catalog_query_ranges(catalog, active_prefix)
        widths = right - left
        query_order = torch.argsort(widths, stable=True)
    active_plan = conditioning.plan[valid_mask]
    constrained = torch.zeros((), dtype=torch.long, device=active_tail.device)
    fallback = torch.zeros((), dtype=torch.long, device=active_tail.device)
    if model.latent_target_scope == "full_sid":
        query_chunks = list(
            query_order.split(_CATALOG_DECODE_QUERY_CHUNK_SIZE)
        )
    else:
        query_chunks = _catalog_decode_query_chunks(query_order, widths)
    for query_rows in query_chunks:
        if model.latent_target_scope == "full_sid":
            candidate_indices = global_indices[query_rows]
            legal = torch.ones_like(candidate_indices, dtype=torch.bool)
        else:
            assert left is not None and right is not None
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
        keep = legal
        if active_city is not None:
            assert candidate_city is not None
            in_city = (
                candidate_city[candidate_indices].eq(
                    active_city[query_rows, None]
                )
                & legal
            )
            has_in_city = in_city.any(dim=-1)
            constrained += has_in_city.sum()
            fallback += (~has_in_city).sum()
            keep = torch.where(has_in_city[:, None], in_city, legal)
        logits = logits.masked_fill(~keep, float("-inf"))
        probabilities = torch.softmax(logits / catalog_temperature, dim=-1)
        sampled = torch.multinomial(
            probabilities, 1, generator=generator
        ).squeeze(-1)
        active_codes[query_rows] = candidate_codes[
            torch.arange(query_rows.numel(), device=query_rows.device), sampled
        ]
    output[valid_mask] = active_codes
    if stats is not None:
        stats["city_constrained_events"] = stats.get(
            "city_constrained_events", 0
        ) + int(constrained.item())
        stats["city_constraint_fallback_events"] = stats.get(
            "city_constraint_fallback_events", 0
        ) + int(fallback.item())
    return output


@torch.inference_mode()
def sample_prefix_time_suffix_latent_flow(
    model: PrefixTimeSuffixLatentFlowRealizer,
    prefix_codes: torch.Tensor,
    time_tokens: torch.Tensor,
    valid_mask: torch.BoolTensor,
    *,
    catalog: PrefixTailCatalog,
    tail_mean: torch.Tensor,
    tail_std: torch.Tensor,
    num_steps: int = 32,
    solver: str = "heun",
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    max_batch_size: int | None = None,
    tail_whitener: torch.Tensor | None = None,
    tail_dewhitener: torch.Tensor | None = None,
    catalog_prior_weight: float = 0.25,
    catalog_temperature: float = 1.0,
    global_candidate_count: int = 64,
    city_id: torch.LongTensor | None = None,
    hop_distance_km: torch.Tensor | None = None,
    trace_indices: torch.LongTensor | None = None,
) -> PrefixTimeSuffixLatentFlowSample:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if global_candidate_count < 1:
        raise ValueError("global_candidate_count must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be 'euler' or 'heun'")
    if prefix_codes.ndim != 3 or time_tokens.shape != valid_mask.shape:
        raise ValueError("prefix/time/mask shapes are inconsistent")
    if prefix_codes.shape[:2] != valid_mask.shape:
        raise ValueError("prefix_codes and valid_mask must align")
    if not valid_mask.any(dim=1).all():
        raise ValueError("each batch row must contain a valid event")
    if model.tail_representation == "factorized":
        expected_stats_shape = (model.suffix_depth, model.latent_dim)
    else:
        expected_stats_shape = (model.latent_dim,)
    if (
        tuple(tail_mean.shape) != expected_stats_shape
        or tail_std.shape != tail_mean.shape
    ):
        raise ValueError(
            "tail statistics do not match the model representation: "
            f"expected {expected_stats_shape}"
        )
    if (tail_std <= 0).any() or not torch.isfinite(tail_std).all():
        raise ValueError("tail_std must be finite and strictly positive")
    if tail_whitener is not None or tail_dewhitener is not None:
        if model.tail_representation != "summed":
            raise ValueError(
                "tail whitening is supported only for summed tails"
            )
        matrix_shape = (model.latent_dim, model.latent_dim)
        if (
            tail_whitener is None
            or tail_dewhitener is None
            or tuple(tail_whitener.shape) != matrix_shape
            or tuple(tail_dewhitener.shape) != matrix_shape
        ):
            raise ValueError("tail whiteners must both have shape [D, D]")

    model_device = next(model.parameters()).device
    resolved_device = model_device if device is None else torch.device(device)
    if resolved_device != model_device:
        raise ValueError(
            f"model is on {model_device}, but sampler requested {resolved_device}"
        )
    prefix = prefix_codes.to(resolved_device, dtype=torch.long)
    times = time_tokens.to(resolved_device, dtype=torch.long)
    valid = valid_mask.to(resolved_device, dtype=torch.bool)
    mean = tail_mean.to(resolved_device).float()
    std = tail_std.to(resolved_device).float()
    if tail_whitener is not None:
        tail_whitener = tail_whitener.to(resolved_device).float()
    if tail_dewhitener is not None:
        tail_dewhitener = tail_dewhitener.to(resolved_device).float()
    catalog = catalog.to(resolved_device)

    if generator is None:
        generator = torch.Generator(device=resolved_device)
        generator.manual_seed(torch.seed())
    if torch.device(generator.device) != resolved_device:
        raise ValueError("generator and model must be on the same device")

    batch_size, seq_len = valid.shape
    chunk_size = batch_size if max_batch_size is None else max_batch_size
    if chunk_size <= 0:
        raise ValueError("max_batch_size must be positive when supplied")
    trace_indices_cpu: torch.LongTensor | None = None
    trace_restore: torch.LongTensor | None = None
    if trace_indices is not None:
        if trace_indices.ndim != 1 or trace_indices.dtype.is_floating_point:
            raise ValueError(
                "trace_indices must be a one-dimensional integer tensor"
            )
        trace_indices_cpu = trace_indices.detach().to(
            device="cpu", dtype=torch.long
        )
        if trace_indices_cpu.numel():
            if (
                int(trace_indices_cpu.min()) < 0
                or int(trace_indices_cpu.max()) >= batch_size
            ):
                raise ValueError("trace_indices contain an out-of-range row")
            if (
                torch.unique(trace_indices_cpu).numel()
                != trace_indices_cpu.numel()
            ):
                raise ValueError("trace_indices must contain unique rows")
            sorted_indices, sort_order = torch.sort(trace_indices_cpu)
            trace_indices_cpu = sorted_indices
            trace_restore = torch.empty_like(sort_order)
            trace_restore[sort_order] = torch.arange(
                sort_order.numel(), dtype=torch.long
            )
    times_grid = torch.linspace(
        1.0, 0.0, num_steps + 1, device=resolved_device
    )
    initial_noise = torch.randn(
        batch_size,
        seq_len,
        model.flow_latent_dim,
        device=resolved_device,
        generator=generator,
    ).masked_fill(~valid[..., None], 0.0)
    was_training = model.training
    model.eval()
    normalized_chunks: list[torch.Tensor] = []
    conditioning_chunks: list[torch.Tensor] = []
    flow_path_chunks: list[torch.Tensor] = []
    try:
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            prefix_chunk = prefix[start:end]
            times_chunk = times[start:end]
            valid_chunk = valid[start:end]
            conditioning = model.encode_conditioning(
                prefix_chunk,
                times_chunk,
                valid_chunk,
                city_id=(None if city_id is None else city_id[start:end]),
                hop_distance_km=(
                    None
                    if hop_distance_km is None
                    else hop_distance_km[start:end]
                ),
            )
            conditioning_chunks.append(conditioning.plan.detach())
            state = initial_noise[start:end].clone()
            local_trace = None
            path_states: list[torch.Tensor] = []
            if trace_indices_cpu is not None and trace_indices_cpu.numel():
                selected = (trace_indices_cpu >= start) & (
                    trace_indices_cpu < end
                )
                local_trace = trace_indices_cpu[selected] - start
                local_trace_device = local_trace.to(resolved_device)
                path_states.append(
                    state.index_select(0, local_trace_device).detach().cpu()
                )
            for step in range(num_steps):
                t_value = times_grid[step]
                t = t_value.expand(end - start)
                x0 = model(state, t, conditioning, valid_mask=valid_chunk)
                v1 = velocity_from_x0(state, x0, t)
                v1 = v1.masked_fill(~valid_chunk[..., None], 0.0)
                dt = times_grid[step + 1] - t_value
                if solver == "heun" and step < num_steps - 1:
                    predicted = euler_step(state, v1, dt).masked_fill(
                        ~valid_chunk[..., None], 0.0
                    )
                    next_t = times_grid[step + 1].expand(end - start)
                    next_x0 = model(
                        predicted,
                        next_t,
                        conditioning,
                        valid_mask=valid_chunk,
                    )
                    v2 = velocity_from_x0(predicted, next_x0, next_t)
                    v2 = v2.masked_fill(~valid_chunk[..., None], 0.0)
                    state = heun_step(state, v1, v2, dt)
                else:
                    state = euler_step(state, v1, dt)
                state = state.masked_fill(~valid_chunk[..., None], 0.0)
                if local_trace is not None:
                    path_states.append(
                        state.index_select(0, local_trace_device)
                        .detach()
                        .cpu()
                    )
            normalized_chunks.append(state)
            if path_states:
                flow_path_chunks.append(torch.stack(path_states, dim=0))
    finally:
        model.train(was_training)

    normalized = torch.cat(normalized_chunks, dim=0)
    if model.tail_representation == "factorized":
        tail_level_embeddings = (
            normalized.reshape(
                batch_size,
                seq_len,
                model.suffix_depth,
                model.latent_dim,
            )
            * std
            + mean
        )
        tail = tail_level_embeddings.sum(dim=2)
        tail_level_embeddings = tail_level_embeddings.masked_fill(
            ~valid[..., None, None], 0.0
        )
    else:
        tail_level_embeddings = None
        tail = denormalize_tail(
            normalized,
            mean,
            std,
            tail_dewhitener=tail_dewhitener,
        )
    tail = tail.masked_fill(~valid[..., None], 0.0)
    decode_stats: dict[str, int] = {}
    if model.catalog_decoder_mode == "calibrated":
        conditioning = LatentSuffixConditioning(
            plan=torch.cat(conditioning_chunks, dim=0),
            valid_mask=valid,
        )
        full_sids = _decode_calibrated_catalog(
            model,
            catalog,
            prefix,
            normalized,
            conditioning,
            valid,
            mean,
            std,
            tail_whitener=tail_whitener,
            catalog_prior_weight=catalog_prior_weight,
            catalog_temperature=catalog_temperature,
            global_candidate_count=global_candidate_count,
            generator=generator,
            row_city=city_id,
            stats=decode_stats,
        )
    elif model.latent_target_scope == "full_sid":
        full_sids = _decode_global_nearest_catalog(
            catalog,
            tail,
            valid,
            pad_id=model.pad_id,
        )
    else:
        full_sids = catalog.decode(
            prefix,
            tail,
            valid,
            pad_id=model.pad_id,
            row_city=city_id,
            stats=decode_stats,
        )
    normalized_flow_path = None
    flow_times = None
    if trace_indices_cpu is not None:
        if trace_indices_cpu.numel():
            normalized_flow_path = torch.cat(flow_path_chunks, dim=1)
            if trace_restore is not None:
                normalized_flow_path = normalized_flow_path.index_select(
                    1, trace_restore
                )
        else:
            normalized_flow_path = torch.empty(
                (
                    num_steps + 1,
                    0,
                    seq_len,
                    model.flow_latent_dim,
                ),
                dtype=normalized.dtype,
            )
        flow_times = times_grid.detach().cpu()
    return PrefixTimeSuffixLatentFlowSample(
        full_sids=full_sids,
        tail_embeddings=tail,
        valid_mask=valid,
        tail_level_embeddings=tail_level_embeddings,
        city_constrained_events=decode_stats.get("city_constrained_events", 0),
        city_constraint_fallback_events=decode_stats.get(
            "city_constraint_fallback_events", 0
        ),
        normalized_flow_path=normalized_flow_path,
        flow_times=flow_times,
    )


__all__ = [
    "PrefixTailCatalog",
    "PrefixTimeSuffixLatentFlowSample",
    "sample_prefix_time_suffix_latent_flow",
]
