from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from semoflow.realizer.flow_matching import (
    compute_target_velocity,
    linear_interpolate,
    sample_logit_normal_timesteps,
    velocity_from_x0,
)
from semoflow.realizer.model import (
    LatentSuffixConditioning,
    PrefixTimeSuffixLatentFlowRealizer,
)
from semoflow.tokenization.decoding.codebooks import (
    codes_to_embeddings_from_codebooks,
)


def _sid_keys(
    codes: torch.Tensor,
    *,
    codebook_size: int,
    width: int,
) -> torch.LongTensor:
    powers = torch.tensor(
        [codebook_size**power for power in range(width - 1, -1, -1)],
        dtype=torch.long,
        device=codes.device,
    )
    return (codes.long() * powers).sum(dim=-1)


def _count_catalog_occurrences(
    catalog_keys: torch.Tensor,
    observed_codes: torch.Tensor,
    observed_valid_mask: torch.BoolTensor | None,
    *,
    codebook_size: int,
    width: int,
    count_device: torch.device | str | None,
    chunk_size: int,
) -> torch.Tensor:
    """Count how often each catalog SID occurs among the observed codes."""

    device = (
        observed_codes.device
        if count_device is None
        else torch.device(count_device)
    )
    catalog_order = torch.argsort(catalog_keys, stable=True)
    sorted_catalog_keys = catalog_keys[catalog_order].to(device)
    sorted_counts = torch.zeros(
        catalog_keys.numel(), dtype=torch.long, device=device
    )
    flat_observed = observed_codes.reshape(-1, width)
    flat_mask = (
        None
        if observed_valid_mask is None
        else observed_valid_mask.reshape(-1)
    )
    for start in range(0, flat_observed.size(0), chunk_size):
        end = min(start + chunk_size, flat_observed.size(0))
        chunk = flat_observed[start:end]
        if flat_mask is not None:
            chunk_mask = flat_mask[start:end]
            if not chunk_mask.any():
                continue
            chunk = chunk[chunk_mask]
        chunk_keys = _sid_keys(
            chunk.to(device, dtype=torch.long, non_blocking=True),
            codebook_size=codebook_size,
            width=width,
        )
        positions = torch.searchsorted(sorted_catalog_keys, chunk_keys)
        in_range = positions < sorted_catalog_keys.numel()
        safe = positions.clamp_max(sorted_catalog_keys.numel() - 1)
        matched = in_range & sorted_catalog_keys[safe].eq(chunk_keys)
        sorted_counts.scatter_add_(
            0, safe[matched], torch.ones_like(safe[matched])
        )
    counts = torch.zeros(catalog_keys.numel(), dtype=torch.long)
    counts[catalog_order] = sorted_counts.cpu()
    return counts


def _prefix_candidate_rows(
    rows: list[int],
    *,
    catalog_counts: torch.Tensor,
    catalog_rows: list[list[int]],
    max_candidates: int,
) -> torch.Tensor:
    """Most frequent half of the budget, then strided coverage."""
    rows = sorted(
        rows,
        key=lambda row_index: (
            -int(catalog_counts[row_index]),
            tuple(catalog_rows[row_index]),
        ),
    )
    if len(rows) > max_candidates:
        frequent_count = max_candidates // 2
        remainder = rows[frequent_count:]
        stride = max(1, len(remainder) // max_candidates)
        rows = (rows[:frequent_count] + remainder[::stride])[:max_candidates]
    return torch.tensor(rows, dtype=torch.long)


def _candidate_table(
    catalog: torch.Tensor,
    catalog_log_prior: torch.Tensor,
    groups: list[torch.Tensor],
    max_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad each group's catalog rows into [groups, max_candidates] tables."""
    shape = (len(groups), max_candidates)
    codes = torch.zeros((*shape, catalog.shape[1]), dtype=torch.long)
    log_prior = torch.full(shape, float("-inf"), dtype=torch.float32)
    mask = torch.zeros(shape, dtype=torch.bool)
    for index, rows in enumerate(groups):
        count = rows.numel()
        codes[index, :count] = catalog[rows]
        log_prior[index, :count] = catalog_log_prior[rows]
        mask[index, :count] = True
    return codes, log_prior, mask


@dataclass(frozen=True, slots=True)
class CatalogCandidateBank:
    prefix_depth: int
    codebook_size: int
    max_candidates: int
    codebooks: torch.Tensor
    prefix_keys: torch.LongTensor
    candidate_codes: torch.LongTensor
    candidate_log_prior: torch.Tensor
    candidate_mask: torch.BoolTensor
    full_code_keys: torch.LongTensor
    full_log_prior: torch.Tensor

    @classmethod
    def from_codes(
        cls,
        catalog_codes: torch.Tensor,
        codebooks: torch.Tensor,
        *,
        prefix_depth: int,
        observed_codes: torch.Tensor,
        observed_valid_mask: torch.BoolTensor,
        count_device: torch.device | str | None = None,
        count_chunk_size: int = 1_048_576,
        max_candidates: int = 64,
    ) -> CatalogCandidateBank:
        codebook_size = int(codebooks.shape[1])

        catalog = torch.unique(catalog_codes.long().cpu(), dim=0).contiguous()
        catalog_keys = _sid_keys(
            catalog,
            codebook_size=codebook_size,
            width=catalog.shape[1],
        )
        catalog_counts = _count_catalog_occurrences(
            catalog_keys,
            observed_codes,
            observed_valid_mask,
            codebook_size=codebook_size,
            width=catalog.shape[1],
            count_device=count_device,
            chunk_size=count_chunk_size,
        )
        catalog_log_prior = torch.log1p(catalog_counts.float())
        full_order = torch.argsort(catalog_keys, stable=True)

        prefix_groups: dict[tuple[int, ...], list[int]] = {}
        catalog_rows = catalog.tolist()
        for row_index, row in enumerate(catalog_rows):
            prefix_groups.setdefault(tuple(row[:prefix_depth]), []).append(
                row_index
            )
        sorted_prefixes = sorted(prefix_groups)
        prefix_keys = _sid_keys(
            torch.tensor(sorted_prefixes, dtype=torch.long),
            codebook_size=codebook_size,
            width=prefix_depth,
        )
        groups = [
            _prefix_candidate_rows(
                prefix_groups[prefix],
                catalog_counts=catalog_counts,
                catalog_rows=catalog_rows,
                max_candidates=max_candidates,
            )
            for prefix in sorted_prefixes
        ]
        candidate_codes, candidate_log_prior, candidate_mask = (
            _candidate_table(
                catalog, catalog_log_prior, groups, max_candidates
            )
        )
        return cls(
            prefix_depth=prefix_depth,
            codebook_size=codebook_size,
            max_candidates=max_candidates,
            codebooks=codebooks.float().cpu().contiguous(),
            prefix_keys=prefix_keys,
            candidate_codes=candidate_codes,
            candidate_log_prior=candidate_log_prior,
            candidate_mask=candidate_mask,
            full_code_keys=catalog_keys[full_order].contiguous(),
            full_log_prior=catalog_log_prior[full_order].contiguous(),
        )

    def to(self, device: torch.device | str) -> CatalogCandidateBank:
        return dataclasses.replace(
            self,
            codebooks=self.codebooks.to(device),
            prefix_keys=self.prefix_keys.to(device),
            candidate_codes=self.candidate_codes.to(device),
            candidate_log_prior=self.candidate_log_prior.to(device),
            candidate_mask=self.candidate_mask.to(device),
            full_code_keys=self.full_code_keys.to(device),
            full_log_prior=self.full_log_prior.to(device),
        )

    def _prefix_indices(self, prefixes: torch.Tensor) -> torch.LongTensor:
        keys = _sid_keys(
            prefixes, codebook_size=self.codebook_size, width=self.prefix_depth
        )
        indices = torch.searchsorted(self.prefix_keys, keys)
        valid = indices < self.prefix_keys.numel()
        valid = valid & self.prefix_keys[
            indices.clamp_max(max(self.prefix_keys.numel() - 1, 0))
        ].eq(keys)
        if not valid.all():
            missing = prefixes[~valid][0].tolist()
            raise ValueError(
                "catalog has no legal SID continuation for prefix "
                f"{tuple(int(value) for value in missing)}"
            )
        return indices

    def log_prior_for_codes(self, codes: torch.Tensor) -> torch.Tensor:
        keys = _sid_keys(
            codes,
            codebook_size=self.codebook_size,
            width=codes.shape[-1],
        )
        indices = torch.searchsorted(self.full_code_keys, keys)
        valid = indices < self.full_code_keys.numel()
        safe = indices.clamp_max(max(self.full_code_keys.numel() - 1, 0))
        valid = valid & self.full_code_keys[safe].eq(keys)
        return torch.where(
            valid,
            self.full_log_prior[safe],
            torch.zeros_like(keys, dtype=torch.float32),
        )


def suffix_tail_embeddings(
    clean_codes: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    prefix_depth: int,
) -> torch.Tensor:
    """Summed codebook embeddings of the levels after ``prefix_depth``."""
    return codes_to_embeddings_from_codebooks(
        clean_codes[..., prefix_depth:].long(),
        codebooks.to(clean_codes.device)[prefix_depth:],
    )


def suffix_tail_level_embeddings(
    clean_codes: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    prefix_depth: int,
) -> torch.Tensor:
    """Per-level codebook embeddings ``[..., S, D]`` after ``prefix_depth``."""
    suffix_codes = clean_codes[..., prefix_depth:].long()
    suffix_codebooks = codebooks.to(clean_codes.device)[prefix_depth:]
    suffix_depth = suffix_codes.shape[-1]
    flat_codes = suffix_codes.reshape(-1, suffix_depth)
    level_idx = torch.arange(suffix_depth, device=clean_codes.device)
    selected = suffix_codebooks[level_idx[None, :], flat_codes]
    return selected.reshape(*suffix_codes.shape, codebooks.shape[-1])


def normalize_tail(
    tail: torch.Tensor, tail_mean: torch.Tensor, tail_whitener: torch.Tensor
) -> torch.Tensor:
    return torch.matmul(tail - tail_mean, tail_whitener.transpose(-1, -2))


def compute_tail_whitening_stats(
    clean_codes: torch.Tensor,
    valid_mask: torch.BoolTensor,
    codebooks: torch.Tensor,
    *,
    prefix_depth: int,
    shrinkage: float = 0.05,
    eps: float = 1e-4,
    chunk_size: int = 16384,
    compute_device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_codes = clean_codes.reshape(-1, clean_codes.shape[-1])
    flat_valid = valid_mask.reshape(-1)
    stats_device = (
        clean_codes.device
        if compute_device is None
        else torch.device(compute_device)
    )
    codebooks = codebooks.float().to(stats_device)
    latent_dim = int(codebooks.shape[-1])
    total = torch.zeros((), dtype=torch.float64, device=stats_device)
    first = torch.zeros(latent_dim, dtype=torch.float64, device=stats_device)
    second = torch.zeros(
        latent_dim,
        latent_dim,
        dtype=torch.float64,
        device=stats_device,
    )
    for start in range(0, flat_codes.size(0), chunk_size):
        end = min(start + chunk_size, flat_codes.size(0))
        chunk_mask = flat_valid[start:end]
        if not chunk_mask.any():
            continue
        chunk = flat_codes[start:end][chunk_mask].to(
            stats_device, dtype=torch.long, non_blocking=True
        )
        tail = (
            suffix_tail_embeddings(
                chunk.unsqueeze(1),
                codebooks,
                prefix_depth=prefix_depth,
            )
            .squeeze(1)
            .double()
        )
        total += tail.size(0)
        first += tail.sum(dim=0)
        second += tail.transpose(0, 1).matmul(tail)
    mean = first / total
    covariance = second / total - torch.outer(mean, mean)
    covariance = (covariance + covariance.transpose(0, 1)) * 0.5
    average_variance = covariance.diagonal().mean().clamp_min(eps)
    identity = torch.eye(latent_dim, dtype=torch.float64, device=stats_device)
    covariance = (
        1.0 - shrinkage
    ) * covariance + shrinkage * average_variance * identity
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(eps)
    root = eigenvalues.sqrt()
    whitener = (
        eigenvectors * eigenvalues.rsqrt().unsqueeze(0)
    ) @ eigenvectors.transpose(0, 1)
    dewhitener = (eigenvectors * root.unsqueeze(0)) @ eigenvectors.transpose(
        0, 1
    )
    return mean.float(), whitener.float(), dewhitener.float()


def _catalog_features(
    model: PrefixTimeSuffixLatentFlowRealizer,
    codes: torch.Tensor,
    codebooks: torch.Tensor,
    tail_mean: torch.Tensor,
    tail_whitener: torch.Tensor,
    *,
    prefix_depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized tails and catalog features of [N, M] or [N, K, M] SIDs."""
    rows = codes if codes.ndim == 3 else codes.unsqueeze(1)
    raw = suffix_tail_embeddings(rows, codebooks, prefix_depth=prefix_depth)
    levels = suffix_tail_level_embeddings(
        rows, codebooks, prefix_depth=prefix_depth
    )
    if codes.ndim == 2:
        raw, levels = raw.squeeze(1), levels.squeeze(1)
    normalized = normalize_tail(raw, tail_mean, tail_whitener)
    return normalized, model.catalog_candidate_features(normalized, levels)


def _catalog_softmax_losses(
    logits: torch.Tensor,
    log_prior: torch.Tensor,
    valid_candidates: torch.BoolTensor,
    *,
    catalog_temperature: float,
    population_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy of the target (column 0) against its candidates, and
    the prior-to-posterior KL when weighted (zero otherwise)."""
    logits = (logits / catalog_temperature).masked_fill(
        ~valid_candidates, float("-inf")
    )
    catalog_loss = F.cross_entropy(
        logits,
        torch.zeros(logits.size(0), dtype=torch.long, device=logits.device),
    )
    if population_loss_weight <= 0.0:
        return catalog_loss, catalog_loss * 0.0
    prior = torch.softmax(
        log_prior.masked_fill(~valid_candidates, float("-inf")), dim=-1
    )
    log_q = torch.log_softmax(logits, dim=-1)
    safe_log_prior = torch.where(
        valid_candidates, prior.clamp_min(1e-8).log(), torch.zeros_like(prior)
    )
    population_loss = (
        (prior * (safe_log_prior - log_q).masked_fill(~valid_candidates, 0.0))
        .sum(dim=-1)
        .mean()
    )
    return catalog_loss, population_loss


def _catalog_losses(
    model: PrefixTimeSuffixLatentFlowRealizer,
    conditioning: LatentSuffixConditioning,
    predicted_normalized: torch.Tensor,
    clean_codes: torch.Tensor,
    valid_mask: torch.BoolTensor,
    codebooks: torch.Tensor,
    candidates: CatalogCandidateBank,
    tail_mean: torch.Tensor,
    tail_whitener: torch.Tensor,
    *,
    catalog_prior_weight: float,
    catalog_temperature: float,
    population_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score each target SID against its prefix's catalog candidates."""
    active_codes = clean_codes[valid_mask].long()
    active_pred = predicted_normalized[valid_mask]
    prefix_indices = candidates._prefix_indices(
        active_codes[:, : candidates.prefix_depth]
    )
    unique_indices, inverse = torch.unique(
        prefix_indices, sorted=True, return_inverse=True
    )
    features = functools.partial(
        _catalog_features,
        model,
        codebooks=codebooks,
        tail_mean=tail_mean,
        tail_whitener=tail_whitener,
        prefix_depth=candidates.prefix_depth,
    )
    base_normalized, base_features = features(
        candidates.candidate_codes[unique_indices]
    )
    target_normalized, target_features = features(active_codes)
    query = model.catalog_query_features(conditioning, active_pred, valid_mask)
    scale = model.model_dim**0.5
    target_compatibility = (query * target_features).sum(dim=-1) / scale
    base_compatibility = torch.einsum(
        "nd,nkd->nk", query, base_features[inverse]
    ).div(scale)

    distance_pred = active_pred.float()
    target_distance = (
        (distance_pred - target_normalized.float()).square().mean(dim=-1)
    )
    selected_normalized = base_normalized[inverse].float()
    base_distance = (
        distance_pred.square().mean(dim=-1, keepdim=True)
        + selected_normalized.square().mean(dim=-1)
        - 2.0
        * torch.einsum("nd,nkd->nk", distance_pred, selected_normalized)
        / distance_pred.size(-1)
    ).clamp_min(0.0)

    duplicate = (
        candidates.candidate_codes[prefix_indices]
        .eq(active_codes[:, None, :])
        .all(dim=-1)
    )
    valid_candidates = torch.cat(
        (
            torch.ones_like(duplicate[:, :1]),
            candidates.candidate_mask[prefix_indices] & ~duplicate,
        ),
        dim=-1,
    )
    compatibility = torch.cat(
        (target_compatibility[:, None], base_compatibility), dim=-1
    )
    distance = torch.cat((target_distance[:, None], base_distance), dim=-1)
    log_prior = torch.cat(
        (
            candidates.log_prior_for_codes(active_codes)[:, None],
            candidates.candidate_log_prior[prefix_indices],
        ),
        dim=-1,
    )
    return _catalog_softmax_losses(
        compatibility - 0.5 * distance + catalog_prior_weight * log_prior,
        log_prior,
        valid_candidates,
        catalog_temperature=catalog_temperature,
        population_loss_weight=population_loss_weight,
    )


def prefix_time_suffix_latent_flow_loss(
    model: torch.nn.Module,
    clean_codes: torch.Tensor,
    time_tokens: torch.Tensor,
    valid_mask: torch.BoolTensor,
    codebooks: torch.Tensor,
    tail_mean: torch.Tensor,
    tail_whitener: torch.Tensor,
    candidate_bank: CatalogCandidateBank,
    *,
    p_mean: float = -0.5,
    p_std: float = 1.2,
    catalog_loss_weight: float,
    catalog_temperature: float = 1.0,
    endpoint_loss_weight: float = 0.0,
    population_loss_weight: float = 0.0,
    catalog_prior_weight: float = 0.25,
    city_id: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Flow-matching loss on whitened suffix tails plus the weighted
    endpoint, calibrated-catalog and population terms."""
    core = getattr(model, "module", model)
    core = getattr(core, "_orig_mod", core)

    clean_tail = normalize_tail(
        suffix_tail_embeddings(
            clean_codes, codebooks, prefix_depth=core.prefix_depth
        ).to(dtype=tail_mean.dtype),
        tail_mean,
        tail_whitener,
    )
    padding = ~valid_mask[..., None]
    clean_tail = clean_tail.masked_fill(padding, 0.0)
    noise = torch.randn_like(clean_tail).masked_fill(padding, 0.0)
    timesteps = sample_logit_normal_timesteps(
        clean_codes.size(0), clean_codes.device, p_mean=p_mean, p_std=p_std
    )
    noisy_tail = linear_interpolate(clean_tail, noise, timesteps).masked_fill(
        padding, 0.0
    )
    conditioning = core.encode_conditioning(
        clean_codes[..., : core.prefix_depth].long(),
        time_tokens.long(),
        valid_mask,
        city_id=city_id,
    )
    x0_pred = model(
        noisy_tail,
        timesteps,
        conditioning,
        valid_mask=valid_mask,
    )
    target_velocity = compute_target_velocity(clean_tail, noise)
    predicted_velocity = velocity_from_x0(noisy_tail, x0_pred, timesteps)
    per_event = (predicted_velocity - target_velocity).square().mean(dim=-1)
    loss = per_event[valid_mask].mean()
    if endpoint_loss_weight > 0.0:
        loss = loss + endpoint_loss_weight * F.smooth_l1_loss(
            x0_pred[valid_mask], clean_tail[valid_mask]
        )
    if catalog_loss_weight > 0.0:
        catalog_loss, population_loss = _catalog_losses(
            core,
            conditioning,
            x0_pred,
            clean_codes,
            valid_mask,
            codebooks,
            candidate_bank,
            tail_mean,
            tail_whitener,
            catalog_prior_weight=catalog_prior_weight,
            catalog_temperature=catalog_temperature,
            population_loss_weight=population_loss_weight,
        )
        loss = (
            loss
            + catalog_loss_weight * catalog_loss
            + population_loss_weight * population_loss
        )
    return loss


__all__ = [
    "CatalogCandidateBank",
    "compute_tail_whitening_stats",
    "normalize_tail",
    "prefix_time_suffix_latent_flow_loss",
    "suffix_tail_embeddings",
    "suffix_tail_level_embeddings",
]
