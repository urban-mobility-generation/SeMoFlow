from __future__ import annotations

import functools
from typing import cast

import torch
import torch.nn.functional as F

from semoflow.poi.sids import codes_to_embeddings
from semoflow.realizer.catalog import CandidateTable
from semoflow.realizer.model import (
    LatentSuffixConditioning,
    PrefixTimeSuffixLatentFlowRealizer,
)
from semoflow.utils import unwrap_ddp

_TIMESTEP_MEAN = -0.5
_TIMESTEP_STD = 1.2
_WHITENING_SHRINKAGE = 0.05
_WHITENING_EPS = 1e-4


def velocity_from_x0(
    zt: torch.Tensor, x0_pred: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    return (zt - x0_pred) / t[:, None, None].clamp_min(1e-6)


def suffix_tail_embeddings(
    clean_codes: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    prefix_depth: int,
) -> torch.Tensor:
    return codes_to_embeddings(
        clean_codes[..., prefix_depth:].long(),
        codebooks.to(clean_codes.device)[prefix_depth:],
    )


def suffix_tail_level_embeddings(
    clean_codes: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    prefix_depth: int,
) -> torch.Tensor:
    suffix_codes = clean_codes[..., prefix_depth:].long()
    levels = torch.arange(suffix_codes.shape[-1], device=clean_codes.device)
    return codebooks.to(clean_codes.device)[prefix_depth:][
        levels, suffix_codes
    ]


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
    chunk_size: int,
    compute_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_codes = clean_codes.reshape(-1, clean_codes.shape[-1])
    flat_valid = valid_mask.reshape(-1)
    stats_device = compute_device
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
        chunk = flat_codes[start:end][flat_valid[start:end]].to(
            stats_device, dtype=torch.long, non_blocking=True
        )
        tail = suffix_tail_embeddings(
            chunk, codebooks, prefix_depth=prefix_depth
        ).double()
        total += tail.size(0)
        first += tail.sum(dim=0)
        second += tail.transpose(0, 1).matmul(tail)
    mean = first / total
    covariance = second / total - torch.outer(mean, mean)
    covariance = (covariance + covariance.transpose(0, 1)) * 0.5
    average_variance = covariance.diagonal().mean().clamp_min(_WHITENING_EPS)
    identity = torch.eye(latent_dim, dtype=torch.float64, device=stats_device)
    covariance = (
        1.0 - _WHITENING_SHRINKAGE
    ) * covariance + _WHITENING_SHRINKAGE * average_variance * identity
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(_WHITENING_EPS)
    whitener = (
        eigenvectors * eigenvalues.rsqrt().unsqueeze(0)
    ) @ eigenvectors.transpose(0, 1)
    return mean.float(), whitener.float()


def _catalog_features(
    model: PrefixTimeSuffixLatentFlowRealizer,
    codes: torch.Tensor,
    codebooks: torch.Tensor,
    tail_mean: torch.Tensor,
    tail_whitener: torch.Tensor,
    *,
    prefix_depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = suffix_tail_embeddings(codes, codebooks, prefix_depth=prefix_depth)
    levels = suffix_tail_level_embeddings(
        codes, codebooks, prefix_depth=prefix_depth
    )
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
    candidates: CandidateTable,
    tail_mean: torch.Tensor,
    tail_whitener: torch.Tensor,
    *,
    catalog_prior_weight: float,
    catalog_temperature: float,
    population_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_codes = clean_codes[valid_mask].long()
    active_pred = predicted_normalized[valid_mask]
    prefix_depth = candidates.catalog.prefix_depth
    prefix_indices = candidates.prefix_index(active_codes)
    unique_indices, inverse = torch.unique(
        prefix_indices, sorted=True, return_inverse=True
    )
    features = functools.partial(
        _catalog_features,
        model,
        codebooks=codebooks,
        tail_mean=tail_mean,
        tail_whitener=tail_whitener,
        prefix_depth=prefix_depth,
    )
    base_normalized, base_features = features(candidates.codes[unique_indices])
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
        candidates.codes[prefix_indices]
        .eq(active_codes[:, None, :])
        .all(dim=-1)
    )
    valid_candidates = torch.cat(
        (
            torch.ones_like(duplicate[:, :1]),
            candidates.mask[prefix_indices] & ~duplicate,
        ),
        dim=-1,
    )
    compatibility = torch.cat(
        (target_compatibility[:, None], base_compatibility), dim=-1
    )
    distance = torch.cat((target_distance[:, None], base_distance), dim=-1)
    log_prior = torch.cat(
        (
            candidates.catalog.log_prior_for_codes(active_codes)[:, None],
            candidates.log_prior[prefix_indices],
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
    candidates: CandidateTable,
    *,
    catalog_loss_weight: float,
    catalog_temperature: float,
    endpoint_loss_weight: float,
    population_loss_weight: float,
    catalog_prior_weight: float,
    city_id: torch.LongTensor | None,
) -> torch.Tensor:
    core = cast(PrefixTimeSuffixLatentFlowRealizer, unwrap_ddp(model))

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
    timesteps = torch.sigmoid(
        torch.randn(clean_codes.size(0), device=clean_codes.device)
        * _TIMESTEP_STD
        + _TIMESTEP_MEAN
    )
    t = timesteps[:, None, None]
    noisy_tail = ((1.0 - t) * clean_tail + t * noise).masked_fill(padding, 0.0)
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
    target_velocity = noise - clean_tail
    predicted_velocity = velocity_from_x0(noisy_tail, x0_pred, timesteps)
    per_event = (predicted_velocity - target_velocity).square().mean(dim=-1)
    loss = per_event[valid_mask].mean()
    if endpoint_loss_weight > 0.0:
        loss = loss + endpoint_loss_weight * F.smooth_l1_loss(
            x0_pred[valid_mask], clean_tail[valid_mask]
        )
    catalog_loss, population_loss = _catalog_losses(
        core,
        conditioning,
        x0_pred,
        clean_codes,
        valid_mask,
        codebooks,
        candidates,
        tail_mean,
        tail_whitener,
        catalog_prior_weight=catalog_prior_weight,
        catalog_temperature=catalog_temperature,
        population_loss_weight=population_loss_weight,
    )
    return (
        loss
        + catalog_loss_weight * catalog_loss
        + population_loss_weight * population_loss
    )


__all__ = [
    "compute_tail_whitening_stats",
    "normalize_tail",
    "prefix_time_suffix_latent_flow_loss",
    "suffix_tail_embeddings",
    "suffix_tail_level_embeddings",
    "velocity_from_x0",
]
