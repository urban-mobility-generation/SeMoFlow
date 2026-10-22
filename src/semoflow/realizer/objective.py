from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Self

import torch
import torch.nn.functional as F

from semoflow.realizer.flow_matching import (
    compute_target_velocity,
    linear_interpolate,
    sample_logit_normal_timesteps,
    sample_uniform_timesteps,
    velocity_from_x0,
)
from semoflow.realizer.model import (
    LatentSuffixConditioning,
    PrefixTimeSuffixLatentFlowRealizer,
)
from semoflow.tokenization.decoding.codebooks import (
    codes_to_embeddings_from_codebooks,
)

logger = logging.getLogger(__name__)


def _debug_tensor_stats(name: str, value: torch.Tensor) -> None:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum())
    finite_values = detached[finite].float()
    logger.debug(
        "catalog non-finite tensor=%s shape=%s dtype=%s finite=%d nan=%d "
        "posinf=%d neginf=%d finite_min=%s finite_max=%s",
        name,
        tuple(detached.shape),
        detached.dtype,
        finite_count,
        int(torch.isnan(detached).sum()),
        int(torch.isposinf(detached).sum()),
        int(torch.isneginf(detached).sum()),
        float(finite_values.min()) if finite_count else None,
        float(finite_values.max()) if finite_count else None,
    )


def _debug_catalog_anomaly(
    *,
    branch: str,
    tensors: dict[str, torch.Tensor],
    valid_candidates: torch.BoolTensor,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    counts = valid_candidates.sum(dim=-1)
    logger.debug(
        "catalog non-finite branch=%s rows=%d candidates=%d "
        "valid_candidates_min=%d valid_candidates_max=%d",
        branch,
        valid_candidates.size(0),
        valid_candidates.size(1),
        int(counts.min()),
        int(counts.max()),
    )
    for name, value in tensors.items():
        _debug_tensor_stats(name, value)


@dataclass(frozen=True, slots=True)
class LatentSuffixFlowObjectiveOutput:
    loss: torch.Tensor
    conditioning: LatentSuffixConditioning
    clean_tail: torch.Tensor
    noisy_tail: torch.Tensor
    target_velocity: torch.Tensor
    predicted_velocity: torch.Tensor
    x0_pred: torch.Tensor
    timesteps: torch.Tensor
    flow_loss: torch.Tensor
    endpoint_loss: torch.Tensor
    sum_consistency_loss: torch.Tensor
    catalog_loss: torch.Tensor
    population_loss: torch.Tensor


def _sid_keys(
    codes: torch.Tensor,
    *,
    codebook_size: int,
    width: int,
) -> torch.LongTensor:
    if codes.ndim != 2 or codes.shape[-1] != width:
        raise ValueError("codes do not match the requested key width")
    powers = torch.tensor(
        [codebook_size**power for power in range(width - 1, -1, -1)],
        dtype=torch.long,
        device=codes.device,
    )
    return (codes.long() * powers).sum(dim=-1)


@dataclass(frozen=True, slots=True)
class CatalogTailCandidates:
    prefix_depth: int
    codebook_size: int
    groups: dict[int, tuple[torch.LongTensor, torch.Tensor]]

    @classmethod
    def from_codes(
        cls,
        catalog_codes: torch.Tensor,
        codebooks: torch.Tensor,
        *,
        prefix_depth: int,
    ) -> CatalogTailCandidates:
        if catalog_codes.ndim != 2 or codebooks.ndim != 3:
            raise ValueError("catalog codes/codebooks have invalid shapes")
        if catalog_codes.shape[1] != codebooks.shape[0]:
            raise ValueError(
                "catalog codes and codebooks have different widths"
            )
        if not 0 <= prefix_depth < catalog_codes.shape[1]:
            raise ValueError("prefix_depth must be within the SID width")
        if catalog_codes.numel() and (
            catalog_codes.min() < 0
            or catalog_codes.max() >= codebooks.shape[1]
        ):
            raise ValueError(
                "catalog codes contain values outside codebook_size"
            )
        codes = torch.unique(catalog_codes.long(), dim=0).contiguous()
        tails = suffix_tail_embeddings(
            codes.reshape(-1, 1, codes.shape[-1]),
            codebooks,
            prefix_depth=prefix_depth,
        ).reshape(-1, codebooks.shape[-1])
        codebook_size = int(codebooks.shape[1])
        if prefix_depth:
            powers = torch.tensor(
                [
                    codebook_size**power
                    for power in range(prefix_depth - 1, -1, -1)
                ],
                dtype=torch.long,
                device=codes.device,
            )
            keys = (codes[:, :prefix_depth] * powers).sum(dim=-1)
        else:
            keys = torch.zeros(
                codes.size(0), dtype=torch.long, device=codes.device
            )
        order = torch.argsort(keys, stable=True)
        codes = codes[order].cpu()
        tails = tails[order].float().cpu()
        keys = keys[order].cpu()
        groups: dict[int, tuple[torch.LongTensor, torch.Tensor]] = {}
        for key in torch.unique(keys, sorted=True).tolist():
            active = keys == key
            groups[int(key)] = (codes[active], tails[active])
        return cls(
            prefix_depth=prefix_depth,
            codebook_size=codebook_size,
            groups=groups,
        )

    def to(self, device: torch.device | str) -> Self:
        return dataclasses.replace(
            self,
            groups={
                key: (codes.to(device), tails.to(device))
                for key, (codes, tails) in self.groups.items()
            },
        )


@dataclass(frozen=True, slots=True)
class CatalogCandidateBatch:
    codes: torch.LongTensor
    tail_embeddings: torch.Tensor
    level_embeddings: torch.Tensor
    log_prior: torch.Tensor
    valid_mask: torch.BoolTensor


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
        observed_codes: torch.Tensor | None = None,
        observed_valid_mask: torch.BoolTensor | None = None,
        count_device: torch.device | str | None = None,
        count_chunk_size: int = 1_048_576,
        max_candidates: int = 64,
    ) -> CatalogCandidateBank:
        if catalog_codes.ndim != 2 or codebooks.ndim != 3:
            raise ValueError("catalog codes/codebooks have invalid shapes")
        if catalog_codes.shape[1] != codebooks.shape[0]:
            raise ValueError(
                "catalog codes and codebooks have different widths"
            )
        if not 0 <= prefix_depth < catalog_codes.shape[1]:
            raise ValueError("prefix_depth must be within the SID width")
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if catalog_codes.numel() == 0:
            raise ValueError("catalog_codes cannot be empty")
        codebook_size = int(codebooks.shape[1])
        if catalog_codes.min() < 0 or catalog_codes.max() >= codebook_size:
            raise ValueError("catalog_codes contain invalid code values")

        catalog = torch.unique(catalog_codes.long().cpu(), dim=0).contiguous()
        catalog_keys = _sid_keys(
            catalog,
            codebook_size=codebook_size,
            width=catalog.shape[1],
        )
        catalog_counts = torch.zeros(catalog.size(0), dtype=torch.long)
        if observed_codes is not None:
            if observed_codes.ndim not in (2, 3):
                raise ValueError(
                    "observed_codes must have shape [N, M] or [B, L, M]"
                )
            if observed_codes.shape[-1] != catalog.shape[1]:
                raise ValueError(
                    "observed_codes and catalog have different widths"
                )
            if observed_codes.ndim == 3:
                if (
                    observed_valid_mask is None
                    or observed_valid_mask.dtype != torch.bool
                    or observed_valid_mask.shape != observed_codes.shape[:2]
                ):
                    raise ValueError(
                        "rank-three observed_codes require a boolean "
                        "observed_valid_mask with shape [B, L]"
                    )
            elif observed_valid_mask is not None and (
                observed_valid_mask.dtype != torch.bool
                or observed_valid_mask.shape != observed_codes.shape[:1]
            ):
                raise ValueError(
                    "observed_valid_mask must align with observed_codes"
                )
            if count_chunk_size <= 0:
                raise ValueError("count_chunk_size must be positive")

            device = (
                observed_codes.device
                if count_device is None
                else torch.device(count_device)
            )
            catalog_order = torch.argsort(catalog_keys, stable=True)
            sorted_catalog_keys = catalog_keys[catalog_order].to(device)
            sorted_counts = torch.zeros(
                catalog.size(0), dtype=torch.long, device=device
            )
            flat_observed = observed_codes.reshape(-1, catalog.shape[1])
            flat_mask = (
                None
                if observed_valid_mask is None
                else observed_valid_mask.reshape(-1)
            )
            for start in range(0, flat_observed.size(0), count_chunk_size):
                end = min(start + count_chunk_size, flat_observed.size(0))
                chunk = flat_observed[start:end]
                if flat_mask is not None:
                    chunk_mask = flat_mask[start:end]
                    if not chunk_mask.any():
                        continue
                    chunk = chunk[chunk_mask]
                chunk_keys = _sid_keys(
                    chunk.to(device, dtype=torch.long, non_blocking=True),
                    codebook_size=codebook_size,
                    width=catalog.shape[1],
                )
                positions = torch.searchsorted(sorted_catalog_keys, chunk_keys)
                in_range = positions < sorted_catalog_keys.numel()
                safe = positions.clamp_max(sorted_catalog_keys.numel() - 1)
                matched = in_range & sorted_catalog_keys[safe].eq(chunk_keys)
                sorted_counts.scatter_add_(
                    0,
                    safe[matched],
                    torch.ones_like(safe[matched]),
                )
            catalog_counts[catalog_order] = sorted_counts.cpu()

        catalog_log_prior = torch.log1p(catalog_counts.float())
        full_order = torch.argsort(catalog_keys, stable=True)
        full_code_keys = catalog_keys[full_order].contiguous()
        full_log_prior = catalog_log_prior[full_order].contiguous()

        if prefix_depth == 0:
            frequency_order = torch.argsort(
                catalog_counts, descending=True, stable=True
            )
            if catalog.size(0) > max_candidates:
                frequent_count = max_candidates // 2
                frequent = frequency_order[:frequent_count]
                remaining_mask = torch.ones(catalog.size(0), dtype=torch.bool)
                remaining_mask[frequent] = False
                remainder = torch.where(remaining_mask)[0]
                coverage_count = max_candidates - frequent_count
                positions = (
                    torch.linspace(
                        0,
                        remainder.numel() - 1,
                        steps=coverage_count,
                    )
                    .round()
                    .long()
                )
                selected_rows = torch.cat((frequent, remainder[positions]))
            else:
                selected_rows = frequency_order
            count = int(selected_rows.numel())
            candidate_codes = torch.zeros(
                1, max_candidates, catalog.shape[1], dtype=torch.long
            )
            candidate_log_prior = torch.full(
                (1, max_candidates), float("-inf"), dtype=torch.float32
            )
            candidate_mask = torch.zeros(1, max_candidates, dtype=torch.bool)
            candidate_codes[0, :count] = catalog[selected_rows]
            candidate_log_prior[0, :count] = catalog_log_prior[selected_rows]
            candidate_mask[0, :count] = True
            return cls(
                prefix_depth=0,
                codebook_size=codebook_size,
                max_candidates=max_candidates,
                codebooks=codebooks.float().cpu().contiguous(),
                prefix_keys=torch.zeros(1, dtype=torch.long),
                candidate_codes=candidate_codes,
                candidate_log_prior=candidate_log_prior,
                candidate_mask=candidate_mask,
                full_code_keys=full_code_keys,
                full_log_prior=full_log_prior,
            )

        prefix_groups: dict[tuple[int, ...], list[int]] = {}
        catalog_rows = catalog.tolist()
        for row_index, row in enumerate(catalog_rows):
            prefix_groups.setdefault(tuple(row[:prefix_depth]), []).append(
                row_index
            )
        sorted_prefixes = sorted(prefix_groups)
        group_count = len(sorted_prefixes)
        candidate_codes = torch.zeros(
            group_count,
            max_candidates,
            catalog.shape[1],
            dtype=torch.long,
        )
        candidate_log_prior = torch.full(
            (group_count, max_candidates),
            float("-inf"),
            dtype=torch.float32,
        )
        candidate_mask = torch.zeros(
            group_count, max_candidates, dtype=torch.bool
        )
        prefix_keys: list[int] = []
        for group_index, prefix in enumerate(sorted_prefixes):
            rows = prefix_groups[prefix]
            rows.sort(
                key=lambda row_index: (
                    -int(catalog_counts[row_index]),
                    tuple(int(value) for value in catalog_rows[row_index]),
                )
            )
            if len(rows) > max_candidates:
                frequent_count = max_candidates // 2
                frequent = rows[:frequent_count]
                remainder = rows[frequent_count:]
                stride = max(1, len(remainder) // max_candidates)
                covered = remainder[::stride]
                rows = (frequent + covered)[:max_candidates]
            selected = catalog[rows]
            count = selected.size(0)
            candidate_codes[group_index, :count] = selected
            candidate_mask[group_index, :count] = True
            candidate_log_prior[group_index, :count] = catalog_log_prior[
                torch.tensor(rows, dtype=torch.long)
            ]
            prefix_keys.append(
                0
                if prefix_depth == 0
                else int(
                    _sid_keys(
                        torch.tensor([prefix], dtype=torch.long),
                        codebook_size=codebook_size,
                        width=prefix_depth,
                    )[0]
                )
            )

        order = torch.argsort(torch.tensor(prefix_keys), stable=True)
        return cls(
            prefix_depth=prefix_depth,
            codebook_size=codebook_size,
            max_candidates=max_candidates,
            codebooks=codebooks.float().cpu().contiguous(),
            prefix_keys=torch.tensor(prefix_keys, dtype=torch.long)[order],
            candidate_codes=candidate_codes[order],
            candidate_log_prior=candidate_log_prior[order],
            candidate_mask=candidate_mask[order],
            full_code_keys=full_code_keys,
            full_log_prior=full_log_prior,
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
        keys = (
            torch.zeros(
                prefixes.size(0), dtype=torch.long, device=prefixes.device
            )
            if self.prefix_depth == 0
            else _sid_keys(
                prefixes,
                codebook_size=self.codebook_size,
                width=self.prefix_depth,
            )
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

    def gather(self, prefixes: torch.Tensor) -> CatalogCandidateBatch:
        if prefixes.ndim != 2 or prefixes.shape[-1] != self.prefix_depth:
            raise ValueError("prefixes must have shape [N, prefix_depth]")
        indices = self._prefix_indices(prefixes.long())
        codes = self.candidate_codes[indices]
        return CatalogCandidateBatch(
            codes=codes,
            tail_embeddings=suffix_tail_embeddings(
                codes,
                self.codebooks,
                prefix_depth=self.prefix_depth,
            ),
            level_embeddings=suffix_tail_level_embeddings(
                codes,
                self.codebooks,
                prefix_depth=self.prefix_depth,
            ),
            log_prior=self.candidate_log_prior[indices],
            valid_mask=self.candidate_mask[indices],
        )

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


def sid_level_sum_embeddings(
    clean_codes: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    level_start: int,
) -> torch.Tensor:
    if clean_codes.ndim != 3:
        raise ValueError("clean_codes must have shape [B, L, M]")
    if clean_codes.dtype.is_floating_point or clean_codes.dtype.is_complex:
        raise TypeError("clean_codes must contain integer SID codes")
    if codebooks.ndim != 3:
        raise ValueError("codebooks must have shape [M, C, D]")
    codebooks = codebooks.to(clean_codes.device)
    if not 0 <= level_start < clean_codes.shape[-1]:
        raise ValueError("level_start must be within the SID width")
    if codebooks.shape[0] != clean_codes.shape[-1]:
        raise ValueError("codebooks and clean_codes have different SID widths")
    if clean_codes.numel() and (
        clean_codes.min() < 0 or clean_codes.max() >= codebooks.shape[1]
    ):
        raise ValueError("clean_codes contain values outside codebook_size")
    return codes_to_embeddings_from_codebooks(
        clean_codes[..., level_start:].long(),
        codebooks[level_start:],
    )


def sid_level_embeddings(
    clean_codes: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    level_start: int,
) -> torch.Tensor:
    if clean_codes.ndim != 3:
        raise ValueError("clean_codes must have shape [B, L, M]")
    if clean_codes.dtype.is_floating_point or clean_codes.dtype.is_complex:
        raise TypeError("clean_codes must contain integer SID codes")
    if codebooks.ndim != 3:
        raise ValueError("codebooks must have shape [M, C, D]")
    codebooks = codebooks.to(clean_codes.device)
    if codebooks.shape[0] != clean_codes.shape[-1]:
        raise ValueError("codebooks and clean_codes have different SID widths")
    if not 0 <= level_start < clean_codes.shape[-1]:
        raise ValueError("level_start must be within the SID width")
    if clean_codes.numel() and (
        clean_codes.min() < 0 or clean_codes.max() >= codebooks.shape[1]
    ):
        raise ValueError("clean_codes contain values outside codebook_size")
    suffix_codes = clean_codes[..., level_start:].long()
    suffix_codebooks = codebooks[level_start:]
    suffix_depth = suffix_codes.shape[-1]
    flat_codes = suffix_codes.reshape(-1, suffix_depth)
    level_idx = torch.arange(suffix_depth, device=clean_codes.device)
    selected = suffix_codebooks[level_idx[None, :], flat_codes]
    return selected.reshape(
        *suffix_codes.shape,
        codebooks.shape[-1],
    )


def suffix_tail_embeddings(
    clean_codes: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    prefix_depth: int,
) -> torch.Tensor:
    return sid_level_sum_embeddings(
        clean_codes, codebooks, level_start=prefix_depth
    )


def suffix_tail_level_embeddings(
    clean_codes: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    prefix_depth: int,
) -> torch.Tensor:
    return sid_level_embeddings(
        clean_codes, codebooks, level_start=prefix_depth
    )


def normalize_tail(
    tail: torch.Tensor,
    tail_mean: torch.Tensor,
    tail_std: torch.Tensor,
    *,
    tail_whitener: torch.Tensor | None = None,
) -> torch.Tensor:
    if tail_mean.ndim != 1:
        raise ValueError("tail_mean must have shape [latent_dim]")
    if tail_whitener is None:
        if tail_std.shape != tail_mean.shape:
            raise ValueError("tail_std must have shape [latent_dim]")
        return (tail - tail_mean) / tail_std
    if tail_whitener.shape != (tail_mean.numel(), tail_mean.numel()):
        raise ValueError("tail_whitener must have shape [D, D]")
    return torch.matmul(tail - tail_mean, tail_whitener.transpose(-1, -2))


def denormalize_tail(
    normalized_tail: torch.Tensor,
    tail_mean: torch.Tensor,
    tail_std: torch.Tensor,
    *,
    tail_dewhitener: torch.Tensor | None = None,
) -> torch.Tensor:
    if tail_mean.ndim != 1:
        raise ValueError("tail_mean must have shape [latent_dim]")
    if tail_dewhitener is None:
        if tail_std.shape != tail_mean.shape:
            raise ValueError("tail_std must have shape [latent_dim]")
        return normalized_tail * tail_std + tail_mean
    if tail_dewhitener.shape != (tail_mean.numel(), tail_mean.numel()):
        raise ValueError("tail_dewhitener must have shape [D, D]")
    return (
        torch.matmul(
            normalized_tail,
            tail_dewhitener.transpose(-1, -2),
        )
        + tail_mean
    )


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
    if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with shape [B, L]")
    if clean_codes.shape[:2] != valid_mask.shape:
        raise ValueError("clean_codes and valid_mask must share [B, L]")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    if eps <= 0.0 or chunk_size <= 0:
        raise ValueError("eps and chunk_size must be positive")
    flat_codes = clean_codes.reshape(-1, clean_codes.shape[-1])
    flat_valid = valid_mask.reshape(-1)
    if not flat_valid.any():
        raise ValueError("cannot compute tail statistics without valid events")
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


def compute_tail_stats(
    clean_codes: torch.Tensor,
    valid_mask: torch.BoolTensor,
    codebooks: torch.Tensor,
    *,
    prefix_depth: int,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with shape [B, L]")
    if clean_codes.shape[:2] != valid_mask.shape:
        raise ValueError("clean_codes and valid_mask must share [B, L]")
    if not valid_mask.any():
        raise ValueError("cannot compute tail statistics without valid events")
    tails = suffix_tail_embeddings(
        clean_codes,
        codebooks,
        prefix_depth=prefix_depth,
    ).float()
    active = tails[valid_mask]
    mean = active.mean(dim=0)
    std = active.std(dim=0, unbiased=False).clamp_min(eps)
    return mean, std


def compute_factorized_tail_stats(
    clean_codes: torch.Tensor,
    valid_mask: torch.BoolTensor,
    codebooks: torch.Tensor,
    *,
    prefix_depth: int,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with shape [B, L]")
    if clean_codes.shape[:2] != valid_mask.shape:
        raise ValueError("clean_codes and valid_mask must share [B, L]")
    if not valid_mask.any():
        raise ValueError("cannot compute tail statistics without valid events")
    tails = suffix_tail_level_embeddings(
        clean_codes,
        codebooks,
        prefix_depth=prefix_depth,
    ).float()
    active = tails[valid_mask]
    mean = active.mean(dim=0)
    std = active.std(dim=0, unbiased=False).clamp_min(eps)
    return mean, std


def catalog_aware_tail_ranking_loss(
    predicted_tail: torch.Tensor,
    target_tail: torch.Tensor,
    clean_codes: torch.Tensor,
    valid_mask: torch.BoolTensor,
    candidates: CatalogTailCandidates,
    *,
    num_negatives: int = 8,
    temperature: float = 1.0,
) -> torch.Tensor:
    if num_negatives <= 0 or temperature <= 0.0:
        raise ValueError("catalog ranking hyperparameters must be positive")
    if predicted_tail.shape != target_tail.shape:
        raise ValueError("predicted and target tails must have the same shape")
    if predicted_tail.ndim != 3 or clean_codes.ndim != 3:
        raise ValueError("tails/codes must have rank three")
    if valid_mask.shape != predicted_tail.shape[:2]:
        raise ValueError("valid_mask does not align with tails")
    active_pred = predicted_tail[valid_mask].float()
    active_target = target_tail[valid_mask].float()
    active_codes = clean_codes[valid_mask].long()
    if active_pred.numel() == 0:
        return predicted_tail.sum() * 0.0
    powers = torch.tensor(
        [
            candidates.codebook_size**power
            for power in range(candidates.prefix_depth - 1, -1, -1)
        ],
        dtype=torch.long,
        device=active_codes.device,
    )
    keys = (active_codes[:, : candidates.prefix_depth] * powers).sum(dim=-1)
    losses: list[torch.Tensor] = []
    for key in torch.unique(keys, sorted=True).tolist():
        rows = torch.where(keys == key)[0]
        group = candidates.groups.get(int(key))
        if group is None:
            continue
        candidate_codes, candidate_tails = group
        candidate_codes = candidate_codes.to(active_codes.device)
        candidate_tails = candidate_tails.to(active_pred.device).float()
        if candidate_tails.size(0) < 2:
            continue
        group_codes = active_codes[rows]
        exact = (group_codes[:, None, :] == candidate_codes[None, :, :]).all(
            dim=-1
        )
        has_positive = exact.any(dim=1)
        if not has_positive.any():
            continue
        distances = (
            (active_pred[rows, None, :] - candidate_tails[None, :, :])
            .square()
            .mean(dim=-1)
        )
        negative_distances = distances.masked_fill(exact, float("inf"))
        negative_count = min(num_negatives, candidate_tails.size(0) - 1)
        hard_negatives = negative_distances.topk(
            negative_count,
            dim=1,
            largest=False,
        ).values
        has_negatives = torch.isfinite(hard_negatives).all(dim=1)
        usable = has_positive & has_negatives
        if not usable.any():
            continue
        positive_distance = (
            (active_pred[rows] - active_target[rows]).square().mean(dim=-1)
        )
        losses.append(
            F.softplus(
                (positive_distance[:, None] - hard_negatives) / temperature
            )[usable].mean()
        )
    if not losses:
        return predicted_tail.sum() * 0.0
    return torch.stack(losses).mean()


def _global_catalog_losses(
    model: PrefixTimeSuffixLatentFlowRealizer,
    conditioning: LatentSuffixConditioning,
    predicted_normalized: torch.Tensor,
    clean_codes: torch.Tensor,
    valid_mask: torch.BoolTensor,
    codebooks: torch.Tensor,
    candidates: CatalogCandidateBank,
    tail_mean: torch.Tensor,
    tail_std: torch.Tensor,
    *,
    tail_whitener: torch.Tensor | None,
    catalog_prior_weight: float,
    catalog_temperature: float,
    population_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidates.prefix_depth != 0:
        raise ValueError("global catalog losses require prefix_depth=0")
    active_codes = clean_codes[valid_mask].long()
    active_pred = predicted_normalized[valid_mask]
    query = model.catalog_query_features(conditioning, active_pred, valid_mask)
    target_raw = sid_level_sum_embeddings(
        active_codes.unsqueeze(1), codebooks, level_start=0
    ).squeeze(1)
    target_normalized = normalize_tail(
        target_raw, tail_mean, tail_std, tail_whitener=tail_whitener
    )
    target_levels = sid_level_embeddings(
        active_codes.unsqueeze(1), codebooks, level_start=0
    ).squeeze(1)
    target_features = model.catalog_candidate_features(
        target_normalized, target_levels
    )

    negative_mask = candidates.candidate_mask[0]
    negative_codes = candidates.candidate_codes[0, negative_mask]
    negative_prior = candidates.candidate_log_prior[0, negative_mask]
    negative_raw = sid_level_sum_embeddings(
        negative_codes.unsqueeze(1), codebooks, level_start=0
    ).squeeze(1)
    negative_normalized = normalize_tail(
        negative_raw, tail_mean, tail_std, tail_whitener=tail_whitener
    )
    negative_levels = sid_level_embeddings(
        negative_codes.unsqueeze(1), codebooks, level_start=0
    ).squeeze(1)
    negative_features = model.catalog_candidate_features(
        negative_normalized, negative_levels
    )

    scale = model.model_dim**0.5
    positive_compatibility = (query * target_features).sum(dim=-1) / scale
    negative_compatibility = query @ negative_features.transpose(0, 1) / scale
    positive_distance = (active_pred - target_normalized).square().mean(dim=-1)
    distance_dtype = torch.float32
    distance_pred = active_pred.to(distance_dtype)
    distance_candidates = negative_normalized.to(distance_dtype)
    negative_distance = (
        distance_pred.square().mean(dim=-1, keepdim=True)
        + distance_candidates.square().mean(dim=-1).unsqueeze(0)
        - 2.0
        * (distance_pred @ distance_candidates.transpose(0, 1))
        / distance_pred.size(-1)
    )
    negative_distance = negative_distance.clamp_min(0.0)
    target_prior = candidates.log_prior_for_codes(active_codes)
    logits = torch.cat(
        (
            (
                positive_compatibility
                - 0.5 * positive_distance
                + catalog_prior_weight * target_prior
            )[:, None],
            negative_compatibility
            - 0.5 * negative_distance
            + catalog_prior_weight * negative_prior[None, :],
        ),
        dim=-1,
    )
    duplicate = (
        negative_codes[None, :, :].eq(active_codes[:, None, :]).all(dim=-1)
    )
    valid_candidates = torch.cat(
        (
            torch.ones(
                active_codes.size(0),
                1,
                dtype=torch.bool,
                device=active_codes.device,
            ),
            ~duplicate,
        ),
        dim=-1,
    )
    logits = (logits / catalog_temperature).masked_fill(
        ~valid_candidates, float("-inf")
    )
    catalog_loss = F.cross_entropy(
        logits,
        torch.zeros(logits.size(0), dtype=torch.long, device=logits.device),
    )
    population_loss = positive_compatibility.sum() * 0.0
    prior = None
    log_q = None
    if population_loss_weight > 0.0:
        log_prior = torch.cat(
            (
                target_prior[:, None],
                negative_prior[None, :].expand(active_codes.size(0), -1),
            ),
            dim=-1,
        ).masked_fill(~valid_candidates, float("-inf"))
        prior = torch.softmax(log_prior, dim=-1)
        log_q = torch.log_softmax(logits, dim=-1)
        safe_log_prior = torch.where(
            valid_candidates,
            prior.clamp_min(1e-8).log(),
            torch.zeros_like(prior),
        )
        population_loss = (
            (
                prior
                * (safe_log_prior - log_q).masked_fill(~valid_candidates, 0.0)
            )
            .sum(dim=-1)
            .mean()
        )
    if logger.isEnabledFor(logging.DEBUG) and (
        not bool(torch.isfinite(catalog_loss))
        or not bool(torch.isfinite(population_loss))
    ):
        diagnostics = {
            "positive_compatibility": positive_compatibility,
            "negative_compatibility": negative_compatibility,
            "positive_distance": positive_distance,
            "negative_distance": negative_distance,
            "target_prior": target_prior,
            "negative_prior": negative_prior,
            "valid_logits": logits[valid_candidates],
        }
        if prior is not None and log_q is not None:
            diagnostics["valid_prior_probability"] = prior[valid_candidates]
            diagnostics["valid_log_q"] = log_q[valid_candidates]
        _debug_catalog_anomaly(
            branch="global",
            tensors=diagnostics,
            valid_candidates=valid_candidates,
        )
    return catalog_loss, population_loss


def _prefix_catalog_losses(
    model: PrefixTimeSuffixLatentFlowRealizer,
    conditioning: LatentSuffixConditioning,
    predicted_normalized: torch.Tensor,
    clean_codes: torch.Tensor,
    valid_mask: torch.BoolTensor,
    codebooks: torch.Tensor,
    candidates: CatalogCandidateBank,
    tail_mean: torch.Tensor,
    tail_std: torch.Tensor,
    *,
    tail_whitener: torch.Tensor | None,
    catalog_prior_weight: float,
    catalog_temperature: float,
    population_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidates.prefix_depth <= 0:
        raise ValueError(
            "prefix catalog losses require a positive prefix depth"
        )
    active_codes = clean_codes[valid_mask].long()
    active_pred = predicted_normalized[valid_mask]
    prefix_indices = candidates._prefix_indices(
        active_codes[:, : candidates.prefix_depth]
    )
    unique_indices, inverse = torch.unique(
        prefix_indices, sorted=True, return_inverse=True
    )

    base_codes = candidates.candidate_codes[unique_indices]
    base_mask = candidates.candidate_mask[unique_indices]
    base_prior = candidates.candidate_log_prior[unique_indices]
    base_raw = suffix_tail_embeddings(
        base_codes,
        codebooks,
        prefix_depth=candidates.prefix_depth,
    )
    base_normalized = normalize_tail(
        base_raw,
        tail_mean,
        tail_std,
        tail_whitener=tail_whitener,
    )
    base_levels = suffix_tail_level_embeddings(
        base_codes,
        codebooks,
        prefix_depth=candidates.prefix_depth,
    )
    base_features = model.catalog_candidate_features(
        base_normalized, base_levels
    )

    target_raw = suffix_tail_embeddings(
        active_codes.unsqueeze(1),
        codebooks,
        prefix_depth=candidates.prefix_depth,
    ).squeeze(1)
    target_normalized = normalize_tail(
        target_raw,
        tail_mean,
        tail_std,
        tail_whitener=tail_whitener,
    )
    target_levels = suffix_tail_level_embeddings(
        active_codes.unsqueeze(1),
        codebooks,
        prefix_depth=candidates.prefix_depth,
    ).squeeze(1)
    target_features = model.catalog_candidate_features(
        target_normalized, target_levels
    )
    query = model.catalog_query_features(conditioning, active_pred, valid_mask)
    scale = model.model_dim**0.5
    target_compatibility = (query * target_features).sum(dim=-1) / scale
    selected_features = base_features[inverse]
    base_compatibility = torch.einsum(
        "nd,nkd->nk", query, selected_features
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

    target_prior = candidates.log_prior_for_codes(active_codes)
    selected_prior = base_prior[inverse]
    selected_mask = base_mask[inverse]
    selected_codes = base_codes[inverse]
    duplicate = selected_codes.eq(active_codes[:, None, :]).all(dim=-1)
    valid_candidates = torch.cat(
        (
            torch.ones(
                active_codes.size(0),
                1,
                dtype=torch.bool,
                device=active_codes.device,
            ),
            selected_mask & ~duplicate,
        ),
        dim=-1,
    )
    compatibility = torch.cat(
        (target_compatibility[:, None], base_compatibility), dim=-1
    )
    distance = torch.cat((target_distance[:, None], base_distance), dim=-1)
    log_prior = torch.cat((target_prior[:, None], selected_prior), dim=-1)
    logits = (
        compatibility - 0.5 * distance + catalog_prior_weight * log_prior
    ) / catalog_temperature
    logits = logits.masked_fill(~valid_candidates, float("-inf"))
    catalog_loss = F.cross_entropy(
        logits,
        torch.zeros(logits.size(0), dtype=torch.long, device=logits.device),
    )

    population_loss = target_compatibility.sum() * 0.0
    prior = None
    log_q = None
    if population_loss_weight > 0.0:
        prior = torch.softmax(
            log_prior.masked_fill(~valid_candidates, float("-inf")), dim=-1
        )
        log_q = torch.log_softmax(logits, dim=-1)
        safe_log_prior = torch.where(
            valid_candidates,
            prior.clamp_min(1e-8).log(),
            torch.zeros_like(prior),
        )
        population_loss = (
            (
                prior
                * (safe_log_prior - log_q).masked_fill(~valid_candidates, 0.0)
            )
            .sum(dim=-1)
            .mean()
        )
    if logger.isEnabledFor(logging.DEBUG) and (
        not bool(torch.isfinite(catalog_loss))
        or not bool(torch.isfinite(population_loss))
    ):
        diagnostics = {
            "compatibility": compatibility,
            "distance": distance,
            "valid_log_prior": log_prior[valid_candidates],
            "valid_logits": logits[valid_candidates],
        }
        if prior is not None and log_q is not None:
            diagnostics["valid_prior_probability"] = prior[valid_candidates]
            diagnostics["valid_log_q"] = log_q[valid_candidates]
        _debug_catalog_anomaly(
            branch="prefix",
            tensors=diagnostics,
            valid_candidates=valid_candidates,
        )
    return catalog_loss, population_loss


def prefix_time_suffix_latent_flow_loss(
    model: torch.nn.Module,
    clean_codes: torch.Tensor,
    time_tokens: torch.Tensor,
    valid_mask: torch.BoolTensor,
    codebooks: torch.Tensor,
    tail_mean: torch.Tensor,
    tail_std: torch.Tensor,
    *,
    tail_whitener: torch.Tensor | None = None,
    tail_dewhitener: torch.Tensor | None = None,
    lognormal_t: bool = True,
    p_mean: float = -0.5,
    p_std: float = 1.2,
    factorized_sum_consistency_weight: float = 0.0,
    catalog_candidates: CatalogTailCandidates | None = None,
    catalog_loss_weight: float = 0.0,
    catalog_negative_count: int = 8,
    catalog_temperature: float = 1.0,
    candidate_bank: CatalogCandidateBank | None = None,
    endpoint_loss_weight: float = 0.0,
    population_loss_weight: float = 0.0,
    catalog_prior_weight: float = 0.25,
    proposal_prefix_codes: torch.Tensor | None = None,
    city_id: torch.LongTensor | None = None,
    hop_distance_km: torch.Tensor | None = None,
) -> LatentSuffixFlowObjectiveOutput:
    core = getattr(model, "module", model)
    core = getattr(core, "_orig_mod", core)
    if not isinstance(core, PrefixTimeSuffixLatentFlowRealizer):
        raise TypeError("model does not wrap a latent suffix-flow realizer")
    if clean_codes.ndim != 3:
        raise ValueError("clean_codes must have shape [B, L, M]")
    if valid_mask.shape != clean_codes.shape[:2]:
        raise ValueError("valid_mask must align with clean_codes")
    if time_tokens.shape != valid_mask.shape:
        raise ValueError("time_tokens and valid_mask must align")
    if proposal_prefix_codes is not None:
        expected_proposal_shape = (*valid_mask.shape, core.prefix_depth)
        if proposal_prefix_codes.shape != expected_proposal_shape:
            raise ValueError(
                "proposal_prefix_codes must have shape [B, L, prefix_depth]"
            )
    if core.tail_representation == "factorized":
        expected_stats_shape = (core.suffix_depth, core.latent_dim)
    else:
        expected_stats_shape = (core.latent_dim,)
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
    factorized = core.tail_representation == "factorized"
    if tail_whitener is not None or tail_dewhitener is not None:
        if factorized:
            raise ValueError(
                "tail whitening is supported only for summed tails"
            )
        matrix_shape = (core.latent_dim, core.latent_dim)
        if (
            tail_whitener is None
            or tail_dewhitener is None
            or tuple(tail_whitener.shape) != matrix_shape
            or tuple(tail_dewhitener.shape) != matrix_shape
        ):
            raise ValueError("tail whiteners must both have shape [D, D]")
    if endpoint_loss_weight < 0.0 or population_loss_weight < 0.0:
        raise ValueError("auxiliary loss weights must be non-negative")
    if catalog_prior_weight < 0.0:
        raise ValueError("catalog_prior_weight must be non-negative")
    if catalog_temperature <= 0.0:
        raise ValueError("catalog_temperature must be positive")
    if core.catalog_decoder_mode == "calibrated":
        if factorized:
            raise ValueError(
                "factorized tails are incompatible with calibrated catalog "
                "decoding"
            )
        if catalog_loss_weight <= 0.0:
            raise ValueError(
                "catalog_loss_weight must be positive for calibrated "
                "catalog decoding"
            )
        if candidate_bank is None:
            raise ValueError(
                "candidate_bank is required for calibrated catalog decoding"
            )

    if factorized:
        clean_tail_raw = suffix_tail_level_embeddings(
            clean_codes,
            codebooks,
            prefix_depth=core.prefix_depth,
        ).to(dtype=tail_mean.dtype)
        clean_tail_levels = (clean_tail_raw - tail_mean) / tail_std
        clean_tail = clean_tail_levels.reshape(
            clean_codes.size(0), clean_codes.size(1), -1
        )
    else:
        clean_tail_raw = sid_level_sum_embeddings(
            clean_codes,
            codebooks,
            level_start=core.target_level_start,
        ).to(dtype=tail_mean.dtype)
        clean_tail = normalize_tail(
            clean_tail_raw,
            tail_mean,
            tail_std,
            tail_whitener=tail_whitener,
        )
    mask = valid_mask[..., None]
    clean_tail = clean_tail.masked_fill(~mask, 0.0)
    noise = torch.randn_like(clean_tail).masked_fill(~mask, 0.0)
    if lognormal_t:
        timesteps = sample_logit_normal_timesteps(
            clean_codes.size(0),
            clean_codes.device,
            p_mean=p_mean,
            p_std=p_std,
        )
    else:
        timesteps = sample_uniform_timesteps(
            clean_codes.size(0), clean_codes.device
        )
    noisy_tail = linear_interpolate(clean_tail, noise, timesteps)
    noisy_tail = noisy_tail.masked_fill(~mask, 0.0)
    conditioning = core.encode_conditioning(
        (
            clean_codes[..., : core.prefix_depth].long()
            if proposal_prefix_codes is None
            else proposal_prefix_codes.long()
        ),
        time_tokens.long(),
        valid_mask,
        city_id=city_id,
        hop_distance_km=hop_distance_km,
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
    flow_loss = per_event[valid_mask].mean()
    endpoint_loss = clean_tail.sum() * 0.0
    if endpoint_loss_weight > 0.0:
        endpoint_loss = F.smooth_l1_loss(
            x0_pred[valid_mask],
            clean_tail[valid_mask],
        )
    sum_consistency_loss = clean_tail.sum() * 0.0
    if factorized:
        predicted_levels = x0_pred.reshape(
            clean_codes.size(0),
            clean_codes.size(1),
            core.suffix_depth,
            core.latent_dim,
        )
        target_levels = clean_tail.reshape(
            clean_codes.size(0),
            clean_codes.size(1),
            core.suffix_depth,
            core.latent_dim,
        )
        sum_consistency_loss = (
            (predicted_levels.sum(dim=2) - target_levels.sum(dim=2))
            .square()
            .mean(dim=-1)[valid_mask]
            .mean()
        )
        predicted_tail_raw = (predicted_levels * tail_std + tail_mean).sum(
            dim=2
        )
        target_tail_raw = clean_tail_raw.sum(dim=2)
    else:
        predicted_tail_raw = denormalize_tail(
            x0_pred,
            tail_mean,
            tail_std,
            tail_dewhitener=tail_dewhitener,
        )
        target_tail_raw = clean_tail_raw
    catalog_loss = clean_tail.sum() * 0.0
    population_loss = clean_tail.sum() * 0.0
    if catalog_loss_weight < 0.0:
        raise ValueError("catalog_loss_weight must be non-negative")
    if catalog_loss_weight > 0.0:
        if candidate_bank is not None:
            if candidate_bank.prefix_depth == 0:
                catalog_loss, population_loss = _global_catalog_losses(
                    core,
                    conditioning,
                    x0_pred,
                    clean_codes,
                    valid_mask,
                    codebooks,
                    candidate_bank,
                    tail_mean,
                    tail_std,
                    tail_whitener=tail_whitener,
                    catalog_prior_weight=catalog_prior_weight,
                    catalog_temperature=catalog_temperature,
                    population_loss_weight=population_loss_weight,
                )
            else:
                catalog_loss, population_loss = _prefix_catalog_losses(
                    core,
                    conditioning,
                    x0_pred,
                    clean_codes,
                    valid_mask,
                    codebooks,
                    candidate_bank,
                    tail_mean,
                    tail_std,
                    tail_whitener=tail_whitener,
                    catalog_prior_weight=catalog_prior_weight,
                    catalog_temperature=catalog_temperature,
                    population_loss_weight=population_loss_weight,
                )
        elif catalog_candidates is None:
            raise ValueError(
                "candidate_bank or catalog_candidates is required when "
                "catalog loss is enabled"
            )
        else:
            catalog_loss = catalog_aware_tail_ranking_loss(
                predicted_tail_raw,
                target_tail_raw,
                clean_codes,
                valid_mask,
                catalog_candidates,
                num_negatives=catalog_negative_count,
                temperature=catalog_temperature,
            )
    loss = (
        flow_loss
        + endpoint_loss_weight * endpoint_loss
        + factorized_sum_consistency_weight * sum_consistency_loss
        + catalog_loss_weight * catalog_loss
        + population_loss_weight * population_loss
    )
    return LatentSuffixFlowObjectiveOutput(
        loss=loss,
        conditioning=conditioning,
        clean_tail=clean_tail.detach(),
        noisy_tail=noisy_tail.detach(),
        target_velocity=target_velocity.detach(),
        predicted_velocity=predicted_velocity.detach(),
        x0_pred=x0_pred,
        timesteps=timesteps.detach(),
        flow_loss=flow_loss,
        endpoint_loss=endpoint_loss,
        sum_consistency_loss=sum_consistency_loss,
        catalog_loss=catalog_loss,
        population_loss=population_loss,
    )


__all__ = [
    "CatalogCandidateBank",
    "CatalogCandidateBatch",
    "CatalogTailCandidates",
    "LatentSuffixFlowObjectiveOutput",
    "catalog_aware_tail_ranking_loss",
    "compute_factorized_tail_stats",
    "compute_tail_stats",
    "compute_tail_whitening_stats",
    "denormalize_tail",
    "normalize_tail",
    "prefix_time_suffix_latent_flow_loss",
    "sid_level_embeddings",
    "sid_level_sum_embeddings",
    "suffix_tail_embeddings",
    "suffix_tail_level_embeddings",
]
