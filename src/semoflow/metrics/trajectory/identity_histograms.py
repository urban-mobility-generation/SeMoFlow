from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch


def identity_histogram_contributions(
    tokens: torch.Tensor, *, pad_id: int
) -> dict[str, np.ndarray]:
    n_rows, width = tokens.shape
    unique = np.zeros((n_rows, width + 1), dtype=np.int32)
    gaps = np.zeros((n_rows, width + 1), dtype=np.int32)
    multiplicity = np.zeros((n_rows, width + 1), dtype=np.int32)
    for row_index, raw_row in enumerate(tokens.detach().cpu().tolist()):
        row = [int(value) for value in raw_row if int(value) != pad_id]
        counts = Counter(row)
        unique[row_index, min(len(counts), width)] = 1
        for count in counts.values():
            multiplicity[row_index, min(count, width)] += 1
        previous: dict[int, int] = {}
        for position, value in enumerate(row):
            if value in previous:
                gap = position - previous[value]
                gaps[row_index, min(gap, width)] += 1
            previous[value] = position
    return {
        "unique_location_count_jsd": unique,
        "return_gap_distribution_jsd": gaps,
        "multiplicity_distribution_jsd": multiplicity,
    }


def normalise_histograms(counts: np.ndarray) -> np.ndarray:
    values = counts.astype(np.float64, copy=False)
    totals = values.sum(axis=-1, keepdims=True)
    return np.divide(
        values,
        totals,
        out=np.zeros_like(values),
        where=totals > 0,
    )


def rowwise_jsd_counts(real: np.ndarray, generated: np.ndarray) -> np.ndarray:
    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if real.shape != generated.shape:
        raise ValueError("histogram inputs must have the same shape")
    if not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError("histogram inputs must contain only finite values")
    if np.any(real < 0.0) or np.any(generated < 0.0):
        raise ValueError("histogram inputs must be non-negative")
    real_totals = real.sum(axis=-1)
    generated_totals = generated.sum(axis=-1)
    p = normalise_histograms(real)
    q = normalise_histograms(generated)
    midpoint = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        p_ratio = np.divide(
            p,
            midpoint,
            out=np.ones_like(p),
            where=midpoint > 0,
        )
        q_ratio = np.divide(
            q,
            midpoint,
            out=np.ones_like(q),
            where=midpoint > 0,
        )
        kl_p = np.sum(np.where(p > 0.0, p * np.log(p_ratio), 0.0), axis=-1)
        kl_q = np.sum(np.where(q > 0.0, q * np.log(q_ratio), 0.0), axis=-1)
    result = 0.5 * (kl_p + kl_q)
    valid = (real_totals > 0.0) & (generated_totals > 0.0)
    result = np.where(valid, np.clip(result, 0.0, np.log(2.0)), np.nan)
    return result


def torch_rowwise_jsd_counts(
    real: torch.Tensor, generated: torch.Tensor
) -> torch.Tensor:
    real = real.to(dtype=torch.float64)
    generated = generated.to(dtype=torch.float64)
    real_totals = real.sum(dim=-1)
    generated_totals = generated.sum(dim=-1)
    p = real / real_totals.clamp_min(1).unsqueeze(-1)
    q = generated / generated_totals.clamp_min(1).unsqueeze(-1)
    midpoint = (p + q) / 2
    kl_p = torch.where(
        p > 0,
        p * torch.log(p / midpoint.clamp_min(torch.finfo(p.dtype).tiny)),
        0,
    ).sum(dim=-1)
    kl_q = torch.where(
        q > 0,
        q * torch.log(q / midpoint.clamp_min(torch.finfo(q.dtype).tiny)),
        0,
    ).sum(dim=-1)
    result = (0.5 * (kl_p + kl_q)).clamp(0.0, np.log(2.0))
    valid = (real_totals > 0) & (generated_totals > 0)
    return torch.where(valid, result, torch.nan)


def cluster_bootstrap_histogram_jsd(
    real_contributions: np.ndarray,
    generated_contributions: np.ndarray,
    *,
    cluster_labels: np.ndarray,
    cluster_weights: np.ndarray,
    protocol: dict[str, Any],
    compute_device: torch.device | str | None = None,
) -> dict[str, Any]:
    n_clusters = int(protocol["n_clusters"])
    real_by_cluster = np.zeros(
        (n_clusters, real_contributions.shape[1]), dtype=np.int32
    )
    generated_by_cluster = np.zeros_like(real_by_cluster)
    np.add.at(real_by_cluster, cluster_labels, real_contributions)
    np.add.at(generated_by_cluster, cluster_labels, generated_contributions)
    point = float(
        rowwise_jsd_counts(
            real_contributions.sum(axis=0, keepdims=True),
            generated_contributions.sum(axis=0, keepdims=True),
        )[0]
    )
    if compute_device is None:
        draws = rowwise_jsd_counts(
            cluster_weights @ real_by_cluster,
            cluster_weights @ generated_by_cluster,
        )
    else:
        device = torch.device(compute_device)
        weights = torch.as_tensor(
            cluster_weights, dtype=torch.float64, device=device
        )
        real_clusters = torch.as_tensor(
            real_by_cluster, dtype=torch.float64, device=device
        )
        generated_clusters = torch.as_tensor(
            generated_by_cluster, dtype=torch.float64, device=device
        )
        draws = (
            torch_rowwise_jsd_counts(
                weights @ real_clusters,
                weights @ generated_clusters,
            )
            .cpu()
            .numpy()
        )
    return {
        "point_estimate": point,
        "bootstrap_95ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "protocol": {
            "resampling_unit": protocol["resampling_unit"],
            "n_rows": protocol["n_rows"],
            "n_clusters": protocol["n_clusters"],
            "bootstrap_resamples": protocol["bootstrap_resamples"],
            **(
                {"bootstrap_seed": protocol["bootstrap_seed"]}
                if "bootstrap_seed" in protocol
                else {}
            ),
            "interval_target": protocol.get(
                "interval_target",
                "resampled_panel_estimator_distribution",
            ),
            "fixed_panel_point_ci": protocol.get(
                "fixed_panel_point_ci", False
            ),
        },
    }


__all__ = [
    "cluster_bootstrap_histogram_jsd",
    "identity_histogram_contributions",
    "normalise_histograms",
    "rowwise_jsd_counts",
    "torch_rowwise_jsd_counts",
]
