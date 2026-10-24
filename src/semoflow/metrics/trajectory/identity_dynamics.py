from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from scipy import sparse

from semoflow.metrics.trajectory.identity_histograms import (
    cluster_bootstrap_histogram_jsd,
    identity_histogram_contributions,
    rowwise_jsd_counts,
    torch_rowwise_jsd_counts,
)

_QS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
_BOOTSTRAP_SEED = 20260714
_BOOTSTRAP_RESAMPLES = 1000
_NGRAM_MAX_VOCAB = 500
_FRAC_KEYS = (
    "adjacent_repeat_rate",
    "revisit_mass",
    "anchor_occurrence_fraction",
    "top_poi_share",
    "top_two_poi_share",
)


def _cluster_labels(
    cluster_ids: Sequence[str] | None, n_rows: int, *, rows: str
) -> tuple[np.ndarray, str, int]:
    """Dense cluster labels, resampling unit, and cluster count."""
    if cluster_ids is None:
        labels = np.arange(n_rows, dtype=np.int64)
        unit = "trajectory_row"
    else:
        if len(cluster_ids) != n_rows:
            raise ValueError(
                f"cluster_ids must align with {rows}: "
                f"{len(cluster_ids)} != {n_rows}"
            )
        _, labels = np.unique(
            np.asarray(cluster_ids, dtype=str), return_inverse=True
        )
        unit = "device_cluster"
    n_clusters = int(labels.max()) + 1 if n_rows else 0
    return labels, unit, n_clusters


def _cluster_weights(
    rng: np.random.Generator,
    size: int,
    n_clusters: int,
    *,
    sample_dtype: type[np.integer] = np.int32,
) -> np.ndarray:
    """Multiplicity of each cluster in ``size`` bootstrap resamples."""
    sampled = rng.integers(
        0, n_clusters, size=(size, n_clusters), dtype=sample_dtype
    )
    weights = np.zeros((size, n_clusters), dtype=np.int32)
    np.add.at(
        weights,
        (np.repeat(np.arange(size), n_clusters), sampled.ravel()),
        1,
    )
    return weights


def _ci95(draws: np.ndarray) -> list[float]:
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def _cluster_count_matrix(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    labels: np.ndarray,
    observed: np.ndarray,
    n_clusters: int,
) -> sparse.csr_matrix:
    """Cluster-by-support counts of the valid ``values``."""
    row_labels = np.broadcast_to(labels[:, None], values.shape)
    columns = np.searchsorted(observed, values[valid])
    return sparse.coo_matrix(
        (
            np.ones(len(columns), dtype=np.int32),
            (row_labels[valid], columns),
        ),
        shape=(n_clusters, len(observed)),
    ).tocsr()


def _to_sparse_tensor(
    matrix: sparse.csr_matrix, device: torch.device
) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.from_numpy(
        np.stack([coo.row, coo.col]).astype(np.int64, copy=False)
    )
    values = torch.from_numpy(coo.data.astype(np.float64, copy=False))
    with torch.sparse.check_sparse_tensor_invariants():
        return torch.sparse_coo_tensor(
            indices,
            values,
            coo.shape,
            dtype=torch.float64,
            device=device,
        ).coalesce()


def _coverage(left_counts: np.ndarray, right_counts: np.ndarray) -> np.ndarray:
    real_seen = left_counts > 0
    denominators = real_seen.sum(axis=1)
    numerators = (real_seen & (right_counts > 0)).sum(axis=1)
    return np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators, dtype=np.float64),
        where=denominators > 0,
    )


def _torch_coverage(
    left_counts: torch.Tensor, right_counts: torch.Tensor
) -> np.ndarray:
    real_seen = left_counts > 0
    denominators = real_seen.sum(dim=1).clamp_min(1).to(dtype=torch.float64)
    numerators = (real_seen & (right_counts > 0)).sum(dim=1)
    return (numerators.to(dtype=torch.float64) / denominators).cpu().numpy()


def _protocol(
    unit: str, n_rows: int, n_clusters: int, **extra: Any
) -> dict[str, Any]:
    """Provenance of a cluster-bootstrap interval."""
    return {
        "resampling_unit": unit,
        "n_rows": n_rows,
        "n_clusters": n_clusters,
        "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": _BOOTSTRAP_SEED,
        **extra,
        "interval_target": "resampled_panel_estimator_distribution",
        "fixed_panel_point_ci": False,
    }


def _sparse_cluster_bootstrap(
    real_by_cluster: sparse.csr_matrix,
    generated_by_cluster: sparse.csr_matrix,
    *,
    compute_device: torch.device | str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap draws of the pooled JSD and real-support coverage."""
    n_clusters, n_support = real_by_cluster.shape
    real_t = real_by_cluster.T.tocsr()
    generated_t = generated_by_cluster.T.tocsr()
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    jsd_parts: list[np.ndarray] = []
    coverage_parts: list[np.ndarray] = []
    if compute_device is not None:
        device = torch.device(compute_device)
        real_sparse = _to_sparse_tensor(real_t, device)
        generated_sparse = _to_sparse_tensor(generated_t, device)
        chunk_size = max(1, min(128, 8_000_000 // max(n_support, 1)))
    else:
        chunk_size = max(1, min(32, 2_000_000 // max(n_support, 1)))
    for start in range(0, _BOOTSTRAP_RESAMPLES, chunk_size):
        size = min(chunk_size, _BOOTSTRAP_RESAMPLES - start)
        weights = _cluster_weights(rng, size, n_clusters)
        if compute_device is None:
            real_draw = np.asarray((real_t @ weights.T).T)
            generated_draw = np.asarray((generated_t @ weights.T).T)
            jsd_parts.append(rowwise_jsd_counts(real_draw, generated_draw))
            coverage_parts.append(_coverage(real_draw, generated_draw))
            continue
        weight_tensor = torch.as_tensor(
            weights, dtype=torch.float64, device=device
        )
        real_draw = torch.sparse.mm(real_sparse, weight_tensor.T).T
        generated_draw = torch.sparse.mm(generated_sparse, weight_tensor.T).T
        jsd_parts.append(
            torch_rowwise_jsd_counts(real_draw, generated_draw).cpu().numpy()
        )
        coverage_parts.append(_torch_coverage(real_draw, generated_draw))
    return np.concatenate(jsd_parts), np.concatenate(coverage_parts)


def _sparse_support_bootstrap(
    real: tuple[np.ndarray, np.ndarray],
    generated: tuple[np.ndarray, np.ndarray],
    *,
    labels: np.ndarray,
    n_clusters: int,
    compute_device: torch.device | str | None,
) -> tuple[tuple[float, np.ndarray], tuple[float, np.ndarray]]:
    """``(point, draws)`` of the pooled JSD and of real-support coverage.

    ``real`` and ``generated`` are ``(values, valid)`` matrices whose rows
    align with the cluster ``labels``.
    """
    observed = np.unique(
        np.concatenate([values[valid] for values, valid in (real, generated)])
    )
    real_by_cluster, generated_by_cluster = (
        _cluster_count_matrix(
            values,
            valid,
            labels=labels,
            observed=observed,
            n_clusters=n_clusters,
        )
        for values, valid in (real, generated)
    )
    real_total = np.asarray(real_by_cluster.sum(axis=0))
    generated_total = np.asarray(generated_by_cluster.sum(axis=0))
    point_jsd = float(rowwise_jsd_counts(real_total, generated_total)[0])
    point_coverage = float(_coverage(real_total, generated_total)[0])
    if n_clusters == 0:
        jsd_draws = np.asarray([point_jsd])
        coverage_draws = np.asarray([point_coverage])
    else:
        jsd_draws, coverage_draws = _sparse_cluster_bootstrap(
            real_by_cluster,
            generated_by_cluster,
            compute_device=compute_device,
        )
    return (point_jsd, jsd_draws), (point_coverage, coverage_draws)


def cluster_histogram_jsd_bootstrap(
    real_contributions: np.ndarray,
    generated_contributions: np.ndarray,
    *,
    cluster_ids: Sequence[str] | None,
    compute_device: torch.device | str | None = None,
) -> dict[str, Any]:
    real = np.asarray(real_contributions)
    generated = np.asarray(generated_contributions)
    if real.ndim != 2 or generated.shape != real.shape:
        raise ValueError("histogram contributions must be aligned matrices")
    if np.any(real < 0) or np.any(generated < 0):
        raise ValueError("histogram contributions must be non-negative")
    n_rows = real.shape[0]
    labels, unit, n_clusters = _cluster_labels(
        cluster_ids, n_rows, rows="histogram rows"
    )
    protocol = _protocol(unit, n_rows, n_clusters)
    if n_clusters == 0:
        point = float(
            rowwise_jsd_counts(
                real.sum(axis=0, keepdims=True),
                generated.sum(axis=0, keepdims=True),
            )[0]
        )
        return {
            "point_estimate": point,
            "bootstrap_95ci": [point, point],
            "protocol": protocol,
        }
    return cluster_bootstrap_histogram_jsd(
        real,
        generated,
        cluster_labels=labels,
        cluster_weights=_cluster_weights(
            np.random.default_rng(_BOOTSTRAP_SEED),
            _BOOTSTRAP_RESAMPLES,
            n_clusters,
        ),
        protocol=protocol,
        compute_device=compute_device,
    )


def cluster_sparse_token_distribution_bootstrap(
    real: torch.Tensor,
    generated: torch.Tensor,
    *,
    vocab_size: int,
    pad_id: int,
    cluster_ids: Sequence[str] | None,
    compute_device: torch.device | str | None = None,
) -> dict[str, Any]:
    if real.ndim != 2 or generated.shape != real.shape:
        raise ValueError("sparse token distributions require aligned matrices")
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    n_rows = real.shape[0]
    labels, unit, n_clusters = _cluster_labels(
        cluster_ids, n_rows, rows="sparse token rows"
    )

    def _valid_tokens(tokens: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        values = tokens.long().cpu().numpy()
        return values, (
            (values != pad_id) & (values >= 0) & (values < vocab_size)
        )

    (point_jsd, jsd_draws), (point_coverage, coverage_draws) = (
        _sparse_support_bootstrap(
            _valid_tokens(real),
            _valid_tokens(generated),
            labels=labels,
            n_clusters=n_clusters,
            compute_device=compute_device,
        )
    )
    protocol = _protocol(
        unit, n_rows, n_clusters, estimator="exact_sparse_cluster_bootstrap_v1"
    )
    return {
        "poi_frequency_jsd": {
            "point_estimate": point_jsd,
            "bootstrap_95ci": _ci95(jsd_draws),
            "protocol": protocol,
        },
        "poi_coverage": {
            "point_estimate": point_coverage,
            "bootstrap_95ci": _ci95(coverage_draws),
            "protocol": protocol,
        },
    }


def cluster_sparse_ngram_jsd_bootstrap(
    real: torch.Tensor,
    generated: torch.Tensor,
    *,
    vocab_size: int,
    n: int,
    pad_id: int,
    cluster_ids: Sequence[str] | None,
    compute_device: torch.device | str | None = None,
) -> dict[str, Any] | None:
    if vocab_size > _NGRAM_MAX_VOCAB:
        return None
    if real.ndim != 2 or generated.shape != real.shape:
        raise ValueError("n-gram distributions require aligned matrices")
    if n < 1:
        raise ValueError("n must be positive")
    n_rows = real.size(0)
    labels, unit, n_clusters = _cluster_labels(
        cluster_ids, n_rows, rows="n-gram rows"
    )

    def _hashed_windows(tokens: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        values = tokens.long().cpu()
        if values.size(1) < n:
            return (
                np.empty((n_rows, 0), dtype=np.int64),
                np.empty((n_rows, 0), dtype=bool),
            )
        windows = values.unfold(1, n, 1)
        valid = (windows != pad_id).all(dim=-1)
        weights = torch.tensor(
            [vocab_size ** (n - 1 - index) for index in range(n)],
            dtype=torch.int64,
        )
        hashes = (windows * weights).sum(dim=-1)
        return hashes.numpy(), valid.numpy()

    (point, draws), _ = _sparse_support_bootstrap(
        _hashed_windows(real),
        _hashed_windows(generated),
        labels=labels,
        n_clusters=n_clusters,
        compute_device=compute_device,
    )
    return {
        "point_estimate": point,
        "bootstrap_95ci": _ci95(draws),
        "protocol": _protocol(
            unit,
            n_rows,
            n_clusters,
            estimator="exact_sparse_ngram_cluster_bootstrap_v1",
            n=n,
        ),
    }


def trajectory_identity_metrics(
    tokens: torch.Tensor, *, pad_id: int = 0
) -> list[dict[str, float]]:
    out = []
    for row in tokens.detach().cpu().tolist():
        p = [int(x) for x in row if int(x) != pad_id]
        L = len(p)
        if not L:
            out.append(
                {
                    k: 0.0
                    for k in (
                        *_FRAC_KEYS,
                        "unique_poi_ratio",
                        "anchor_count",
                        "non_adjacent_return_event_count",
                        "non_adjacent_return_rate",
                        "non_adjacent_return_rate_per_decision",
                        "non_adjacent_return_identity_count",
                        "any_repeated_poi",
                        "any_non_adjacent_return",
                    )
                }
            )
            continue
        c = Counter(p)
        counts = sorted(c.values(), reverse=True)
        anchors = {v for v, n in c.items() if n >= 2}
        seen: set[int] = set()
        return_ids: set[int] = set()
        nonadj = 0
        for i, v in enumerate(p):
            if i and v in seen and v != p[i - 1]:
                nonadj += 1
                return_ids.add(v)
            seen.add(v)
        out.append(
            {
                "adjacent_repeat_rate": sum(
                    p[i] == p[i - 1] for i in range(1, L)
                )
                / max(L - 1, 1),
                "revisit_mass": (L - len(c)) / L,
                "unique_poi_ratio": len(c) / L,
                "anchor_count": float(len(anchors)),
                "anchor_occurrence_fraction": sum(c[v] for v in anchors) / L,
                "top_poi_share": counts[0] / L,
                "top_two_poi_share": sum(counts[:2]) / L,
                "non_adjacent_return_event_count": float(nonadj),
                "non_adjacent_return_rate": nonadj / L,
                "non_adjacent_return_rate_per_decision": (
                    nonadj / max(L - 1, 1)
                ),
                "non_adjacent_return_identity_count": float(len(return_ids)),
                "any_repeated_poi": float(bool(anchors)),
                "any_non_adjacent_return": float(bool(nonadj)),
            }
        )
    return out


def _summary(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    x = np.array([r[key] for r in rows], dtype=float)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        **{f"p{int(q * 100)}": float(np.quantile(x, q)) for q in _QS},
    }


def identity_dynamics_comparison(
    real: torch.Tensor,
    generated: torch.Tensor,
    *,
    pad_id: int = 0,
    kind: str = "poi",
    cluster_ids: Sequence[str] | None = None,
    compute_device: torch.device | str | None = None,
) -> dict[str, Any]:
    r, g = (
        trajectory_identity_metrics(real, pad_id=pad_id),
        trajectory_identity_metrics(generated, pad_id=pad_id),
    )
    keys = tuple(r[0])
    n_rows = len(r)
    cluster_labels, resampling_unit, n_clusters = _cluster_labels(
        cluster_ids, n_rows, rows="trajectory rows"
    )

    def adjacent_counts(
        toks: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        prev, nxt = toks[:, :-1], toks[:, 1:]
        valid = (prev != pad_id) & (nxt != pad_id)
        return (
            ((prev == nxt) & valid).sum(dim=1).cpu().numpy(),
            valid.sum(dim=1).cpu().numpy(),
        )

    r_adj, r_trans = adjacent_counts(real)
    g_adj, g_trans = adjacent_counts(generated)

    def side(
        rows: list[dict[str, float]], toks: torch.Tensor
    ) -> dict[str, Any]:
        adj, trans = adjacent_counts(toks)
        adjacent_micro = float(adj.sum() / trans.sum()) if trans.sum() else 0.0
        o: dict[str, Any] = {key: _summary(rows, key) for key in keys}
        o["adjacent_repeat_rate_micro"] = adjacent_micro
        o["adjacent_repeat_rate_macro"] = o.pop("adjacent_repeat_rate")
        o["threshold_fractions"] = {
            key: {
                str(t): float(np.mean([x[key] >= t for x in rows]))
                for t in (0.1, 0.2, 0.3, 0.5)
            }
            for key in _FRAC_KEYS
        }
        return o

    result: dict[str, Any] = {
        "definition_version": 2,
        kind: {"real": side(r, real), "generated": side(g, generated)},
        "definitions": {
            "adjacent_repeat_rate_micro": "total adjacent equalities divided by total valid adjacent transitions",
            "adjacent_repeat_rate_macro": "mean of per-trajectory adjacent-repeat rates",
            "revisit_mass": "(L-U)/L; unique_poi_ratio=U/L and their sum is one",
            "anchor_count": "M=|{v: count(v)>=2}|",
            "anchor_occurrence_fraction": "sum_{v:count(v)>=2} count(v)/L",
            "top_poi_share": "max_v count(v)/L; top_two_poi_share is the two largest counts divided by L",
            "non_adjacent_return_event_count": "count of i where p_i appeared earlier and p_i != p_(i-1)",
            "non_adjacent_return_identity_count": "number of identities with at least one non-adjacent return event",
            "non_adjacent_return_rate": "non-adjacent return events divided by L (events); NOT comparable with adjacent_repeat_rate, which divides by L-1",
            "non_adjacent_return_rate_per_decision": "non-adjacent return events divided by L-1 (decisions); the denominator that makes it comparable with adjacent_repeat_rate_macro",
        },
        "protocol": {
            "pad_id": pad_id,
            "bootstrap_seed": _BOOTSTRAP_SEED,
            "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
            "resampling_unit": resampling_unit,
            "n_rows": n_rows,
            "n_clusters": n_clusters,
        },
    }
    cluster_weights = _cluster_weights(
        np.random.default_rng(_BOOTSTRAP_SEED),
        _BOOTSTRAP_RESAMPLES,
        n_clusters,
        sample_dtype=np.int64,
    )

    def _cluster_sum(values: np.ndarray) -> np.ndarray:
        return np.bincount(
            cluster_labels, weights=values, minlength=n_clusters
        )

    diff: dict[str, dict[str, float | list[float]]] = {}
    weights_device = (
        torch.as_tensor(
            cluster_weights,
            dtype=torch.float64,
            device=torch.device(compute_device),
        )
        if compute_device is not None
        else None
    )

    def _weighted_draw(values: np.ndarray) -> np.ndarray:
        clustered = _cluster_sum(values)
        if weights_device is None:
            return cluster_weights @ clustered
        return (
            (
                weights_device
                @ torch.as_tensor(
                    clustered,
                    dtype=torch.float64,
                    device=weights_device.device,
                )
            )
            .cpu()
            .numpy()
        )

    micro_draw = _weighted_draw(g_adj) / np.maximum(
        _weighted_draw(g_trans), 1
    ) - _weighted_draw(r_adj) / np.maximum(_weighted_draw(r_trans), 1)
    diff["adjacent_repeat_rate_micro"] = {
        "generated_minus_real": float(
            g_adj.sum() / max(g_trans.sum(), 1)
            - r_adj.sum() / max(r_trans.sum(), 1)
        ),
        "bootstrap_95ci": _ci95(micro_draw),
    }
    for key in keys:
        rv, gv = np.array([x[key] for x in r]), np.array([x[key] for x in g])
        d = _weighted_draw(gv - rv) / np.maximum(
            _weighted_draw(np.ones(n_rows, dtype=float)), 1
        )
        diff[
            "adjacent_repeat_rate_macro"
            if key == "adjacent_repeat_rate"
            else key
        ] = {
            "generated_minus_real": float(gv.mean() - rv.mean()),
            "bootstrap_95ci": _ci95(d),
        }
    result[kind]["generated_minus_real"] = diff
    histogram_contributions_real = identity_histogram_contributions(
        real, pad_id=pad_id
    )
    histogram_contributions_gen = identity_histogram_contributions(
        generated, pad_id=pad_id
    )
    result[kind]["distribution_distances"] = {
        name: cluster_bootstrap_histogram_jsd(
            histogram_contributions_real[name],
            histogram_contributions_gen[name],
            cluster_labels=cluster_labels,
            cluster_weights=cluster_weights,
            protocol=result["protocol"],
            compute_device=compute_device,
        )
        for name in histogram_contributions_real
    }
    return result


def category_identity_dynamics_comparison(
    real: torch.Tensor,
    generated: torch.Tensor,
    *,
    pad_id: int = 0,
    cluster_ids: Sequence[str] | None = None,
    compute_device: torch.device | str | None = None,
) -> dict[str, Any]:
    raw = identity_dynamics_comparison(
        real,
        generated,
        pad_id=pad_id,
        kind="category",
        cluster_ids=cluster_ids,
        compute_device=compute_device,
    )["category"]
    rename = {
        "adjacent_repeat_rate_micro": "category_adjacent_repeat_rate_micro",
        "adjacent_repeat_rate_macro": "category_adjacent_repeat_rate_macro",
        "revisit_mass": "category_revisit_mass",
        "unique_poi_ratio": "unique_category_ratio",
        "anchor_count": "repeated_category_count",
        "anchor_occurrence_fraction": "repeated_category_occurrence_fraction",
        "any_repeated_poi": "any_repeated_category",
        "top_poi_share": "top_category_share",
        "top_two_poi_share": "top_two_category_share",
    }
    for side in ("real", "generated"):
        raw[side] = {rename.get(k, k): v for k, v in raw[side].items()}
    raw["generated_minus_real"] = {
        rename.get(k, k): v for k, v in raw["generated_minus_real"].items()
    }
    distribution_rename = {
        "unique_location_count_jsd": "unique_category_count_jsd",
        "return_gap_distribution_jsd": "category_return_gap_distribution_jsd",
        "multiplicity_distribution_jsd": "category_multiplicity_distribution_jsd",
    }
    raw["distribution_distances"] = {
        distribution_rename.get(k, k): v
        for k, v in raw["distribution_distances"].items()
    }
    return raw
