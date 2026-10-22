from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from scipy import sparse

from semoflow.metrics.trajectory.identity_histograms import (
    _torch_rowwise_jsd_counts,
)
from semoflow.metrics.trajectory.identity_histograms import (
    cluster_bootstrap_histogram_jsd as _cluster_bootstrap_histogram_jsd,
)
from semoflow.metrics.trajectory.identity_histograms import (
    identity_histogram_contributions as _identity_histogram_contributions,
)
from semoflow.metrics.trajectory.identity_histograms import (
    rowwise_jsd_counts as _rowwise_jsd_counts,
)

_QS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
_FRAC_KEYS = (
    "adjacent_repeat_rate",
    "revisit_mass",
    "anchor_occurrence_fraction",
    "top_poi_share",
    "top_two_poi_share",
)


def cluster_histogram_jsd_bootstrap(
    real_contributions: np.ndarray,
    generated_contributions: np.ndarray,
    *,
    cluster_ids: Sequence[str] | None,
    seed: int = 20260714,
    resamples: int = 1000,
    compute_device: torch.device | str | None = None,
) -> dict[str, Any]:
    real = np.asarray(real_contributions)
    generated = np.asarray(generated_contributions)
    if real.ndim != 2 or generated.shape != real.shape:
        raise ValueError("histogram contributions must be aligned matrices")
    if np.any(real < 0) or np.any(generated < 0):
        raise ValueError("histogram contributions must be non-negative")
    n_rows = real.shape[0]
    if cluster_ids is None:
        labels = np.arange(n_rows, dtype=np.int64)
        unit = "trajectory_row"
    else:
        if len(cluster_ids) != n_rows:
            raise ValueError("cluster_ids must align with histogram rows")
        _, labels = np.unique(
            np.asarray(cluster_ids, dtype=str), return_inverse=True
        )
        unit = "device_cluster"
    n_clusters = int(labels.max()) + 1 if n_rows else 0
    protocol: dict[str, Any] = {
        "resampling_unit": unit,
        "n_rows": n_rows,
        "n_clusters": n_clusters,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "interval_target": "resampled_panel_estimator_distribution",
        "fixed_panel_point_ci": False,
    }
    if n_clusters == 0 or resamples < 1:
        point = float(
            _rowwise_jsd_counts(
                real.sum(axis=0, keepdims=True),
                generated.sum(axis=0, keepdims=True),
            )[0]
        )
        return {
            "point_estimate": point,
            "bootstrap_95ci": [point, point],
            "protocol": protocol,
        }
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        n_clusters,
        size=(resamples, n_clusters),
        dtype=np.int32,
    )
    cluster_weights = np.zeros((resamples, n_clusters), dtype=np.int32)
    np.add.at(
        cluster_weights,
        (
            np.repeat(np.arange(resamples), n_clusters),
            sampled.ravel(),
        ),
        1,
    )
    return _cluster_bootstrap_histogram_jsd(
        real,
        generated,
        cluster_labels=labels,
        cluster_weights=cluster_weights,
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
    seed: int = 20260714,
    resamples: int = 1000,
    compute_device: torch.device | str | None = None,
) -> dict[str, Any]:
    if real.ndim != 2 or generated.shape != real.shape:
        raise ValueError("sparse token distributions require aligned matrices")
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    n_rows, width = real.shape
    if cluster_ids is None:
        labels = np.arange(n_rows, dtype=np.int64)
        unit = "trajectory_row"
    else:
        if len(cluster_ids) != n_rows:
            raise ValueError("cluster_ids must align with sparse token rows")
        _, labels = np.unique(
            np.asarray(cluster_ids, dtype=str), return_inverse=True
        )
        unit = "device_cluster"
    n_clusters = int(labels.max()) + 1 if n_rows else 0
    real_values = real.long().cpu().numpy()
    generated_values = generated.long().cpu().numpy()
    real_valid = (
        (real_values != pad_id)
        & (real_values >= 0)
        & (real_values < vocab_size)
    )
    generated_valid = (
        (generated_values != pad_id)
        & (generated_values >= 0)
        & (generated_values < vocab_size)
    )
    observed = np.unique(
        np.concatenate(
            [real_values[real_valid], generated_values[generated_valid]]
        )
    )
    n_support = len(observed)
    row_labels = np.broadcast_to(labels[:, None], (n_rows, width))

    def _cluster_matrix(
        values: np.ndarray, valid: np.ndarray
    ) -> sparse.csr_matrix:
        columns = np.searchsorted(observed, values[valid])
        return sparse.coo_matrix(
            (
                np.ones(len(columns), dtype=np.int32),
                (row_labels[valid], columns),
            ),
            shape=(n_clusters, n_support),
        ).tocsr()

    real_by_cluster = _cluster_matrix(real_values, real_valid)
    generated_by_cluster = _cluster_matrix(generated_values, generated_valid)
    real_by_cluster_t = real_by_cluster.T.tocsr()
    generated_by_cluster_t = generated_by_cluster.T.tocsr()

    def _coverage(
        left_counts: np.ndarray, right_counts: np.ndarray
    ) -> np.ndarray:
        real_seen = left_counts > 0
        denominators = real_seen.sum(axis=1)
        numerators = (real_seen & (right_counts > 0)).sum(axis=1)
        return np.divide(
            numerators,
            denominators,
            out=np.zeros_like(numerators, dtype=np.float64),
            where=denominators > 0,
        )

    real_total = np.asarray(real_by_cluster.sum(axis=0))
    generated_total = np.asarray(generated_by_cluster.sum(axis=0))
    point_jsd = float(_rowwise_jsd_counts(real_total, generated_total)[0])
    point_coverage = float(_coverage(real_total, generated_total)[0])
    protocol = {
        "resampling_unit": unit,
        "n_rows": n_rows,
        "n_clusters": n_clusters,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "estimator": "exact_sparse_cluster_bootstrap_v1",
        "interval_target": "resampled_panel_estimator_distribution",
        "fixed_panel_point_ci": False,
    }
    if n_clusters == 0 or resamples < 1:
        jsd_draws = np.asarray([point_jsd])
        coverage_draws = np.asarray([point_coverage])
    elif compute_device is not None:
        device = torch.device(compute_device)

        def _to_sparse_tensor(matrix: sparse.csr_matrix) -> torch.Tensor:
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

        real_sparse = _to_sparse_tensor(real_by_cluster_t)
        generated_sparse = _to_sparse_tensor(generated_by_cluster_t)
        rng = np.random.default_rng(seed)
        jsd_parts = []
        coverage_parts = []
        chunk_size = max(1, min(128, 8_000_000 // max(n_support, 1)))
        for start in range(0, resamples, chunk_size):
            size = min(chunk_size, resamples - start)
            sampled = rng.integers(
                0,
                n_clusters,
                size=(size, n_clusters),
                dtype=np.int32,
            )
            weights = np.zeros((size, n_clusters), dtype=np.int32)
            np.add.at(
                weights,
                (
                    np.repeat(np.arange(size), n_clusters),
                    sampled.ravel(),
                ),
                1,
            )
            weight_tensor = torch.as_tensor(
                weights, dtype=torch.float64, device=device
            )
            real_draw = torch.sparse.mm(real_sparse, weight_tensor.T).T
            generated_draw = torch.sparse.mm(
                generated_sparse, weight_tensor.T
            ).T
            jsd_parts.append(
                _torch_rowwise_jsd_counts(real_draw, generated_draw)
                .cpu()
                .numpy()
            )
            real_seen = real_draw > 0
            denominators = real_seen.sum(dim=1)
            coverage_parts.append(
                (
                    (real_seen & (generated_draw > 0))
                    .sum(dim=1)
                    .to(dtype=torch.float64)
                    / denominators.clamp_min(1).to(dtype=torch.float64)
                )
                .cpu()
                .numpy()
            )
        jsd_draws = np.concatenate(jsd_parts)
        coverage_draws = np.concatenate(coverage_parts)
    else:
        rng = np.random.default_rng(seed)
        jsd_parts: list[np.ndarray] = []
        coverage_parts: list[np.ndarray] = []
        chunk_size = max(1, min(32, 2_000_000 // max(n_support, 1)))
        for start in range(0, resamples, chunk_size):
            size = min(chunk_size, resamples - start)
            sampled = rng.integers(
                0,
                n_clusters,
                size=(size, n_clusters),
                dtype=np.int32,
            )
            weights = np.zeros((size, n_clusters), dtype=np.int32)
            np.add.at(
                weights,
                (
                    np.repeat(np.arange(size), n_clusters),
                    sampled.ravel(),
                ),
                1,
            )
            real_draw = np.asarray((real_by_cluster_t @ weights.T).T)
            generated_draw = np.asarray((generated_by_cluster_t @ weights.T).T)
            jsd_parts.append(_rowwise_jsd_counts(real_draw, generated_draw))
            coverage_parts.append(_coverage(real_draw, generated_draw))
        jsd_draws = np.concatenate(jsd_parts)
        coverage_draws = np.concatenate(coverage_parts)

    def _summary(point: float, draws: np.ndarray) -> dict[str, Any]:
        return {
            "point_estimate": point,
            "bootstrap_95ci": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
            "protocol": protocol,
        }

    return {
        "poi_frequency_jsd": _summary(point_jsd, jsd_draws),
        "poi_coverage": _summary(point_coverage, coverage_draws),
    }


def cluster_sparse_ngram_jsd_bootstrap(
    real: torch.Tensor,
    generated: torch.Tensor,
    *,
    vocab_size: int,
    n: int,
    pad_id: int,
    cluster_ids: Sequence[str] | None,
    seed: int = 20260714,
    resamples: int = 1000,
    max_vocab: int = 500,
    compute_device: torch.device | str | None = None,
) -> dict[str, Any] | None:
    if vocab_size > max_vocab:
        return None
    if real.ndim != 2 or generated.shape != real.shape:
        raise ValueError("n-gram distributions require aligned matrices")
    if n < 1:
        raise ValueError("n must be positive")
    n_rows = real.size(0)
    if cluster_ids is None:
        labels = np.arange(n_rows, dtype=np.int64)
        unit = "trajectory_row"
    else:
        if len(cluster_ids) != n_rows:
            raise ValueError("cluster_ids must align with n-gram rows")
        _, labels = np.unique(
            np.asarray(cluster_ids, dtype=str), return_inverse=True
        )
        unit = "device_cluster"
    n_clusters = int(labels.max()) + 1 if n_rows else 0

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

    real_hashes, real_valid = _hashed_windows(real)
    generated_hashes, generated_valid = _hashed_windows(generated)
    observed = np.unique(
        np.concatenate(
            [real_hashes[real_valid], generated_hashes[generated_valid]]
        )
    )
    n_support = len(observed)

    def _cluster_matrix(
        hashes: np.ndarray, valid: np.ndarray
    ) -> sparse.csr_matrix:
        row_labels = np.broadcast_to(labels[:, None], hashes.shape)
        columns = np.searchsorted(observed, hashes[valid])
        return sparse.coo_matrix(
            (
                np.ones(len(columns), dtype=np.int32),
                (row_labels[valid], columns),
            ),
            shape=(n_clusters, n_support),
        ).tocsr()

    real_by_cluster = _cluster_matrix(real_hashes, real_valid)
    generated_by_cluster = _cluster_matrix(generated_hashes, generated_valid)
    real_by_cluster_t = real_by_cluster.T.tocsr()
    generated_by_cluster_t = generated_by_cluster.T.tocsr()
    real_total = np.asarray(real_by_cluster.sum(axis=0))
    generated_total = np.asarray(generated_by_cluster.sum(axis=0))
    point = float(_rowwise_jsd_counts(real_total, generated_total)[0])
    protocol = {
        "resampling_unit": unit,
        "n_rows": n_rows,
        "n_clusters": n_clusters,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "estimator": "exact_sparse_ngram_cluster_bootstrap_v1",
        "n": n,
        "interval_target": "resampled_panel_estimator_distribution",
        "fixed_panel_point_ci": False,
    }
    if n_clusters == 0 or resamples < 1:
        draws = np.asarray([point])
    elif compute_device is not None:
        device = torch.device(compute_device)

        def _to_sparse_tensor(matrix: sparse.csr_matrix) -> torch.Tensor:
            coo = matrix.tocoo()
            with torch.sparse.check_sparse_tensor_invariants():
                return torch.sparse_coo_tensor(
                    torch.from_numpy(
                        np.stack([coo.row, coo.col]).astype(
                            np.int64, copy=False
                        )
                    ),
                    torch.from_numpy(coo.data.astype(np.float64, copy=False)),
                    coo.shape,
                    dtype=torch.float64,
                    device=device,
                ).coalesce()

        real_sparse = _to_sparse_tensor(real_by_cluster_t)
        generated_sparse = _to_sparse_tensor(generated_by_cluster_t)
        rng = np.random.default_rng(seed)
        parts = []
        chunk_size = max(1, min(128, 8_000_000 // max(n_support, 1)))
        for start in range(0, resamples, chunk_size):
            size = min(chunk_size, resamples - start)
            sampled = rng.integers(
                0,
                n_clusters,
                size=(size, n_clusters),
                dtype=np.int32,
            )
            weights = np.zeros((size, n_clusters), dtype=np.int32)
            np.add.at(
                weights,
                (
                    np.repeat(np.arange(size), n_clusters),
                    sampled.ravel(),
                ),
                1,
            )
            weight_tensor = torch.as_tensor(
                weights, dtype=torch.float64, device=device
            )
            real_draw = torch.sparse.mm(real_sparse, weight_tensor.T).T
            generated_draw = torch.sparse.mm(
                generated_sparse, weight_tensor.T
            ).T
            parts.append(
                _torch_rowwise_jsd_counts(real_draw, generated_draw)
                .cpu()
                .numpy()
            )
        draws = np.concatenate(parts)
    else:
        rng = np.random.default_rng(seed)
        parts: list[np.ndarray] = []
        chunk_size = max(1, min(32, 2_000_000 // max(n_support, 1)))
        for start in range(0, resamples, chunk_size):
            size = min(chunk_size, resamples - start)
            sampled = rng.integers(
                0,
                n_clusters,
                size=(size, n_clusters),
                dtype=np.int32,
            )
            weights = np.zeros((size, n_clusters), dtype=np.int32)
            np.add.at(
                weights,
                (
                    np.repeat(np.arange(size), n_clusters),
                    sampled.ravel(),
                ),
                1,
            )
            real_draw = np.asarray((real_by_cluster_t @ weights.T).T)
            generated_draw = np.asarray((generated_by_cluster_t @ weights.T).T)
            parts.append(_rowwise_jsd_counts(real_draw, generated_draw))
        draws = np.concatenate(parts)
    return {
        "point_estimate": point,
        "bootstrap_95ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "protocol": protocol,
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
    bootstrap_seed: int = 20260714,
    bootstrap_resamples: int = 1000,
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
    if cluster_ids is None:
        cluster_labels = np.arange(n_rows, dtype=np.int64)
        resampling_unit = "trajectory_row"
    else:
        if len(cluster_ids) != n_rows:
            raise ValueError(
                "cluster_ids must align with trajectory rows: "
                f"{len(cluster_ids)} != {n_rows}"
            )
        _, cluster_labels = np.unique(
            np.asarray(cluster_ids, dtype=str), return_inverse=True
        )
        resampling_unit = "device_cluster"
    n_clusters = int(cluster_labels.max()) + 1 if n_rows else 0

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
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_resamples": bootstrap_resamples,
            "resampling_unit": resampling_unit,
            "n_rows": n_rows,
            "n_clusters": n_clusters,
        },
    }
    rng = np.random.default_rng(bootstrap_seed)
    sampled_clusters = rng.integers(
        0, n_clusters, size=(bootstrap_resamples, n_clusters)
    )
    cluster_weights = np.zeros(
        (bootstrap_resamples, n_clusters), dtype=np.int32
    )
    np.add.at(
        cluster_weights,
        (
            np.repeat(np.arange(bootstrap_resamples), n_clusters),
            sampled_clusters.ravel(),
        ),
        1,
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
        "bootstrap_95ci": [
            float(np.quantile(micro_draw, 0.025)),
            float(np.quantile(micro_draw, 0.975)),
        ],
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
            "bootstrap_95ci": [
                float(np.quantile(d, 0.025)),
                float(np.quantile(d, 0.975)),
            ],
        }
    result[kind]["generated_minus_real"] = diff
    histogram_contributions_real = _identity_histogram_contributions(
        real, pad_id=pad_id
    )
    histogram_contributions_gen = _identity_histogram_contributions(
        generated, pad_id=pad_id
    )
    result[kind]["distribution_distances"] = {
        name: _cluster_bootstrap_histogram_jsd(
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
