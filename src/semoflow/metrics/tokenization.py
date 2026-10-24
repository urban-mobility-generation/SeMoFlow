from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from semoflow.metrics.semantic_id_baselines import select_baseline_metrics
from semoflow.metrics.semantic_id_cf import cf_alignment_metrics
from semoflow.metrics.semantic_id_eval import evaluate_semantic_id_families
from semoflow.metrics.semantic_id_quality import (
    poi_recovery_metrics,
    prefix_catalog_metrics,
    prefix_category_metrics,
    prefix_diversity_counts,
    sid_quality_metrics,
    spatial_displacement,
)
from semoflow.metrics.semantic_id_trajectory import (
    sample_stays_for_trajectory_eval,
    trajectory_behavioral_metrics,
)
from semoflow.metrics.tokenization_reporting import (
    write_semantic_id_eval_report,
)
from semoflow.tokenization.decoding.grounding_index import (
    GroundingStats,
    SemanticIDGroundingIndex,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CFEmbeddingMetrics:
    n_pois: int
    embedding_dim: int
    evaluated_pois: int
    k: int

    norm_mean: float
    norm_std: float
    norm_min: float
    norm_max: float
    zero_norm_rate: float

    transition_neighbor_recall_at_k: float | None
    covisit_neighbor_recall_at_k: float | None
    same_category_purity_at_k: float | None
    temporal_profile_similarity_at_k: float | None
    long_tail_transition_neighbor_recall_at_k: float | None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SemanticIDEvaluation:
    n_pois: int
    n_keys: int
    n_codebooks: int
    codebook_size: int

    codebook_utilization: list[float]
    codebook_active_codes: list[int]
    codebook_entropy: list[float]
    codebook_perplexity: list[float]
    codebook_gini: list[float]
    codebook_top1_concentration: list[float]
    collision_rate: float
    category_purity_mean: float | None
    ambiguous_key_rate: float
    singleton_key_rate: float
    candidate_count_histogram: dict[int, int]
    collision_bucket_count: int
    collision_category_entropy_mean: float | None
    collision_geo_radius_mean_km: float | None
    collision_geo_radius_p95_km: float | None

    prefix_diversity: dict[int, int] | None

    knn_recall_at_k: float | None
    knn_k: int
    cosine_hamming_spearman: float | None
    cf_sample_size: int

    spatial_displacement_median_km: float | None

    knn_recall_at_k_h2: float | None = None
    knn_recall_at_k_h3: float | None = None
    prefix_knn_recall_at_k: float | None = None
    prefix_knn_recall_at_k_by_depth: dict[int, float] | None = None
    cf_transition_recall_at_k: float | None = None
    cf_transition_recall_at_k_by_prefix: dict[int, float] | None = None
    weighted_hamming_spearman: float | None = None
    prefix_cf_coherence: float | None = None
    prefix_cf_coherence_baseline: float | None = None
    cf_pair_sample_size: int = 0

    prefix_category_purity: dict[int, float] | None = None
    prefix_category_macro_purity: dict[int, float] | None = None
    prefix_category_entropy_weighted: dict[int, float] | None = None
    prefix_category_ami: dict[int, float] | None = None
    prefix_category_nmi: dict[int, float] | None = None
    prefix_category_neighbor_recall: dict[int, float] | None = None
    prefix_category_purity_ci: dict[int, list[float]] | None = None
    prefix_candidate_set_size_mean: dict[int, float] | None = None
    prefix_geo_radius_mean_km: dict[int, float] | None = None
    prefix_geo_radius_p95_km: dict[int, float] | None = None
    prefix_collision_rate: dict[int, float] | None = None

    poi_sid_only_top1_recovery_rate: float | None = None
    poi_candidate_set_recovery_rate: float | None = None
    poi_sid_only_topk_recovery_rates: dict[int, float] | None = None
    poi_sid_only_top1_recovery_noncollision_rate: float | None = None
    poi_embedding_assisted_recovery_rate: float | None = None
    poi_embedding_assisted_scored: int = 0

    temporal_distinctiveness: float | None = None
    transition_entropy: float | None = None
    prefix_sharing_rate: dict[int, float] | None = None
    prefix_temporal_coherence: float | None = None
    prefix_temporal_coherence_baseline: float | None = None
    transition_entropy_by_prefix: dict[int, float] | None = None
    prefix_temporal_coherence_by_depth: dict[int, float] | None = None
    prefix_temporal_coherence_baseline_by_depth: dict[int, float] | None = None
    prefix_temporal_coherence_effect_ci_by_depth: (
        dict[int, list[float]] | None
    ) = None
    next_prefix_prediction_accuracy: dict[int, float] | None = None
    next_poi_mutual_information: dict[int, float | None] | None = None
    anchor_repetition_rate: float | None = None
    raw_event_count: int = 0
    collapsed_event_count: int = 0
    bootstrap_confidence_intervals: dict[str, Any] | None = None

    summed_space_metrics_by_depth: (
        dict[int, dict[str, float | None]] | None
    ) = None
    summed_space_k: int = 10
    summed_space_evaluated_pois: int = 0
    prefix_geometry_lcp_distance: dict[int, float] | None = None
    prefix_geometry_lcp_pair_count: dict[int, int] | None = None
    prefix_geometry_lcp_total_pairs: dict[int, int] | None = None
    prefix_geometry_strictly_decreasing: bool | None = None
    prefix_geometry_monotonicity_violations: list[int] | None = None
    per_level_component_mean_norm: dict[int, float] | None = None
    per_level_component_coordinate_variance: dict[int, float] | None = None
    per_level_component_total_variance: dict[int, float] | None = None

    baseline_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)

        def stringify_keys(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    str(key) if isinstance(key, int) else key: stringify_keys(
                        item
                    )
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [stringify_keys(item) for item in value]
            return value

        return stringify_keys(d)


def _subset_rows(n: int, max_pois: int, seed: int) -> torch.Tensor:
    if n <= max_pois:
        return torch.arange(n)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randperm(n, generator=gen)[:max_pois].sort().values


def _topk_neighbor_ids(
    embeddings: torch.Tensor,
    poi_ids: list[int],
    *,
    k: int,
) -> dict[int, list[int]]:
    n = int(embeddings.shape[0])
    if n <= 1:
        return {pid: [] for pid in poi_ids}
    kk = min(k, n - 1)
    z = F.normalize(embeddings.float(), dim=1)
    sim = z @ z.T
    sim.fill_diagonal_(float("-inf"))
    top = sim.topk(kk, dim=1).indices.cpu().tolist()
    return {
        pid: [poi_ids[j] for j in row]
        for pid, row in zip(poi_ids, top, strict=True)
    }


def _transition_neighbors(
    stays: pl.DataFrame, valid_ids: set[int]
) -> dict[int, set[int]]:
    if "device_key" not in stays.columns or "placekey_id" not in stays.columns:
        return {}
    sort_cols = ["device_key"]
    select_cols = ["device_key", "placekey_id"]
    ts_col = "timestamp" if "timestamp" in stays.columns else None
    if ts_col is not None:
        sort_cols.append(ts_col)
        select_cols.append(ts_col)
    seq = (
        stays.select(select_cols)
        .drop_nulls(["device_key", "placekey_id"])
        .with_columns(pl.col("placekey_id").cast(pl.Int64))
        .filter(pl.col("placekey_id").is_in(valid_ids))
        .sort(sort_cols)
        .with_columns(
            pl.col("placekey_id").shift(-1).over("device_key").alias("dst")
        )
        .drop_nulls("dst")
        .filter(pl.col("dst").is_in(valid_ids))
        .filter(pl.col("placekey_id") != pl.col("dst"))
        .select("placekey_id", "dst")
        .unique()
    )
    out: dict[int, set[int]] = defaultdict(set)
    for src, dst in seq.iter_rows():
        out[int(src)].add(int(dst))
    return dict(out)


def _covisit_neighbors(
    stays: pl.DataFrame,
    valid_ids: set[int],
    *,
    max_group_size: int = 200,
) -> dict[int, set[int]]:
    if "device_key" not in stays.columns or "placekey_id" not in stays.columns:
        return {}
    grouped = (
        stays.select(["device_key", "placekey_id"])
        .drop_nulls()
        .with_columns(pl.col("placekey_id").cast(pl.Int64))
        .filter(pl.col("placekey_id").is_in(valid_ids))
        .group_by("device_key")
        .agg(pl.col("placekey_id").unique().alias("pois"))
    )
    out: dict[int, set[int]] = defaultdict(set)
    for (pois,) in grouped.select("pois").iter_rows():
        ids = sorted({int(pid) for pid in pois if int(pid) in valid_ids})
        if len(ids) <= 1 or len(ids) > max_group_size:
            continue
        for pid in ids:
            out[pid].update(other for other in ids if other != pid)
    return dict(out)


def _neighbor_recall(
    predicted: Mapping[int, list[int]],
    target: Mapping[int, set[int]],
    *,
    eligible: set[int] | None = None,
) -> float | None:
    vals: list[float] = []
    for pid, truth in target.items():
        if eligible is not None and pid not in eligible:
            continue
        if not truth or pid not in predicted:
            continue
        hits = len(set(predicted[pid]) & truth)
        vals.append(hits / len(truth))
    return float(np.mean(vals)) if vals else None


def _purity_neigh(
    predicted: Mapping[int, list[int]],
    labels: Mapping[int, object],
) -> float | None:
    vals: list[float] = []
    for pid, neigh in predicted.items():
        label = labels.get(pid)
        if label is None or label == "":
            continue
        valid = [n for n in neigh if labels.get(n) is not None]
        if not valid:
            continue
        vals.append(sum(labels.get(n) == label for n in valid) / len(valid))
    return float(np.mean(vals)) if vals else None


def _labels(index: pl.DataFrame, column: str) -> dict[int, object]:
    if column not in index.columns:
        return {}
    return {
        int(row["placekey_id"]): row[column]
        for row in index.select(["placekey_id", column]).iter_rows(named=True)
    }


def _with_hour(stays: pl.DataFrame) -> pl.DataFrame:
    if "hour" in stays.columns:
        return stays.with_columns(
            pl.col("hour").fill_null(0).cast(pl.Int32).clip(0, 23)
        )
    ts_col = "timestamp" if "timestamp" in stays.columns else None
    if ts_col is not None:
        return stays.with_columns(
            pl.col(ts_col)
            .cast(pl.Datetime)
            .dt.hour()
            .fill_null(0)
            .cast(pl.Int32)
            .alias("hour")
        )
    return stays.with_columns(pl.lit(0, dtype=pl.Int32).alias("hour"))


def _hour_profiles(
    stays: pl.DataFrame, valid_ids: set[int]
) -> dict[int, torch.Tensor]:
    if "placekey_id" not in stays.columns:
        return {}
    frame = (
        _with_hour(stays)
        .select(["placekey_id", "hour"])
        .drop_nulls()
        .with_columns(pl.col("placekey_id").cast(pl.Int64))
        .filter(pl.col("placekey_id").is_in(valid_ids))
        .group_by("placekey_id")
        .agg(
            *[
                (pl.col("hour") == hour).mean().alias(f"h{hour}")
                for hour in range(24)
            ]
        )
    )
    out: dict[int, torch.Tensor] = {}
    for row in frame.iter_rows(named=True):
        vec = torch.tensor([float(row[f"h{i}"]) for i in range(24)])
        out[int(row["placekey_id"])] = F.normalize(vec, dim=0)
    return out


def _temporal_similarity(
    predicted: Mapping[int, list[int]],
    profiles: Mapping[int, torch.Tensor],
) -> float | None:
    vals: list[float] = []
    for pid, neigh in predicted.items():
        anchor = profiles.get(pid)
        if anchor is None:
            continue
        sims = [
            float(torch.dot(anchor, profiles[n]).item())
            for n in neigh
            if n in profiles
        ]
        if sims:
            vals.append(float(np.mean(sims)))
    return float(np.mean(vals)) if vals else None


def _long_tail_ids(
    stays: pl.DataFrame, valid_ids: set[int], quantile: float
) -> set[int]:
    if "placekey_id" not in stays.columns:
        return set()
    counts = (
        stays.select("placekey_id")
        .drop_nulls()
        .with_columns(pl.col("placekey_id").cast(pl.Int64))
        .filter(pl.col("placekey_id").is_in(valid_ids))
        .group_by("placekey_id")
        .agg(pl.len().alias("n"))
    )
    if counts.is_empty():
        return set()
    threshold = counts["n"].quantile(quantile)
    if threshold is None:
        return set()
    return {
        int(row["placekey_id"])
        for row in counts.filter(pl.col("n") <= threshold).iter_rows(
            named=True
        )
    }


def compute_cf_embedding_metrics(
    embeddings: torch.Tensor,
    index: pl.DataFrame,
    stays: pl.DataFrame,
    *,
    k: int = 10,
    max_pois: int = 5_000,
    seed: int = 0,
    long_tail_quantile: float = 0.25,
) -> CFEmbeddingMetrics:
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D, got shape={tuple(embeddings.shape)}"
        )
    if index.height != embeddings.shape[0]:
        raise ValueError(
            "index row count must match embeddings: "
            f"index={index.height} embeddings={embeddings.shape[0]}"
        )
    if "placekey_id" not in index.columns:
        raise ValueError("index must contain placekey_id")

    rows = _subset_rows(int(embeddings.shape[0]), max_pois, seed)
    emb = embeddings.index_select(0, rows).float().cpu()
    row_list = [int(v) for v in rows.tolist()]
    idx = (
        index.with_row_index("_cf_eval_row")
        .filter(pl.col("_cf_eval_row").is_in(row_list))
        .drop("_cf_eval_row")
    )
    poi_ids = [int(v) for v in idx["placekey_id"].to_list()]
    valid_ids = set(poi_ids)
    predicted = _topk_neighbor_ids(emb, poi_ids, k=k)

    norms = emb.norm(dim=1)
    transitions = _transition_neighbors(stays, valid_ids)
    covisits = _covisit_neighbors(stays, valid_ids)
    profiles = _hour_profiles(stays, valid_ids)
    long_tail = _long_tail_ids(stays, valid_ids, long_tail_quantile)

    return CFEmbeddingMetrics(
        n_pois=int(embeddings.shape[0]),
        embedding_dim=int(embeddings.shape[1]),
        evaluated_pois=int(emb.shape[0]),
        k=int(min(k, max(int(emb.shape[0]) - 1, 0))),
        norm_mean=float(norms.mean().item()) if norms.numel() else 0.0,
        norm_std=float(norms.std(unbiased=False).item())
        if norms.numel()
        else 0.0,
        norm_min=float(norms.min().item()) if norms.numel() else 0.0,
        norm_max=float(norms.max().item()) if norms.numel() else 0.0,
        zero_norm_rate=(
            float((norms <= 1e-8).float().mean().item())
            if norms.numel()
            else 0.0
        ),
        transition_neighbor_recall_at_k=_neighbor_recall(
            predicted, transitions
        ),
        covisit_neighbor_recall_at_k=_neighbor_recall(predicted, covisits),
        same_category_purity_at_k=_purity_neigh(
            predicted, _labels(idx, "top_cat")
        ),
        temporal_profile_similarity_at_k=_temporal_similarity(
            predicted, profiles
        ),
        long_tail_transition_neighbor_recall_at_k=_neighbor_recall(
            predicted,
            transitions,
            eligible=long_tail,
        ),
    )


def evaluate_semantic_ids(
    grounding: SemanticIDGroundingIndex,
    *,
    placekey_lookup: pl.DataFrame | None = None,
    stays: pl.DataFrame | None = None,
    cf_embeddings: torch.Tensor | None = None,
    cf_index: pl.DataFrame | None = None,
    latlon_map: dict[str, tuple[float, float]] | None = None,
    codebooks: torch.Tensor | None = None,
    codebook_size: int | None = None,
    k: int = 10,
    sample_size: int = 5000,
    pair_sample_size: int = 2_000_000,
    geometry_sample_size: int = 5_000,
    geometry_lcp_pair_sample_size: int = 10_000,
    seed: int = 42,
    bootstrap_resamples: int = 0,
    bootstrap_confidence: float = 0.95,
) -> SemanticIDEvaluation:
    return SemanticIDEvaluation(
        **evaluate_semantic_id_families(
            grounding,
            placekey_lookup=placekey_lookup,
            stays=stays,
            cf_embeddings=cf_embeddings,
            cf_index=cf_index,
            latlon_map=latlon_map,
            codebooks=codebooks,
            codebook_size=codebook_size,
            k=k,
            sample_size=sample_size,
            pair_sample_size=pair_sample_size,
            geometry_sample_size=geometry_sample_size,
            geometry_lcp_pair_sample_size=geometry_lcp_pair_sample_size,
            seed=seed,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_confidence=bootstrap_confidence,
        )
    )


__all__ = [
    "CFEmbeddingMetrics",
    "GroundingStats",
    "SemanticIDEvaluation",
    "SemanticIDGroundingIndex",
    "cf_alignment_metrics",
    "compute_cf_embedding_metrics",
    "evaluate_semantic_ids",
    "poi_recovery_metrics",
    "prefix_catalog_metrics",
    "prefix_category_metrics",
    "prefix_diversity_counts",
    "sample_stays_for_trajectory_eval",
    "select_baseline_metrics",
    "sid_quality_metrics",
    "spatial_displacement",
    "trajectory_behavioral_metrics",
    "write_semantic_id_eval_report",
]
