from __future__ import annotations

import logging
from typing import Any

import polars as pl
import torch

from semoflow.metrics.semantic_id_cf import (
    cf_alignment_metrics,
    empty_cf_alignment_metrics,
)
from semoflow.metrics.semantic_id_geometry import (
    summed_codebook_geometry_metrics,
)
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
from semoflow.tokenization.decoding.grounding_index import (
    SemanticIDGroundingIndex,
)
from semoflow.tokenization.token_space import sid_code_columns

logger = logging.getLogger(__name__)


def _embedding_lookup_by_id(
    grounding: SemanticIDGroundingIndex,
    embeddings: torch.Tensor | None,
    index: pl.DataFrame | None,
) -> dict[Any, Any] | None:
    if embeddings is None or index is None:
        return None
    id_col = grounding.id_column
    if id_col not in index.columns or "embedding_row" not in index.columns:
        return None
    values = embeddings.detach().float().cpu().numpy()
    result: dict[Any, Any] = {}
    for row in index.select([id_col, "embedding_row"]).iter_rows(named=True):
        embedding_row = int(row["embedding_row"])
        if 0 <= embedding_row < values.shape[0]:
            result[row[id_col]] = values[embedding_row]
    return result or None


_QUALITY_KEYS = (
    "n_pois",
    "n_keys",
    "n_codebooks",
    "codebook_size",
    "codebook_utilization",
    "codebook_active_codes",
    "codebook_entropy",
    "codebook_perplexity",
    "codebook_gini",
    "codebook_top1_concentration",
    "collision_rate",
    "category_purity_mean",
    "ambiguous_key_rate",
    "singleton_key_rate",
    "candidate_count_histogram",
)
_COLLISION_KEYS = (
    "collision_bucket_count",
    "collision_category_entropy_mean",
    "collision_geo_radius_mean_km",
    "collision_geo_radius_p95_km",
)
_TRAJECTORY_KEYS = (
    "temporal_distinctiveness",
    "transition_entropy",
    "prefix_sharing_rate",
    "prefix_temporal_coherence",
    "prefix_temporal_coherence_baseline",
    "transition_entropy_by_prefix",
    "prefix_temporal_coherence_by_depth",
    "prefix_temporal_coherence_baseline_by_depth",
    "prefix_temporal_coherence_effect_ci_by_depth",
    "next_prefix_prediction_accuracy",
    "next_poi_mutual_information",
    "anchor_repetition_rate",
)
_TRAJECTORY_COUNT_KEYS = ("raw_event_count", "collapsed_event_count")
_EMPTY_RECOVERY: dict[str, Any] = {
    "poi_sid_only_top1_recovery_rate": None,
    "poi_candidate_set_recovery_rate": None,
    "poi_sid_only_topk_recovery_rates": None,
    "poi_sid_only_top1_recovery_noncollision_rate": None,
    "poi_embedding_assisted_recovery_rate": None,
    "poi_embedding_assisted_scored": 0,
}


def evaluate_semantic_id_families(
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
) -> dict[str, Any]:
    quality = sid_quality_metrics(
        grounding,
        placekey_lookup=placekey_lookup,
        codebook_size=codebook_size,
    )
    prefix_categories = prefix_category_metrics(
        grounding,
        placekey_lookup=placekey_lookup,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=seed,
    )
    code_cols = (
        list(sid_code_columns(placekey_lookup.columns))
        if placekey_lookup is not None
        else []
    )
    sampled_stays = (
        sample_stays_for_trajectory_eval(
            stays, max_events=sample_size, seed=seed
        )
        if stays is not None and placekey_lookup is not None and code_cols
        else None
    )

    cf_result = empty_cf_alignment_metrics(k)
    if cf_embeddings is not None and cf_index is not None:
        cf_result = cf_alignment_metrics(
            grounding,
            cf_embeddings,
            cf_index,
            k=k,
            sample_size=sample_size,
            pair_sample_size=pair_sample_size,
            seed=seed,
            stays=sampled_stays,
        )

    prefix_diversity = None
    recovery = _EMPTY_RECOVERY
    geometry: dict[str, Any] = {}
    if placekey_lookup is not None:
        if code_cols:
            prefix_diversity = prefix_diversity_counts(placekey_lookup)
        recovery = poi_recovery_metrics(
            grounding,
            placekey_lookup,
            embedding_by_id=_embedding_lookup_by_id(
                grounding, cf_embeddings, cf_index
            ),
        )
        if codebooks is not None:
            geometry = summed_codebook_geometry_metrics(
                codebooks,
                placekey_lookup,
                stays=sampled_stays,
                graph_embeddings=cf_embeddings,
                graph_index=cf_index,
                k=k,
                max_pois=geometry_sample_size,
                lcp_pair_sample_size=geometry_lcp_pair_sample_size,
                seed=seed,
            )

    traj: dict[str, Any] = {}
    if sampled_stays is not None:
        assert stays is not None and placekey_lookup is not None
        logger.info(
            "computing trajectory-level behavioral metrics  rows=%d/%d",
            sampled_stays.height,
            stays.height,
        )
        traj = trajectory_behavioral_metrics(
            sampled_stays,
            placekey_lookup,
            code_cols,
            seed=seed,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_confidence=bootstrap_confidence,
        )

    bootstrap_intervals = None
    if bootstrap_resamples > 0:
        traj_intervals = traj.get("bootstrap_confidence_intervals", {})
        bootstrap_intervals = {
            "poi": prefix_categories.get("prefix_category_purity_ci"),
            **{
                unit: traj_intervals.get(unit, {})
                for unit in ("users", "trajectories", "time_periods")
            },
            "temporal_coherence_difference": traj.get(
                "prefix_temporal_coherence_effect_ci_by_depth"
            ),
        }

    return {
        **{key: quality[key] for key in _QUALITY_KEYS},
        **prefix_categories,
        **prefix_catalog_metrics(grounding),
        **{key: quality[key] for key in _COLLISION_KEYS},
        "prefix_diversity": prefix_diversity,
        **cf_result,
        "spatial_displacement_median_km": (
            spatial_displacement(grounding, latlon_map)
            if latlon_map is not None
            else None
        ),
        **recovery,
        **{key: traj.get(key) for key in _TRAJECTORY_KEYS},
        **{key: traj.get(key, 0) for key in _TRAJECTORY_COUNT_KEYS},
        "bootstrap_confidence_intervals": bootstrap_intervals,
        **geometry,
    }


__all__ = ["evaluate_semantic_id_families"]
