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
from semoflow.tokenization.token_space import (
    sid_code_columns,
)

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
    prefix_catalog = prefix_catalog_metrics(grounding)

    pf_div: dict[int, int] | None = None
    code_cols: list[str] = []
    if placekey_lookup is not None:
        code_cols = list(sid_code_columns(placekey_lookup.columns))
        if code_cols:
            pf_div = prefix_diversity_counts(placekey_lookup)

    sampled_stays: pl.DataFrame | None = None
    if stays is not None and placekey_lookup is not None and code_cols:
        sampled_stays = sample_stays_for_trajectory_eval(
            stays, max_events=sample_size, seed=seed
        )

    cf_result: dict[str, Any] = empty_cf_alignment_metrics(k)
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

    sp_disp: float | None = None
    if latlon_map is not None:
        sp_disp = spatial_displacement(grounding, latlon_map)

    recovery: dict[str, Any] = {
        "poi_sid_only_top1_recovery_rate": None,
        "poi_candidate_set_recovery_rate": None,
        "poi_sid_only_topk_recovery_rates": None,
        "poi_sid_only_top1_recovery_noncollision_rate": None,
        "poi_embedding_assisted_recovery_rate": None,
        "poi_embedding_assisted_scored": 0,
    }
    if placekey_lookup is not None:
        recovery = poi_recovery_metrics(
            grounding,
            placekey_lookup,
            embedding_by_id=_embedding_lookup_by_id(
                grounding, cf_embeddings, cf_index
            ),
        )

    td: float | None = None
    te: float | None = None
    psr: dict[int, float] | None = None
    ptc: float | None = None
    ptc_baseline: float | None = None
    traj: dict[str, Any] = {}
    if (
        sampled_stays is not None
        and stays is not None
        and placekey_lookup is not None
    ):
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
        td = traj["temporal_distinctiveness"]
        te = traj["transition_entropy"]
        psr = traj["prefix_sharing_rate"]
        ptc = traj["prefix_temporal_coherence"]
        ptc_baseline = traj["prefix_temporal_coherence_baseline"]

    geometry: dict[str, Any] = {}
    if codebooks is not None and placekey_lookup is not None:
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

    return {
        "n_pois": quality["n_pois"],
        "n_keys": quality["n_keys"],
        "n_codebooks": quality["n_codebooks"],
        "codebook_size": quality["codebook_size"],
        "codebook_utilization": quality["codebook_utilization"],
        "codebook_active_codes": quality["codebook_active_codes"],
        "codebook_entropy": quality["codebook_entropy"],
        "codebook_perplexity": quality["codebook_perplexity"],
        "codebook_gini": quality["codebook_gini"],
        "codebook_top1_concentration": quality["codebook_top1_concentration"],
        "collision_rate": quality["collision_rate"],
        "category_purity_mean": quality["category_purity_mean"],
        "ambiguous_key_rate": quality["ambiguous_key_rate"],
        "singleton_key_rate": quality["singleton_key_rate"],
        "candidate_count_histogram": quality["candidate_count_histogram"],
        **prefix_categories,
        **prefix_catalog,
        "collision_bucket_count": quality["collision_bucket_count"],
        "collision_category_entropy_mean": quality[
            "collision_category_entropy_mean"
        ],
        "collision_geo_radius_mean_km": quality[
            "collision_geo_radius_mean_km"
        ],
        "collision_geo_radius_p95_km": quality["collision_geo_radius_p95_km"],
        "prefix_diversity": pf_div,
        "knn_recall_at_k": cf_result["knn_recall_at_k"],
        "knn_k": cf_result["knn_k"],
        "cosine_hamming_spearman": cf_result["cosine_hamming_spearman"],
        "cf_sample_size": cf_result["cf_sample_size"],
        "spatial_displacement_median_km": sp_disp,
        "knn_recall_at_k_h2": cf_result["knn_recall_at_k_h2"],
        "knn_recall_at_k_h3": cf_result["knn_recall_at_k_h3"],
        "prefix_knn_recall_at_k": cf_result["prefix_knn_recall_at_k"],
        "prefix_knn_recall_at_k_by_depth": cf_result[
            "prefix_knn_recall_at_k_by_depth"
        ],
        "cf_transition_recall_at_k": cf_result["cf_transition_recall_at_k"],
        "cf_transition_recall_at_k_by_prefix": cf_result[
            "cf_transition_recall_at_k_by_prefix"
        ],
        "weighted_hamming_spearman": cf_result["weighted_hamming_spearman"],
        "prefix_cf_coherence": cf_result["prefix_cf_coherence"],
        "prefix_cf_coherence_baseline": cf_result[
            "prefix_cf_coherence_baseline"
        ],
        "cf_pair_sample_size": cf_result["cf_pair_sample_size"],
        **recovery,
        "temporal_distinctiveness": td,
        "transition_entropy": te,
        "prefix_sharing_rate": psr,
        "prefix_temporal_coherence": ptc,
        "prefix_temporal_coherence_baseline": ptc_baseline,
        "transition_entropy_by_prefix": (
            traj.get("transition_entropy_by_prefix")
            if sampled_stays is not None
            else None
        ),
        "prefix_temporal_coherence_by_depth": (
            traj.get("prefix_temporal_coherence_by_depth")
            if sampled_stays is not None
            else None
        ),
        "prefix_temporal_coherence_baseline_by_depth": (
            traj.get("prefix_temporal_coherence_baseline_by_depth")
            if sampled_stays is not None
            else None
        ),
        "prefix_temporal_coherence_effect_ci_by_depth": (
            traj.get("prefix_temporal_coherence_effect_ci_by_depth")
            if sampled_stays is not None
            else None
        ),
        "next_prefix_prediction_accuracy": (
            traj.get("next_prefix_prediction_accuracy")
            if sampled_stays is not None
            else None
        ),
        "next_poi_mutual_information": (
            traj.get("next_poi_mutual_information")
            if sampled_stays is not None
            else None
        ),
        "anchor_repetition_rate": (
            traj.get("anchor_repetition_rate")
            if sampled_stays is not None
            else None
        ),
        "raw_event_count": (
            traj.get("raw_event_count", 0) if sampled_stays is not None else 0
        ),
        "collapsed_event_count": (
            traj.get("collapsed_event_count", 0)
            if sampled_stays is not None
            else 0
        ),
        "bootstrap_confidence_intervals": (
            {
                "poi": prefix_categories.get("prefix_category_purity_ci"),
                "users": traj.get("bootstrap_confidence_intervals", {}).get(
                    "users", {}
                ),
                "trajectories": traj.get(
                    "bootstrap_confidence_intervals", {}
                ).get("trajectories", {}),
                "time_periods": traj.get(
                    "bootstrap_confidence_intervals", {}
                ).get("time_periods", {}),
                "temporal_coherence_difference": traj.get(
                    "prefix_temporal_coherence_effect_ci_by_depth"
                ),
            }
            if bootstrap_resamples > 0
            else None
        ),
        **geometry,
    }


__all__ = ["evaluate_semantic_id_families"]
