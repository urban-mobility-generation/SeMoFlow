from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from itertools import combinations
from typing import Any, cast

import numpy as np
import polars as pl

from semoflow.metrics.semantic_id_uncertainty import (
    bootstrap_mean_ci,
)
from semoflow.metrics.trajectory.geo import haversine_km
from semoflow.tokenization.decoding.grounding_index import (
    SemanticIDGroundingIndex,
)
from semoflow.tokenization.token_space import (
    sid_code_columns,
)


def _codebook_entropy_perplexity(
    codes: np.ndarray, *, codebook_size: int
) -> tuple[float, float]:
    counts = np.bincount(codes, minlength=codebook_size).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs)))
    return entropy, float(np.exp(entropy))


def _code_balance_gini(codes: np.ndarray, *, codebook_size: int) -> float:
    counts = np.sort(
        np.bincount(codes, minlength=codebook_size).astype(np.float64)
    )
    n = len(counts)
    index = np.arange(1, n + 1, dtype=np.float64)
    return float(
        (2.0 * np.sum(index * counts) / (n * np.sum(counts))) - (n + 1) / n
    )


def _top1_concentration(codes: np.ndarray) -> float:
    counts = np.bincount(codes)
    return float(counts.max() / counts.sum())


def _collision_bucket_spread_metrics(
    grounding: SemanticIDGroundingIndex,
) -> dict[str, Any]:
    frame = grounding.to_frame()
    empty = {
        "collision_bucket_count": 0,
        "collision_category_entropy_mean": None,
        "collision_geo_radius_mean_km": None,
        "collision_geo_radius_p95_km": None,
    }
    if frame.is_empty():
        return empty

    code_cols = list(sid_code_columns(frame.columns))
    if not code_cols:
        return empty

    category_col = (
        "top_category"
        if "top_category" in frame.columns
        else "top_cat"
        if "top_cat" in frame.columns
        else None
    )
    has_geo = {"latitude", "longitude"}.issubset(set(frame.columns))
    category_entropy: list[float] = []
    geo_radius_km: list[float] = []
    collision_bucket_count = 0
    for bucket in frame.partition_by(code_cols, maintain_order=False):
        if bucket.height <= 1:
            continue
        collision_bucket_count += 1
        if category_col is not None:
            values = [
                value
                for value in bucket[category_col].to_list()
                if value is not None
            ]
            if values:
                counts = np.array(list(Counter(values).values()))
                probs = counts / counts.sum()
                category_entropy.append(float(-np.sum(probs * np.log(probs))))
        if has_geo:
            coords = (
                bucket.select(["latitude", "longitude"])
                .drop_nulls()
                .to_numpy()
            )
            if len(coords) > 1:
                center_lat = float(coords[:, 0].mean())
                center_lon = float(coords[:, 1].mean())
                distances = [
                    haversine_km(
                        float(lat), float(lon), center_lat, center_lon
                    )
                    for lat, lon in coords
                ]
                geo_radius_km.append(float(max(distances)))

    return {
        "collision_bucket_count": collision_bucket_count,
        "collision_category_entropy_mean": (
            float(np.mean(category_entropy)) if category_entropy else None
        ),
        "collision_geo_radius_mean_km": (
            float(np.mean(geo_radius_km)) if geo_radius_km else None
        ),
        "collision_geo_radius_p95_km": (
            float(np.percentile(geo_radius_km, 95)) if geo_radius_km else None
        ),
    }


def sid_quality_metrics(
    grounding: SemanticIDGroundingIndex,
    *,
    placekey_lookup: pl.DataFrame | None = None,
    codebook_size: int | None = None,
) -> dict[str, Any]:
    stats = grounding.stats()
    if placekey_lookup is not None:
        code_cols = list(sid_code_columns(placekey_lookup.columns))
    else:
        placekey_lookup = grounding.to_frame()
        code_cols = list(sid_code_columns(placekey_lookup.columns))

    n_codebooks = len(code_cols)
    utilization: list[float] = []
    active_codes: list[int] = []
    entropy: list[float] = []
    perplexity: list[float] = []
    gini: list[float] = []
    top1: list[float] = []
    for column in code_cols:
        codes = placekey_lookup[column].drop_nulls()
        n_unique = codes.n_unique()
        max_code = cast(int | None, codes.cast(pl.Int64).max())
        cb_size = codebook_size or (int(max_code or 0) + 1)
        utilization.append(n_unique / cb_size if cb_size > 0 else 0.0)
        active_codes.append(int(n_unique))
        codes_np = codes.cast(pl.Int64).to_numpy().astype(np.int64)
        if cb_size > 0 and codes_np.size:
            level_entropy, level_perplexity = _codebook_entropy_perplexity(
                codes_np, codebook_size=cb_size
            )
            entropy.append(level_entropy)
            perplexity.append(level_perplexity)
            gini.append(_code_balance_gini(codes_np, codebook_size=cb_size))
            top1.append(_top1_concentration(codes_np))
        else:
            entropy.append(0.0)
            perplexity.append(0.0)
            gini.append(0.0)
            top1.append(0.0)

    if codebook_size is None and n_codebooks > 0:
        max_code = cast(
            int | None, placekey_lookup[code_cols[0]].cast(pl.Int64).max()
        )
        codebook_size = int(max_code or 0) + 1

    return {
        "n_pois": stats.n_pois,
        "n_keys": stats.n_keys,
        "n_codebooks": n_codebooks,
        "codebook_size": codebook_size or 0,
        "codebook_utilization": utilization,
        "codebook_active_codes": active_codes,
        "codebook_entropy": entropy,
        "codebook_perplexity": perplexity,
        "codebook_gini": gini,
        "codebook_top1_concentration": top1,
        "collision_rate": stats.duplicate_poi_rate,
        "category_purity_mean": stats.category_purity_mean,
        "ambiguous_key_rate": stats.ambiguous_key_rate,
        "singleton_key_rate": stats.singleton_key_rate,
        "candidate_count_histogram": stats.candidate_count_histogram,
        **_collision_bucket_spread_metrics(grounding),
    }


def prefix_diversity_counts(lookup: pl.DataFrame) -> dict[int, int]:
    code_cols = list(sid_code_columns(lookup.columns))
    return {
        depth: lookup.select(code_cols[:depth]).unique().height
        for depth in range(1, len(code_cols) + 1)
    }


def _category_column(frame: pl.DataFrame) -> str | None:
    return next(
        (
            column
            for column in ("top_category", "top_cat")
            if column in frame.columns
        ),
        None,
    )


def prefix_category_metrics(
    grounding: SemanticIDGroundingIndex,
    *,
    placekey_lookup: pl.DataFrame | None = None,
    bootstrap_resamples: int = 0,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 42,
) -> dict[str, dict[int, float] | dict[int, list[float]] | None]:
    frame = grounding.to_frame()
    code_cols = list(sid_code_columns(frame.columns))
    if frame.is_empty() or not code_cols:
        return {
            "prefix_category_purity": None,
            "prefix_category_macro_purity": None,
            "prefix_category_entropy_weighted": None,
            "prefix_category_ami": None,
            "prefix_category_nmi": None,
            "prefix_category_neighbor_recall": None,
            "prefix_category_purity_ci": None,
        }

    category_col = _category_column(frame)
    if category_col is None and placekey_lookup is not None:
        lookup_category_col = _category_column(placekey_lookup)
        join_col = grounding.id_column
        if lookup_category_col is not None and join_col in frame.columns:
            if join_col in placekey_lookup.columns:
                frame = frame.join(
                    placekey_lookup.select(
                        [join_col, lookup_category_col]
                    ).unique(subset=[join_col]),
                    on=join_col,
                    how="left",
                )
                category_col = lookup_category_col
            elif (
                join_col == "placekey_id"
                and "placekey" in frame.columns
                and "placekey" in placekey_lookup.columns
            ):
                frame = frame.join(
                    placekey_lookup.select(
                        ["placekey", lookup_category_col]
                    ).unique(subset=["placekey"]),
                    on="placekey",
                    how="left",
                )
                category_col = lookup_category_col

    if category_col is None:
        return {
            "prefix_category_purity": None,
            "prefix_category_macro_purity": None,
            "prefix_category_entropy_weighted": None,
            "prefix_category_ami": None,
            "prefix_category_nmi": None,
            "prefix_category_neighbor_recall": None,
            "prefix_category_purity_ci": None,
        }

    valid = frame.filter(pl.col(category_col).is_not_null())
    if valid.is_empty():
        return {
            "prefix_category_purity": None,
            "prefix_category_macro_purity": None,
            "prefix_category_entropy_weighted": None,
            "prefix_category_ami": None,
            "prefix_category_nmi": None,
            "prefix_category_neighbor_recall": None,
            "prefix_category_purity_ci": None,
        }

    code_values = np.column_stack(
        [valid[column].to_numpy() for column in code_cols]
    ).astype(np.int64, copy=False)
    categories = valid[category_col].to_list()
    category_ids: dict[Any, int] = {}
    encoded_categories = np.empty(len(categories), dtype=np.int64)
    for index, category in enumerate(categories):
        if category not in category_ids:
            category_ids[category] = len(category_ids)
        encoded_categories[index] = category_ids[category]

    purity: dict[int, float] = {}
    macro_purity: dict[int, float] = {}
    entropy_weighted: dict[int, float] = {}
    ami: dict[int, float] = {}
    nmi: dict[int, float] = {}
    neighbour_recall: dict[int, float] = {}
    purity_ci: dict[int, list[float]] = {}
    category_totals = Counter(categories)

    try:
        from sklearn.metrics import (
            adjusted_mutual_info_score,
            normalized_mutual_info_score,
        )
    except ImportError:
        adjusted_mutual_info_score = None
        normalized_mutual_info_score = None

    for depth in range(1, len(code_cols) + 1):
        cluster_ids: dict[tuple[int, ...], int] = {}
        cluster_categories: dict[int, Counter[Any]] = defaultdict(Counter)
        encoded_clusters = np.empty(len(categories), dtype=np.int64)
        for index in range(code_values.shape[0]):
            prefix = tuple(
                int(value) for value in code_values[index, :depth].tolist()
            )
            cluster_id = cluster_ids.setdefault(prefix, len(cluster_ids))
            encoded_clusters[index] = cluster_id
            cluster_categories[cluster_id][categories[index]] += 1

        cluster_sizes = np.bincount(encoded_clusters)
        weighted_majority = 0
        cluster_purities: list[float] = []
        weighted_entropy = 0.0
        for cluster_id, counts in cluster_categories.items():
            size = int(cluster_sizes[cluster_id])
            majority = max(counts.values())
            weighted_majority += majority
            cluster_purities.append(majority / size)
            probabilities = np.asarray(list(counts.values()), dtype=np.float64)
            probabilities /= probabilities.sum()
            weighted_entropy += (size / len(categories)) * float(
                -np.sum(probabilities * np.log(probabilities))
            )

        per_poi_purity = [
            float(
                category
                == cluster_categories[int(cluster_id)].most_common(1)[0][0]
            )
            for cluster_id, category in zip(
                encoded_clusters, categories, strict=True
            )
        ]

        purity[depth] = weighted_majority / len(categories)
        macro_purity[depth] = float(np.mean(cluster_purities))
        entropy_weighted[depth] = weighted_entropy

        if adjusted_mutual_info_score is not None:
            ami[depth] = float(
                adjusted_mutual_info_score(
                    encoded_categories, encoded_clusters
                )
            )
        if normalized_mutual_info_score is not None:
            nmi[depth] = float(
                normalized_mutual_info_score(
                    encoded_categories,
                    encoded_clusters,
                    average_method="arithmetic",
                )
            )

        recalls: list[float] = []
        for index, category in enumerate(categories):
            denominator = category_totals[category] - 1
            if denominator <= 0:
                continue
            cluster_id = int(encoded_clusters[index])
            numerator = cluster_categories[cluster_id][category] - 1
            recalls.append(numerator / denominator)
        neighbour_recall[depth] = float(np.mean(recalls)) if recalls else 0.0
        if bootstrap_resamples > 0:
            interval = bootstrap_mean_ci(
                per_poi_purity,
                resamples=bootstrap_resamples,
                confidence=bootstrap_confidence,
                seed=bootstrap_seed + depth,
            )
            if interval is not None:
                purity_ci[depth] = interval

    return {
        "prefix_category_purity": purity,
        "prefix_category_macro_purity": macro_purity,
        "prefix_category_entropy_weighted": entropy_weighted,
        "prefix_category_ami": ami if ami else None,
        "prefix_category_nmi": nmi if nmi else None,
        "prefix_category_neighbor_recall": neighbour_recall,
        "prefix_category_purity_ci": purity_ci or None,
    }


def prefix_catalog_metrics(
    grounding: SemanticIDGroundingIndex,
) -> dict[str, dict[int, float] | None]:
    frame = grounding.to_frame()
    code_cols = list(sid_code_columns(frame.columns))
    if frame.is_empty() or not code_cols:
        return {
            "prefix_candidate_set_size_mean": None,
            "prefix_geo_radius_mean_km": None,
            "prefix_geo_radius_p95_km": None,
            "prefix_collision_rate": None,
        }

    has_geo = {"latitude", "longitude"}.issubset(frame.columns)
    candidate_size: dict[int, float] = {}
    geo_radius_mean: dict[int, float] = {}
    geo_radius_p95: dict[int, float] = {}
    collision_rate: dict[int, float] = {}
    code_values = np.column_stack(
        [frame[column].to_numpy() for column in code_cols]
    ).astype(np.int64, copy=False)
    for depth in range(1, len(code_cols) + 1):
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index in range(code_values.shape[0]):
            prefix = tuple(
                int(value) for value in code_values[index, :depth].tolist()
            )
            groups[prefix].append(index)
        sizes = np.asarray([len(indices) for indices in groups.values()])
        candidate_size[depth] = float(np.mean(sizes))
        collision_rate[depth] = 1.0 - len(groups) / frame.height

        if not has_geo:
            continue
        radii: list[float] = []
        latitudes = frame["latitude"].to_numpy()
        longitudes = frame["longitude"].to_numpy()
        for indices in groups.values():
            coords = np.column_stack(
                [latitudes[indices], longitudes[indices]]
            ).astype(np.float64, copy=False)
            coords = coords[np.all(np.isfinite(coords), axis=1)]
            if len(coords) < 2:
                continue
            center_lat = float(coords[:, 0].mean())
            center_lon = float(coords[:, 1].mean())
            radii.append(
                max(
                    haversine_km(
                        float(latitude),
                        float(longitude),
                        center_lat,
                        center_lon,
                    )
                    for latitude, longitude in coords
                )
            )
        if radii:
            geo_radius_mean[depth] = float(np.mean(radii))
            geo_radius_p95[depth] = float(np.percentile(radii, 95))

    return {
        "prefix_candidate_set_size_mean": candidate_size,
        "prefix_geo_radius_mean_km": geo_radius_mean or None,
        "prefix_geo_radius_p95_km": geo_radius_p95 or None,
        "prefix_collision_rate": collision_rate,
    }


def spatial_displacement(
    grounding: SemanticIDGroundingIndex,
    latlon_map: dict[str, tuple[float, float]],
) -> float | None:
    per_code_distances: list[float] = []
    for payloads in grounding.iter_payload_groups():
        if len(payloads) < 2:
            continue
        coordinates = [
            latlon_map[placekey]
            for payload in payloads
            if isinstance((placekey := payload.get("placekey")), str)
            and placekey in latlon_map
        ]
        if len(coordinates) < 2:
            continue
        distances = [
            haversine_km(left[0], left[1], right[0], right[1])
            for left, right in combinations(coordinates, 2)
        ]
        per_code_distances.append(float(np.mean(distances)))

    if not per_code_distances:
        return None
    return float(np.median(per_code_distances))


def poi_recovery_metrics(
    grounding: SemanticIDGroundingIndex,
    placekey_lookup: pl.DataFrame,
    *,
    embedding_by_id: Mapping[Any, np.ndarray] | None = None,
    topk: tuple[int, ...] = (1, 2, 5),
) -> dict[str, Any]:
    code_cols = list(sid_code_columns(placekey_lookup.columns))
    id_col = grounding.id_column
    if not code_cols or id_col not in placekey_lookup.columns:
        return {
            "poi_sid_only_top1_recovery_rate": None,
            "poi_candidate_set_recovery_rate": None,
            "poi_sid_only_topk_recovery_rates": None,
            "poi_sid_only_top1_recovery_noncollision_rate": None,
            "poi_embedding_assisted_recovery_rate": None,
            "poi_embedding_assisted_scored": 0,
        }

    total = 0
    top1_recovered = 0
    candidate_set_recovered = 0
    noncollision_total = 0
    noncollision_recovered = 0
    embedding_total = 0
    embedding_recovered = 0
    topk_recovered = {int(k): 0 for k in sorted(set(topk)) if int(k) > 0}
    selected_columns = [id_col, *code_cols]
    for row in placekey_lookup.select(selected_columns).iter_rows(named=True):
        candidates = grounding.candidate_payloads(row)
        total += 1
        poi_id = row[id_col]
        ordered_candidates = sorted(
            candidates,
            key=lambda candidate: (
                candidate.get(id_col) is None,
                str(candidate.get(id_col)),
            ),
        )
        candidate_ids = [
            candidate.get(id_col) for candidate in ordered_candidates
        ]
        candidate_set_hit = poi_id in candidate_ids
        candidate_set_recovered += int(candidate_set_hit)
        top1_hit = bool(candidate_ids and candidate_ids[0] == poi_id)
        top1_recovered += int(top1_hit)
        for k, count in topk_recovered.items():
            topk_recovered[k] = count + int(poi_id in candidate_ids[:k])

        if len(candidate_ids) == 1:
            noncollision_total += 1
            noncollision_recovered += int(top1_hit)

        if embedding_by_id is None or poi_id not in embedding_by_id:
            continue
        query = np.asarray(embedding_by_id[poi_id], dtype=np.float64)
        if query.ndim != 1 or not np.all(np.isfinite(query)):
            continue
        candidate_vectors = [
            np.asarray(embedding_by_id[candidate_id], dtype=np.float64)
            for candidate_id in candidate_ids
            if candidate_id in embedding_by_id
        ]
        if not candidate_vectors:
            continue
        query_norm = float(np.linalg.norm(query))
        if query_norm <= 1e-12:
            scores = np.zeros(len(candidate_vectors), dtype=np.float64)
        else:
            scores = np.asarray(
                [
                    float(
                        np.dot(query, vector)
                        / max(query_norm * np.linalg.norm(vector), 1e-12)
                    )
                    for vector in candidate_vectors
                ],
                dtype=np.float64,
            )
        embedding_candidate_ids = [
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id in embedding_by_id
        ]
        order = np.argsort(-scores, kind="stable")
        embedding_total += 1
        embedding_recovered += int(
            embedding_candidate_ids[int(order[0])] == poi_id
        )

    return {
        "poi_sid_only_top1_recovery_rate": (
            top1_recovered / total if total else None
        ),
        "poi_candidate_set_recovery_rate": (
            candidate_set_recovered / total if total else None
        ),
        "poi_sid_only_topk_recovery_rates": {
            k: count / total for k, count in topk_recovered.items()
        }
        if total
        else None,
        "poi_sid_only_top1_recovery_noncollision_rate": (
            noncollision_recovered / noncollision_total
            if noncollision_total
            else None
        ),
        "poi_embedding_assisted_recovery_rate": (
            embedding_recovered / embedding_total if embedding_total else None
        ),
        "poi_embedding_assisted_scored": embedding_total,
    }


__all__ = [
    "poi_recovery_metrics",
    "prefix_catalog_metrics",
    "prefix_category_metrics",
    "prefix_diversity_counts",
    "sid_quality_metrics",
    "spatial_displacement",
]
