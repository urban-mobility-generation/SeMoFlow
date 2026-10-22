from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl
import torch

from semoflow.metrics.semantic_id_trajectory import (
    collapse_consecutive_pois,
)
from semoflow.tokenization.token_space import (
    sid_code_columns,
)


def summed_codebook_geometry_metrics(
    codebooks: torch.Tensor,
    sid_lookup: pl.DataFrame,
    *,
    stays: pl.DataFrame | None = None,
    graph_embeddings: torch.Tensor | None = None,
    graph_index: pl.DataFrame | None = None,
    k: int = 10,
    max_pois: int = 5_000,
    lcp_pair_sample_size: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    empty = _empty_geometry_metrics(k)
    if codebooks.ndim != 3:
        raise ValueError("codebooks must have shape [levels, codes, dim]")

    code_cols = list(sid_code_columns(sid_lookup.columns))
    n_levels = int(codebooks.shape[0])
    if not code_cols or len(code_cols) != n_levels:
        return empty
    id_column = (
        "placekey_id" if "placekey_id" in sid_lookup.columns else "placekey"
    )
    frame = sid_lookup.unique(subset=[id_column], maintain_order=True)
    if frame.height < 2:
        return empty

    graph_rows: np.ndarray | None = None
    if graph_embeddings is not None and graph_index is not None:
        frame, graph_rows = _align_graph_rows(
            frame, graph_index, graph_embeddings, id_column=id_column
        )
    if frame.height < 2:
        return empty

    codes = frame.select(code_cols).to_numpy().astype(np.int64, copy=True)
    n_pois = frame.height
    if np.any(codes < 0) or np.any(codes >= int(codebooks.shape[1])):
        raise ValueError("SID codes fall outside the exported codebooks")
    codebook_np = codebooks.detach().cpu().float().numpy()
    ids = frame[id_column].to_numpy()
    rng = np.random.RandomState(seed)
    query_rows = _sample_rows(n_pois, max_pois=max_pois, rng=rng)
    actual_k = min(k, n_pois - 1)

    categories = _metadata_values(frame, ("top_category", "top_cat"))
    latitudes = _numeric_metadata(frame, "latitude")
    longitudes = _numeric_metadata(frame, "longitude")
    transition_targets = _transition_targets(
        stays,
        ids=set(ids.tolist()),
        query_ids=set(ids[query_rows].tolist()),
        id_column=id_column,
    )
    temporal_profiles = _temporal_profiles(
        stays,
        ids=set(ids.tolist()),
        id_column=id_column,
    )

    graph_neighbors: np.ndarray | None = None
    if graph_rows is not None and graph_embeddings is not None:
        graph_values = (
            graph_embeddings[torch.as_tensor(graph_rows, dtype=torch.long)]
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        graph_neighbors = _cosine_neighbors(
            graph_values,
            query_rows,
            k=actual_k,
        )

    per_depth: dict[int, dict[str, float | None]] = {}
    mean_norm: dict[int, float] = {}
    coordinate_variance: dict[int, float] = {}
    total_variance: dict[int, float] = {}
    summed = np.zeros((n_pois, int(codebooks.shape[2])), dtype=np.float32)
    for level in range(n_levels):
        component = codebook_np[level, codes[:, level]]
        summed += component
        depth = level + 1
        mean_norm[depth] = float(np.linalg.norm(component, axis=1).mean())
        coordinate_variance[depth] = float(component.var(axis=0).mean())
        total_variance[depth] = float(
            np.square(component - component.mean(axis=0)).sum(axis=1).mean()
        )

        neighbors = _l2_neighbors(summed, query_rows, k=actual_k)
        per_depth[depth] = {
            "graph_neighbor_recall_at_k": _graph_neighbor_recall(
                neighbors, graph_neighbors
            ),
            "temporal_transition_neighbor_recall_at_k": _transition_recall(
                ids, query_rows, neighbors, transition_targets
            ),
            "category_purity_at_k": _category_purity(
                categories, query_rows, neighbors
            ),
            "temporal_profile_similarity_at_k": _temporal_profile_similarity(
                ids, query_rows, neighbors, temporal_profiles
            ),
            "geographic_neighbor_distance_km": _geographic_neighbor_distance(
                latitudes, longitudes, query_rows, neighbors
            ),
        }

    lcp_distance, lcp_pair_count, lcp_total_pairs = _lcp_distances(
        codes,
        summed,
        pair_sample_size=lcp_pair_sample_size,
        rng=rng,
    )
    has_all_lcp_strata = all(
        depth in lcp_distance for depth in range(n_levels)
    )
    monotonic = (
        all(
            lcp_distance[depth] > lcp_distance[depth + 1]
            for depth in range(n_levels - 1)
        )
        if has_all_lcp_strata
        else None
    )
    violations = [
        depth
        for depth in range(n_levels - 1)
        if lcp_distance.get(depth) is not None
        and lcp_distance.get(depth + 1) is not None
        and float(lcp_distance[depth]) <= float(lcp_distance[depth + 1])
    ]
    return {
        "summed_space_metrics_by_depth": per_depth,
        "summed_space_k": actual_k,
        "summed_space_evaluated_pois": len(query_rows),
        "prefix_geometry_lcp_distance": lcp_distance or None,
        "prefix_geometry_lcp_pair_count": lcp_pair_count or None,
        "prefix_geometry_lcp_total_pairs": lcp_total_pairs or None,
        "prefix_geometry_strictly_decreasing": monotonic,
        "prefix_geometry_monotonicity_violations": violations,
        "per_level_component_mean_norm": mean_norm,
        "per_level_component_coordinate_variance": coordinate_variance,
        "per_level_component_total_variance": total_variance,
    }


def _empty_geometry_metrics(k: int) -> dict[str, Any]:
    return {
        "summed_space_metrics_by_depth": None,
        "summed_space_k": k,
        "summed_space_evaluated_pois": 0,
        "prefix_geometry_lcp_distance": None,
        "prefix_geometry_lcp_pair_count": None,
        "prefix_geometry_lcp_total_pairs": None,
        "prefix_geometry_strictly_decreasing": None,
        "prefix_geometry_monotonicity_violations": None,
        "per_level_component_mean_norm": None,
        "per_level_component_coordinate_variance": None,
        "per_level_component_total_variance": None,
    }


def _align_graph_rows(
    frame: pl.DataFrame,
    graph_index: pl.DataFrame,
    graph_embeddings: torch.Tensor,
    *,
    id_column: str,
) -> tuple[pl.DataFrame, np.ndarray | None]:
    if id_column not in graph_index.columns:
        return frame, None
    index = graph_index
    if "embedding_row" not in index.columns:
        if index.height != int(graph_embeddings.shape[0]):
            return frame, None
        index = index.with_row_index("embedding_row")
    joined = frame.join(
        index.select([id_column, "embedding_row"]).unique(subset=[id_column]),
        on=id_column,
        how="inner",
    )
    if joined.is_empty():
        return joined, None
    rows = joined["embedding_row"].cast(pl.Int64).to_numpy().copy()
    valid = (rows >= 0) & (rows < int(graph_embeddings.shape[0]))
    return joined.filter(pl.Series(valid)).drop("embedding_row"), rows[valid]


def _sample_rows(
    n_pois: int,
    *,
    max_pois: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    if n_pois <= max_pois:
        return np.arange(n_pois, dtype=np.int64)
    return np.sort(rng.choice(n_pois, size=max_pois, replace=False))


def _l2_neighbors(
    values: np.ndarray,
    query_rows: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    if k <= 0:
        return np.empty((len(query_rows), 0), dtype=np.int64)
    try:
        import faiss

        index: Any = faiss.IndexFlatL2(values.shape[1])
        index.add(np.ascontiguousarray(values.astype(np.float32, copy=False)))
        _distances, raw = index.search(
            np.ascontiguousarray(values[query_rows].astype(np.float32)),
            min(k + 8, values.shape[0]),
        )
    except ImportError:
        raw = _brute_force_l2(values, query_rows, k=k + 8)
    return _exclude_self(raw, query_rows, k=k)


def _cosine_neighbors(
    values: np.ndarray,
    query_rows: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.clip(norms, 1e-12, None)
    try:
        import faiss

        index: Any = faiss.IndexFlatIP(normalized.shape[1])
        index.add(np.ascontiguousarray(normalized.astype(np.float32)))
        _scores, raw = index.search(
            np.ascontiguousarray(normalized[query_rows].astype(np.float32)),
            min(k + 8, normalized.shape[0]),
        )
    except ImportError:
        raw = _brute_force_cosine(normalized, query_rows, k=k + 8)
    return _exclude_self(raw, query_rows, k=k)


def _brute_force_l2(
    values: np.ndarray, query_rows: np.ndarray, *, k: int
) -> np.ndarray:
    result: list[np.ndarray] = []
    for query in query_rows:
        distances = np.square(values - values[query]).sum(axis=1)
        result.append(np.argsort(distances)[:k])
    return np.stack(result)


def _brute_force_cosine(
    values: np.ndarray, query_rows: np.ndarray, *, k: int
) -> np.ndarray:
    result: list[np.ndarray] = []
    for query in query_rows:
        scores = values @ values[query]
        result.append(np.argsort(-scores)[:k])
    return np.stack(result)


def _exclude_self(
    raw_neighbors: np.ndarray,
    query_rows: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    result = np.empty((len(query_rows), k), dtype=np.int64)
    for index, (query, row) in enumerate(
        zip(query_rows, raw_neighbors, strict=True)
    ):
        filtered = row[row != query][:k]
        if len(filtered) != k:
            raise ValueError("unable to retrieve enough non-self neighbours")
        result[index] = filtered
    return result


def _metadata_values(
    frame: pl.DataFrame, columns: tuple[str, ...]
) -> np.ndarray | None:
    column = next((name for name in columns if name in frame.columns), None)
    return frame[column].to_numpy() if column is not None else None


def _numeric_metadata(frame: pl.DataFrame, column: str) -> np.ndarray | None:
    if column not in frame.columns:
        return None
    return frame[column].cast(pl.Float64, strict=False).to_numpy()


def _transition_targets(
    stays: pl.DataFrame | None,
    *,
    ids: set[Any],
    query_ids: set[Any],
    id_column: str,
) -> dict[Any, set[Any]]:
    if (
        stays is None
        or stays.is_empty()
        or "device_key" not in stays.columns
        or id_column not in stays.columns
    ):
        return {}
    collapsed, _anchor_rate, _ = collapse_consecutive_pois(
        stays,
        poi_column=id_column,
    )
    targets = (
        collapsed.with_columns(
            pl.col(id_column).shift(-1).over("device_key").alias("_next_poi")
        )
        .filter(pl.col(id_column).is_in(query_ids))
        .filter(pl.col("_next_poi").is_in(ids))
        .filter(pl.col(id_column) != pl.col("_next_poi"))
        .select([id_column, "_next_poi"])
        .unique()
    )
    result: dict[Any, set[Any]] = defaultdict(set)
    for source, target in targets.iter_rows():
        result[source].add(target)
    return dict(result)


def _temporal_profiles(
    stays: pl.DataFrame | None,
    *,
    ids: set[Any],
    id_column: str,
) -> dict[Any, np.ndarray]:
    if stays is None or stays.is_empty() or id_column not in stays.columns:
        return {}
    if "hour" in stays.columns:
        hour_expr = pl.col("hour").cast(pl.Int32, strict=False)
    elif "local_ts" in stays.columns:
        hour_expr = pl.col("local_ts").dt.hour()
    elif "timestamp" in stays.columns:
        hour_expr = pl.col("timestamp").dt.hour()
    else:
        return {}
    frame = (
        stays.select([id_column, hour_expr.alias("_hour")])
        .drop_nulls([id_column, "_hour"])
        .filter(pl.col(id_column).is_in(ids))
        .filter(pl.col("_hour").is_between(0, 23))
        .group_by(id_column)
        .agg(
            *[
                (pl.col("_hour") == hour).sum().alias(f"h{hour}")
                for hour in range(24)
            ]
        )
    )
    result: dict[Any, np.ndarray] = {}
    for row in frame.iter_rows(named=True):
        profile = np.asarray([row[f"h{hour}"] for hour in range(24)])
        profile = profile.astype(np.float64)
        norm = np.linalg.norm(profile)
        if norm > 0:
            result[row[id_column]] = profile / norm
    return result


def _graph_neighbor_recall(
    neighbors: np.ndarray,
    graph_neighbors: np.ndarray | None,
) -> float | None:
    if graph_neighbors is None or neighbors.size == 0:
        return None
    return float(
        np.mean(
            [
                len(set(left) & set(right)) / neighbors.shape[1]
                for left, right in zip(neighbors, graph_neighbors, strict=True)
            ]
        )
    )


def _transition_recall(
    ids: np.ndarray,
    query_rows: np.ndarray,
    neighbors: np.ndarray,
    targets: dict[Any, set[Any]],
) -> float | None:
    values: list[float] = []
    for query, row in zip(query_rows, neighbors, strict=True):
        truth = targets.get(ids[query])
        if not truth:
            continue
        values.append(len(set(ids[row]) & truth) / len(truth))
    return float(np.mean(values)) if values else None


def _category_purity(
    categories: np.ndarray | None,
    query_rows: np.ndarray,
    neighbors: np.ndarray,
) -> float | None:
    if categories is None:
        return None
    values: list[float] = []
    for query, row in zip(query_rows, neighbors, strict=True):
        category = categories[query]
        if category is None or category == "":
            continue
        neighbour_categories = categories[row]
        valid = [value for value in neighbour_categories if value is not None]
        if valid:
            values.append(float(np.mean(np.asarray(valid) == category)))
    return float(np.mean(values)) if values else None


def _temporal_profile_similarity(
    ids: np.ndarray,
    query_rows: np.ndarray,
    neighbors: np.ndarray,
    profiles: dict[Any, np.ndarray],
) -> float | None:
    values: list[float] = []
    for query, row in zip(query_rows, neighbors, strict=True):
        profile = profiles.get(ids[query])
        if profile is None:
            continue
        similarities = [
            float(profile @ other)
            for neighbor in row
            if (other := profiles.get(ids[neighbor])) is not None
        ]
        if similarities:
            values.append(float(np.mean(similarities)))
    return float(np.mean(values)) if values else None


def _geographic_neighbor_distance(
    latitudes: np.ndarray | None,
    longitudes: np.ndarray | None,
    query_rows: np.ndarray,
    neighbors: np.ndarray,
) -> float | None:
    if latitudes is None or longitudes is None:
        return None
    anchor_lat = latitudes[query_rows, None]
    anchor_lon = longitudes[query_rows, None]
    neighbor_lat = latitudes[neighbors]
    neighbor_lon = longitudes[neighbors]
    valid = (
        np.isfinite(anchor_lat)
        & np.isfinite(anchor_lon)
        & np.isfinite(neighbor_lat)
        & np.isfinite(neighbor_lon)
    )
    if not valid.any():
        return None
    lat1 = np.radians(np.broadcast_to(anchor_lat, neighbor_lat.shape)[valid])
    lon1 = np.radians(np.broadcast_to(anchor_lon, neighbor_lon.shape)[valid])
    lat2 = np.radians(neighbor_lat[valid])
    lon2 = np.radians(neighbor_lon[valid])
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return float(6371.0 * 2 * np.arcsin(np.sqrt(a)).mean())


def _lcp_distances(
    codes: np.ndarray,
    embeddings: np.ndarray,
    *,
    pair_sample_size: int,
    rng: np.random.RandomState,
) -> tuple[dict[int, float], dict[int, int], dict[int, int]]:
    _n_pois, n_levels = codes.shape
    distances: dict[int, float] = {}
    sample_counts: dict[int, int] = {}
    total_counts: dict[int, int] = {}
    for depth in range(n_levels):
        buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index, prefix in enumerate(codes[:, :depth]):
            buckets[tuple(int(value) for value in prefix)].append(index)
        candidates: list[tuple[np.ndarray, np.ndarray]] = []
        weights: list[int] = []
        for indices in buckets.values():
            array = np.asarray(indices, dtype=np.int64)
            if len(array) < 2:
                continue
            next_code = codes[array, depth]
            groups = [
                array[next_code == code] for code in np.unique(next_code)
            ]
            for left_index, left in enumerate(groups):
                for right in groups[left_index + 1 :]:
                    candidates.append((left, right))
                    weights.append(len(left) * len(right))
        total_pairs = int(sum(weights))
        total_counts[depth] = total_pairs
        if total_pairs == 0:
            continue
        draws = min(pair_sample_size, total_pairs)
        selected = rng.choice(
            len(candidates),
            size=draws,
            replace=True,
            p=np.asarray(weights, dtype=np.float64) / total_pairs,
        )
        values: list[float] = []
        for bucket_index in selected:
            left, right = candidates[int(bucket_index)]
            first = int(rng.choice(left))
            second = int(rng.choice(right))
            values.append(
                float(np.linalg.norm(embeddings[first] - embeddings[second]))
            )
        distances[depth] = float(np.mean(values))
        sample_counts[depth] = len(values)

    full_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(codes):
        full_groups[tuple(int(value) for value in row)].append(index)
    full_total = sum(
        len(indices) * (len(indices) - 1) // 2
        for indices in full_groups.values()
    )
    total_counts[n_levels] = full_total
    if full_total:
        distances[n_levels] = 0.0
        sample_counts[n_levels] = min(pair_sample_size, full_total)
    return distances, sample_counts, total_counts


__all__ = ["summed_codebook_geometry_metrics"]
