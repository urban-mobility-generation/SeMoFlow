from __future__ import annotations

from pathlib import Path

import polars as pl
import torch


def prefix_centroid_table(
    catalog_path: Path,
    places_path: Path,
    num_codebooks: int,
    prefix_depth: int,
) -> dict[tuple[int, ...], tuple[float, float]]:
    columns = [f"sem_code_{level}" for level in range(num_codebooks)]
    lookup = pl.read_parquet(catalog_path, columns=["placekey_id", *columns])
    places = pl.read_parquet(
        places_path,
        columns=["placekey_id", "latitude", "longitude", "city_id"],
    )
    place_rows = {
        int(row["placekey_id"]): row for row in places.iter_rows(named=True)
    }
    sums: dict[tuple[int, ...], list[float]] = {}
    counts: dict[tuple[int, ...], int] = {}
    cities: dict[tuple[int, ...], set[int]] = {}
    for row in lookup.iter_rows(named=True):
        place = place_rows.get(int(row["placekey_id"]))
        if (
            place is None
            or place["latitude"] is None
            or place["longitude"] is None
        ):
            continue
        prefix = tuple(int(row[column]) for column in columns[:prefix_depth])
        entry = sums.setdefault(prefix, [0.0, 0.0])
        entry[0] += float(place["latitude"])
        entry[1] += float(place["longitude"])
        counts[prefix] = counts.get(prefix, 0) + 1
        if place["city_id"] is not None:
            cities.setdefault(prefix, set()).add(int(place["city_id"]))
    return {
        prefix: (total[0] / counts[prefix], total[1] / counts[prefix])
        for prefix, total in sums.items()
        if len(cities.get(prefix, set())) <= 1
    }


def prefix_hop_distance_km(
    prefix_codes: torch.Tensor,
    valid_mask: torch.Tensor,
    centroids: dict[tuple[int, ...], tuple[float, float]],
) -> torch.Tensor:
    import numpy as np

    rows, length, _ = prefix_codes.shape
    codes_np = prefix_codes.cpu().numpy()
    valid_np = valid_mask.cpu().numpy().astype(bool)
    prefix_order = list(centroids)
    if not prefix_order:
        return torch.zeros((rows, length), dtype=torch.float32)
    coordinates = np.asarray(
        [centroids[prefix] for prefix in prefix_order], dtype=np.float64
    )
    max_code = max(
        int(codes_np.max(initial=0)),
        max(max(prefix, default=0) for prefix in prefix_order),
    )
    base = max_code + 1
    if base ** codes_np.shape[-1] > np.iinfo(np.int64).max:
        raise ValueError("prefix width exceeds int64 key capacity")
    powers = np.asarray(
        [base**power for power in range(codes_np.shape[-1] - 1, -1, -1)],
        dtype=np.int64,
    )
    centroid_keys = np.asarray(prefix_order, dtype=np.int64) @ powers
    centroid_order = np.argsort(centroid_keys, kind="stable")
    sorted_keys = centroid_keys[centroid_order]
    query_keys = codes_np.astype(np.int64, copy=False) @ powers
    positions = np.searchsorted(sorted_keys, query_keys)
    safe_positions = np.minimum(positions, sorted_keys.size - 1)
    matched = (
        (positions < sorted_keys.size)
        & (sorted_keys[safe_positions] == query_keys)
        & valid_np
    )
    index = np.full((rows, length), -1, dtype=np.int64)
    index[matched] = centroid_order[safe_positions[matched]]
    previous_index = np.full((rows, length), -1, dtype=np.int64)
    previous_index[:, 1:] = index[:, :-1]
    previous_valid = np.zeros((rows, length), dtype=bool)
    previous_valid[:, 1:] = valid_np[:, :-1]
    resolvable = (
        valid_np & previous_valid & (index >= 0) & (previous_index >= 0)
    )
    safe_index = np.clip(index, 0, None)
    safe_previous = np.clip(previous_index, 0, None)
    lat1 = np.radians(coordinates[safe_previous, 0])
    lon1 = np.radians(coordinates[safe_previous, 1])
    lat2 = np.radians(coordinates[safe_index, 0])
    lon2 = np.radians(coordinates[safe_index, 1])
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = (
        np.sin(d_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2.0) ** 2
    )
    hop = (
        2.0
        * 6371.0
        * np.arcsin(np.clip(np.sqrt(np.clip(a, 0.0, 1.0)), 0.0, 1.0))
    )
    hop = np.where(resolvable, hop, 0.0)
    return torch.from_numpy(hop.astype(np.float32))


__all__ = ["prefix_centroid_table", "prefix_hop_distance_km"]
