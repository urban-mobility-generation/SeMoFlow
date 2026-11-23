from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from semoflow.config import PAD_ID
from semoflow.evaluation.distribution import jsd
from semoflow.geodesy import haversine_km

_DISTANCE_BINS = 50
_CACHE_CAPACITY = 16
_table_caches: dict[Callable[..., Any], list[tuple[Any, Any]]] = {}


def _cached_table[K, V](build: Callable[[K], V], lookup: K) -> V:
    cache = _table_caches.setdefault(build, [])
    for index, (stored, table) in enumerate(cache):
        if stored is lookup:
            cache.append(cache.pop(index))
            return table
    table = build(lookup)
    if len(cache) >= _CACHE_CAPACITY:
        cache.pop(0)
    cache.append((lookup, table))
    return table


def _latlon_table(
    placekey_latlon: dict[int, tuple[float, float]],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    keys = np.fromiter(
        placekey_latlon.keys(), dtype=np.int64, count=len(placekey_latlon)
    )
    values = np.asarray(list(placekey_latlon.values()), dtype=np.float32)
    max_id = int(keys.max())
    coords = torch.zeros(max_id + 1, 2, dtype=torch.float32)
    valid = torch.zeros(max_id + 1, dtype=torch.bool)
    coords[keys, 0] = torch.from_numpy(values[:, 0])
    coords[keys, 1] = torch.from_numpy(values[:, 1])
    valid[keys] = True
    return coords, valid, max_id


def _geohash_table(
    geohash_lookup: dict[int, str],
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, int]]:
    all_geohashes = sorted(set(geohash_lookup.values()))
    gh_index = {gh: i for i, gh in enumerate(all_geohashes)}
    keys = np.fromiter(
        geohash_lookup.keys(), dtype=np.int64, count=len(geohash_lookup)
    )
    values = np.asarray(
        [gh_index[gh] for gh in geohash_lookup.values()], dtype=np.int64
    )
    max_id = int(keys.max())
    table = torch.zeros(max_id + 1, dtype=torch.long)
    valid = torch.zeros(max_id + 1, dtype=torch.bool)
    table[keys] = torch.from_numpy(values)
    valid[keys] = True
    return table, valid, max_id, gh_index


def _table_rows(
    tokens: torch.Tensor,
    lookup_valid: torch.Tensor,
    max_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = tokens.long()
    rows = values.clamp(0, max_id)
    keep = (
        (values >= 0)
        & (values <= max_id)
        & (values != PAD_ID)
        & lookup_valid[rows]
    )
    return rows, keep


def _lookup_coordinates(
    tokens: torch.Tensor,
    placekey_latlon: dict[int, tuple[float, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    coords, lookup_valid, max_id = _cached_table(
        _latlon_table, placekey_latlon
    )
    rows, valid = _table_rows(tokens, lookup_valid.to(tokens.device), max_id)
    return coords.to(tokens.device)[rows], valid


def per_sequence_travel_distance(
    tokens: torch.Tensor,
    placekey_latlon: dict[int, tuple[float, float]],
) -> torch.Tensor:
    gathered, valid = _lookup_coordinates(tokens, placekey_latlon)
    n, t = valid.shape
    if t < 2:
        return torch.zeros(n, dtype=gathered.dtype, device=gathered.device)

    ranks = valid.long().cumsum(dim=1) - 1
    valid_flat = valid.reshape(-1)
    row_idx = torch.arange(n, device=valid.device).unsqueeze(1)
    flat_dst = (row_idx * t + ranks.clamp_min(0)).reshape(-1)
    compact = torch.zeros_like(gathered).reshape(n * t, 2)
    compact[flat_dst[valid_flat]] = gathered.reshape(-1, 2)[valid_flat]
    compact = compact.view(n, t, 2)

    counts = valid.sum(dim=1)
    pair_idx = torch.arange(t - 1, device=valid.device).unsqueeze(0)
    pair_valid = pair_idx < (counts - 1).unsqueeze(1)
    lat1 = compact[:, :-1, 0]
    lon1 = compact[:, :-1, 1]
    lat2 = compact[:, 1:, 0]
    lon2 = compact[:, 1:, 1]
    dists = haversine_km(lat1, lon1, lat2, lon2)
    dists = torch.where(pair_valid, dists, torch.zeros_like(dists))
    return dists.sum(dim=1)


def _distance_bins(
    real: torch.Tensor, generated: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    max_value = float(torch.maximum(real.max(), generated.max()).item())

    def _bins(values: torch.Tensor) -> torch.Tensor:
        if max_value == 0.0:
            return torch.zeros_like(values, dtype=torch.long)
        return (
            (values / max_value * _DISTANCE_BINS)
            .long()
            .clamp_(0, _DISTANCE_BINS - 1)
        )

    return _bins(real), _bins(generated)


def _binned_jsd(real: torch.Tensor, generated: torch.Tensor) -> float:

    def _hist(bins: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(bins, minlength=_DISTANCE_BINS).float()
        return counts / counts.sum()

    real_bins, generated_bins = _distance_bins(real, generated)
    return jsd(_hist(real_bins), _hist(generated_bins))


def travel_distance_jsd(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    placekey_latlon: dict[int, tuple[float, float]],
) -> float:
    d_real = per_sequence_travel_distance(tokens_real, placekey_latlon)
    d_gen = per_sequence_travel_distance(tokens_gen, placekey_latlon)
    if d_real.numel() == 0 or d_gen.numel() == 0:
        return float("nan")
    return _binned_jsd(d_real, d_gen)


def radius_of_gyration_jsd(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    placekey_latlon: dict[int, tuple[float, float]],
) -> float | None:
    if not placekey_latlon:
        return None
    r = per_sequence_radius_of_gyration(tokens_real, placekey_latlon)
    g = per_sequence_radius_of_gyration(tokens_gen, placekey_latlon)
    if r.numel() == 0 or g.numel() == 0:
        return None
    return _binned_jsd(r, g)


def per_sequence_radius_of_gyration(
    tokens: torch.Tensor,
    placekey_latlon: dict[int, tuple[float, float]],
) -> torch.Tensor:
    gathered, valid = _lookup_coordinates(tokens, placekey_latlon)
    mask_f = valid.float().unsqueeze(-1)
    totals = (gathered * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).squeeze(-1)
    safe_counts = counts.clamp_min(1.0)
    mean = totals / safe_counts.unsqueeze(-1)
    mean_lat = mean[:, 0].unsqueeze(1).expand_as(gathered[..., 0])
    mean_lon = mean[:, 1].unsqueeze(1).expand_as(gathered[..., 1])
    distances = haversine_km(
        gathered[..., 0], gathered[..., 1], mean_lat, mean_lon
    )
    distances = torch.where(valid, distances, torch.zeros_like(distances))
    radius = torch.sqrt((distances * distances).sum(dim=1) / safe_counts)
    return torch.where(counts >= 2, radius, torch.zeros_like(radius))


def spatial_jsd(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    geohash_lookup: dict[int, str],
) -> float:
    gh_table, valid_tbl, max_id, gh_index = _cached_table(
        _geohash_table, geohash_lookup
    )
    n_hashes = len(gh_index)

    def _hist(tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.flatten()
        safe, keep = _table_rows(flat, valid_tbl.to(flat.device), max_id)
        idx = gh_table.to(flat.device)[safe][keep]
        counts = torch.bincount(idx, minlength=n_hashes).float()
        total = counts.sum()
        return counts / total if total > 0 else counts

    return jsd(_hist(tokens_real), _hist(tokens_gen))
