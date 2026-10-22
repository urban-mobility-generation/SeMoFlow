from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from semoflow.config.constants import PAD_ID
from semoflow.metrics.trajectory.distribution import jsd
from semoflow.metrics.trajectory.identity_dynamics import (
    cluster_histogram_jsd_bootstrap,
)

_EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


_CACHE_CAPACITY = 16
_latlon_cache: list[
    tuple[
        dict[int, tuple[float, float]],
        tuple[torch.Tensor, torch.Tensor, int],
    ]
] = []
_geohash_cache: list[
    tuple[
        dict[int, str],
        tuple[torch.Tensor, torch.Tensor, int, dict[str, int]],
    ]
] = []


def _lookup_cache(cache: list[tuple[Any, Any]], key: Any) -> Any | None:
    for idx, (stored_key, value) in enumerate(cache):
        if stored_key is key:
            if idx != len(cache) - 1:
                cache.append(cache.pop(idx))
            return value
    return None


def _store_cache(cache: list[tuple[Any, Any]], key: Any, value: Any) -> None:
    if len(cache) >= _CACHE_CAPACITY:
        cache.pop(0)
    cache.append((key, value))


def _build_latlon_table(
    placekey_latlon: dict[int, tuple[float, float]],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    cached = _lookup_cache(_latlon_cache, placekey_latlon)
    if cached is not None:
        return cached
    if not placekey_latlon:
        result = (torch.zeros(0, 2), torch.zeros(0, dtype=torch.bool), -1)
        _store_cache(_latlon_cache, placekey_latlon, result)
        return result
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
    result = (coords, valid, max_id)
    _store_cache(_latlon_cache, placekey_latlon, result)
    return result


def _build_geohash_table(
    geohash_lookup: dict[int, str],
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, int]]:
    cached = _lookup_cache(_geohash_cache, geohash_lookup)
    if cached is not None:
        return cached
    if not geohash_lookup:
        empty: dict[str, int] = {}
        result = (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.bool),
            -1,
            empty,
        )
        _store_cache(_geohash_cache, geohash_lookup, result)
        return result
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
    result = (table, valid, max_id, gh_index)
    _store_cache(_geohash_cache, geohash_lookup, result)
    return result


def _gather_coords(
    tokens: torch.Tensor,
    coords: torch.Tensor,
    lookup_valid: torch.Tensor,
    max_id: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coords.device != tokens.device:
        coords = coords.to(tokens.device)
        lookup_valid = lookup_valid.to(tokens.device)
    tokens_long = tokens.long()
    in_range = (tokens_long >= 0) & (tokens_long <= max_id)
    safe_idx = tokens_long.clamp(0, max_id)
    gathered = coords[safe_idx]
    valid = in_range & (tokens_long != pad_id) & lookup_valid[safe_idx]
    return gathered, valid


def _pairwise_haversine_km(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor,
    lon2: torch.Tensor,
) -> torch.Tensor:
    phi1 = torch.deg2rad(lat1)
    phi2 = torch.deg2rad(lat2)
    dphi = torch.deg2rad(lat2 - lat1)
    dlam = torch.deg2rad(lon2 - lon1)
    a = (
        torch.sin(dphi / 2) ** 2
        + torch.cos(phi1) * torch.cos(phi2) * torch.sin(dlam / 2) ** 2
    )
    a = a.clamp(0.0, 1.0)
    return 2 * _EARTH_RADIUS_KM * torch.arcsin(torch.sqrt(a))


def centroid_geo_error(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    centroids: dict[int, tuple[float, float]],
    *,
    pad_id: int = PAD_ID,
) -> float:
    coords_tbl, valid_tbl, max_id = _build_latlon_table(centroids)
    if max_id < 0:
        return 0.0

    def _mean_centroid(tokens: torch.Tensor) -> tuple[float, float] | None:
        gathered, valid = _gather_coords(
            tokens, coords_tbl, valid_tbl, max_id, pad_id
        )
        if not valid.any():
            return None
        gathered = gathered.to(dtype=torch.float64)
        mask_f = valid.to(dtype=torch.float64).unsqueeze(-1)
        totals = (gathered * mask_f).sum(dim=(0, 1))
        denom = mask_f.sum(dim=(0, 1)).clamp_min(1.0)
        mean = totals / denom
        return float(mean[0].item()), float(mean[1].item())

    c_real = _mean_centroid(tokens_real)
    c_gen = _mean_centroid(tokens_gen)
    if c_real is None or c_gen is None:
        return 0.0
    return haversine_km(c_real[0], c_real[1], c_gen[0], c_gen[1])


def per_sequence_travel_distance(
    tokens: torch.Tensor,
    placekey_latlon: dict[int, tuple[float, float]],
    *,
    pad_id: int = PAD_ID,
) -> torch.Tensor:
    coords_tbl, valid_tbl, max_id = _build_latlon_table(placekey_latlon)
    if max_id < 0:
        return torch.zeros(tokens.size(0))
    gathered, valid = _gather_coords(
        tokens, coords_tbl, valid_tbl, max_id, pad_id
    )
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
    dists = _pairwise_haversine_km(lat1, lon1, lat2, lon2)
    dists = torch.where(pair_valid, dists, torch.zeros_like(dists))
    return dists.sum(dim=1)


def travel_distance_jsd(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    placekey_latlon: dict[int, tuple[float, float]],
    *,
    pad_id: int = PAD_ID,
    n_bins: int = 50,
) -> float:
    d_real = per_sequence_travel_distance(
        tokens_real, placekey_latlon, pad_id=pad_id
    )
    d_gen = per_sequence_travel_distance(
        tokens_gen, placekey_latlon, pad_id=pad_id
    )
    if d_real.numel() == 0 or d_gen.numel() == 0:
        return float("nan")
    max_dist = float(torch.maximum(d_real.max(), d_gen.max()).item())
    if max_dist == 0.0:
        return 0.0

    def _hist(dists: torch.Tensor) -> torch.Tensor:
        idx = (dists / max_dist * n_bins).long().clamp_(0, n_bins - 1)
        counts = torch.bincount(idx, minlength=n_bins).float()
        total = counts.sum()
        return counts / total if total > 0 else counts

    return jsd(_hist(d_real), _hist(d_gen))


def radius_of_gyration_jsd(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    placekey_latlon: dict[int, tuple[float, float]],
    *,
    pad_id: int = PAD_ID,
    n_bins: int = 50,
) -> float | None:
    if not placekey_latlon:
        return None
    r = per_sequence_radius_of_gyration(
        tokens_real, placekey_latlon, pad_id=pad_id
    )
    g = per_sequence_radius_of_gyration(
        tokens_gen, placekey_latlon, pad_id=pad_id
    )
    if r.numel() == 0 or g.numel() == 0:
        return None
    max_val = float(torch.maximum(r.max(), g.max()).item())
    if max_val == 0.0:
        return 0.0

    def _hist(vals: torch.Tensor) -> torch.Tensor:
        idx = (vals / max_val * n_bins).long().clamp_(0, n_bins - 1)
        counts = torch.bincount(idx, minlength=n_bins).float()
        total = counts.sum()
        return counts / total if total > 0 else counts

    return jsd(_hist(r), _hist(g))


def per_sequence_radius_of_gyration(
    tokens: torch.Tensor,
    placekey_latlon: dict[int, tuple[float, float]],
    *,
    pad_id: int = PAD_ID,
) -> torch.Tensor:
    coords_tbl, valid_tbl, max_id = _build_latlon_table(placekey_latlon)
    if max_id < 0:
        return torch.zeros(tokens.size(0))
    gathered, valid = _gather_coords(
        tokens, coords_tbl, valid_tbl, max_id, pad_id
    )
    mask_f = valid.float().unsqueeze(-1)
    totals = (gathered * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).squeeze(-1)
    safe_counts = counts.clamp_min(1.0)
    mean = totals / safe_counts.unsqueeze(-1)
    mean_lat = mean[:, 0].unsqueeze(1).expand_as(gathered[..., 0])
    mean_lon = mean[:, 1].unsqueeze(1).expand_as(gathered[..., 1])
    distances = _pairwise_haversine_km(
        gathered[..., 0], gathered[..., 1], mean_lat, mean_lon
    )
    distances = torch.where(valid, distances, torch.zeros_like(distances))
    radius = torch.sqrt((distances * distances).sum(dim=1) / safe_counts)
    return torch.where(counts >= 2, radius, torch.zeros_like(radius))


def _scalar_histogram_contributions(
    real_values: torch.Tensor,
    generated_values: torch.Tensor,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    if real_values.shape != generated_values.shape:
        raise ValueError("paired scalar distributions must align by row")
    if real_values.numel() == 0:
        empty = np.zeros((0, n_bins), dtype=np.int32)
        return empty, empty.copy()
    max_value = float(
        torch.maximum(real_values.max(), generated_values.max()).item()
    )

    def _contributions(values: torch.Tensor) -> np.ndarray:
        out = np.zeros((len(values), n_bins), dtype=np.int32)
        if not len(values):
            return out
        if max_value == 0.0:
            indices = np.zeros(len(values), dtype=np.int64)
        else:
            indices = (
                (values / max_value * n_bins)
                .long()
                .clamp_(0, n_bins - 1)
                .cpu()
                .numpy()
            )
        out[np.arange(len(values)), indices] = 1
        return out

    return _contributions(real_values), _contributions(generated_values)


def _geohash_contributions(
    tokens: torch.Tensor,
    geohash_lookup: dict[int, str],
    *,
    pad_id: int,
) -> np.ndarray:
    table, lookup_valid, max_id, geohash_index = _build_geohash_table(
        geohash_lookup
    )
    out = np.zeros((tokens.size(0), len(geohash_index)), dtype=np.int32)
    if max_id < 0 or not geohash_index:
        return out
    values = tokens.long().cpu()
    in_range = (values >= 0) & (values <= max_id)
    safe = values.clamp(0, max_id)
    keep = in_range & (values != pad_id) & lookup_valid[safe]
    rows = torch.arange(values.size(0)).unsqueeze(1).expand_as(values)
    np.add.at(
        out,
        (rows[keep].numpy(), table[safe][keep].numpy()),
        1,
    )
    return out


def spatial_guardrail_uncertainty(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    placekey_latlon: dict[int, tuple[float, float]],
    *,
    geohash_lookup: dict[int, str] | None = None,
    cluster_ids: Sequence[str] | None = None,
    pad_id: int = PAD_ID,
    bootstrap_seed: int = 20260714,
    bootstrap_resamples: int = 1000,
    compute_device: torch.device | str | None = None,
) -> dict[str, Any]:
    if tokens_real.shape != tokens_gen.shape:
        raise ValueError("paired spatial trajectories must have equal shape")
    travel = _scalar_histogram_contributions(
        per_sequence_travel_distance(
            tokens_real, placekey_latlon, pad_id=pad_id
        ),
        per_sequence_travel_distance(
            tokens_gen, placekey_latlon, pad_id=pad_id
        ),
        n_bins=50,
    )
    radius = _scalar_histogram_contributions(
        per_sequence_radius_of_gyration(
            tokens_real, placekey_latlon, pad_id=pad_id
        ),
        per_sequence_radius_of_gyration(
            tokens_gen, placekey_latlon, pad_id=pad_id
        ),
        n_bins=50,
    )
    result = {
        "travel_distance_jsd": cluster_histogram_jsd_bootstrap(
            *travel,
            cluster_ids=cluster_ids,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
            compute_device=compute_device,
        ),
        "radius_of_gyration_jsd": cluster_histogram_jsd_bootstrap(
            *radius,
            cluster_ids=cluster_ids,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
            compute_device=compute_device,
        ),
    }
    if geohash_lookup is not None:
        result["geohash_spatial_jsd"] = cluster_histogram_jsd_bootstrap(
            _geohash_contributions(tokens_real, geohash_lookup, pad_id=pad_id),
            _geohash_contributions(tokens_gen, geohash_lookup, pad_id=pad_id),
            cluster_ids=cluster_ids,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
            compute_device=compute_device,
        )
    return result


def spatial_jsd(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    geohash_lookup: dict[int, str],
    *,
    pad_id: int = PAD_ID,
) -> float:
    gh_table, valid_tbl, max_id, gh_index = _build_geohash_table(
        geohash_lookup
    )
    if max_id < 0:
        return float("nan")
    n_hashes = len(gh_index)

    def _hist(tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.flatten().long()
        local_table = gh_table.to(flat.device)
        local_valid = valid_tbl.to(flat.device)
        in_range = (flat >= 0) & (flat <= max_id)
        safe = flat.clamp(0, max_id)
        keep = in_range & (flat != pad_id) & local_valid[safe]
        idx = local_table[safe][keep]
        counts = torch.bincount(idx, minlength=n_hashes).float()
        total = counts.sum()
        return counts / total if total > 0 else counts

    return jsd(_hist(tokens_real), _hist(tokens_gen))
