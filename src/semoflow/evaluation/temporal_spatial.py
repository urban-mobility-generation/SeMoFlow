from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from semoflow.data import N_PART_OF_DAY_BINS, N_TIME_TOKENS
from semoflow.evaluation.distribution import (
    jensen_shannon_divergence,
)

TIME_TOKEN_MIDPOINT_HOURS = np.array([3, 8, 12, 16, 20, 23], dtype=np.int64)
GAP_EDGES_HOURS = np.array([0, 2, 4, 8, 12, 24, 48, 96, 168, np.inf])
DISTANCE_EDGES_KM = np.array(
    [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, np.inf]
)
EARTH_RADIUS_KM = 6371.0088
N_GAP_BINS = len(GAP_EDGES_HOURS)
N_DISTANCE_BINS = len(DISTANCE_EDGES_KM)


@dataclasses.dataclass(frozen=True, slots=True)
class TemporalSpatialEvents:
    time_tokens: np.ndarray
    geohash: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        valid = np.asarray(self.valid, dtype=bool)
        tokens = np.asarray(self.time_tokens)[valid]
        if np.any((tokens < 0) | (tokens >= N_TIME_TOKENS)):
            raise ValueError(
                f"active time tokens must lie in [0, {N_TIME_TOKENS})"
            )


def events_from_place_ids(
    place_ids: np.ndarray,
    time_tokens: np.ndarray,
    valid: np.ndarray,
    *,
    coordinates: Mapping[int, tuple[float, float]],
    geohashes: Mapping[int, str],
) -> TemporalSpatialEvents:
    place_ids = np.asarray(place_ids, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    unique, inverse = np.unique(place_ids, return_inverse=True)
    inverse = inverse.reshape(place_ids.shape)
    cells = np.array(
        [geohashes.get(int(pid)) or None for pid in unique], dtype=object
    )
    points = np.array(
        [coordinates.get(int(pid), (np.nan, np.nan)) for pid in unique],
        dtype=np.float64,
    ).reshape(-1, 2)
    return TemporalSpatialEvents(
        time_tokens=np.where(valid, np.asarray(time_tokens), 0),
        geohash=cells[inverse],
        latitude=points[inverse, 0],
        longitude=points[inverse, 1],
        valid=valid,
    )


def _haversine_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.radians(a), np.radians(b)
    delta = b - a
    term = (
        np.sin(delta[:, 0] / 2) ** 2
        + np.cos(a[:, 0]) * np.cos(b[:, 0]) * np.sin(delta[:, 1] / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(term, 0, 1)))


def _gap_distance_counts(
    events: TemporalSpatialEvents,
) -> tuple[np.ndarray, dict[str, int]]:
    valid = np.asarray(events.valid, dtype=bool)
    tokens = np.asarray(events.time_tokens, dtype=np.int64)
    pairs = valid[:, :-1] & valid[:, 1:]
    hours = (tokens // N_PART_OF_DAY_BINS) * 24 + TIME_TOKEN_MIDPOINT_HOURS[
        np.clip(tokens % N_PART_OF_DAY_BINS, 0, N_PART_OF_DAY_BINS - 1)
    ]
    gaps = (hours[:, 1:] - hours[:, :-1])[pairs]
    gap_bins = np.searchsorted(GAP_EDGES_HOURS, gaps, side="right") - 1
    gap_bins[gaps < 0] = N_GAP_BINS - 1
    points = np.stack(
        [np.asarray(events.latitude), np.asarray(events.longitude)], axis=-1
    )
    start, end = points[:, :-1][pairs], points[:, 1:][pairs]
    located = np.isfinite(start).all(1) & np.isfinite(end).all(1)
    distance_bins = np.full(len(gaps), N_DISTANCE_BINS - 1, dtype=np.int64)
    distance_bins[located] = (
        np.searchsorted(
            DISTANCE_EDGES_KM,
            _haversine_km(start[located], end[located]),
            side="right",
        )
        - 1
    )
    counts = np.bincount(
        gap_bins * N_DISTANCE_BINS + distance_bins,
        minlength=N_GAP_BINS * N_DISTANCE_BINS,
    ).reshape(N_GAP_BINS, N_DISTANCE_BINS)
    diagnostics = {
        "pairs": len(gaps),
        "negative_gap_pairs": int((gaps < 0).sum()),
        "zero_quantized_gap_pairs": int((gaps == 0).sum()),
        "missing_coordinate_pairs": int((~located).sum()),
    }
    return counts, diagnostics


def _compare(real: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    return {
        "joint_jsd": jensen_shannon_divergence(
            real.ravel(), generated.ravel()
        ),
        "first_marginal_jsd": jensen_shannon_divergence(
            real.sum(1), generated.sum(1)
        ),
        "second_marginal_jsd": jensen_shannon_divergence(
            real.sum(0), generated.sum(0)
        ),
    }


def temporal_spatial_jsd(
    real: TemporalSpatialEvents,
    generated: TemporalSpatialEvents,
) -> dict[str, dict[str, Any]]:
    real_valid = np.asarray(real.valid, dtype=bool)
    generated_valid = np.asarray(generated.valid, dtype=bool)
    real_cells = np.asarray(real.geohash, dtype=object)[real_valid]
    generated_cells = np.asarray(generated.geohash, dtype=object)[
        generated_valid
    ]
    vocabulary = {
        cell: index + 1
        for index, cell in enumerate(
            sorted({c for c in (*real_cells, *generated_cells) if c})
        )
    }
    n_cells = len(vocabulary) + 1

    def time_location(
        events: TemporalSpatialEvents, valid: np.ndarray, cells: np.ndarray
    ) -> np.ndarray:
        codes = np.array([vocabulary.get(c, 0) for c in cells], dtype=np.int64)
        tokens = np.asarray(events.time_tokens, dtype=np.int64)[valid]
        return np.bincount(
            tokens * n_cells + codes, minlength=N_TIME_TOKENS * n_cells
        ).reshape(N_TIME_TOKENS, n_cells)

    real_tl = time_location(real, real_valid, real_cells)
    generated_tl = time_location(generated, generated_valid, generated_cells)
    real_gd, real_pairs = _gap_distance_counts(real)
    generated_gd, generated_pairs = _gap_distance_counts(generated)
    tl = _compare(real_tl, generated_tl)
    gd = _compare(real_gd, generated_gd)

    def diagnostics(
        valid: np.ndarray, cells: np.ndarray, pairs: dict[str, int]
    ) -> dict[str, int]:
        return {
            "events": int(valid.sum()),
            "missing_geohash_events": int(sum(1 for c in cells if not c)),
            **pairs,
        }

    return {
        "time_location": {
            "joint_jsd": tl["joint_jsd"],
            "time_marginal_jsd": tl["first_marginal_jsd"],
            "location_marginal_jsd": tl["second_marginal_jsd"],
        },
        "gap_distance": {
            "joint_jsd": gd["joint_jsd"],
            "gap_marginal_jsd": gd["first_marginal_jsd"],
            "distance_marginal_jsd": gd["second_marginal_jsd"],
        },
        "real": diagnostics(real_valid, real_cells, real_pairs),
        "generated": diagnostics(
            generated_valid, generated_cells, generated_pairs
        ),
        "protocol": {
            "time_tokens": "weekday x six circadian intervals (42)",
            "location": "geohash_5 plus a missing cell",
            "gap": "token-midpoint hours (quantized proxy)",
            "gap_edges_hours": GAP_EDGES_HOURS[:-1].tolist(),
            "distance_edges_km": DISTANCE_EDGES_KM[:-1].tolist(),
            "units": "natural-log JSD in nats",
        },
    }


__all__ = [
    "N_GAP_BINS",
    "TemporalSpatialEvents",
    "events_from_place_ids",
    "temporal_spatial_jsd",
]
