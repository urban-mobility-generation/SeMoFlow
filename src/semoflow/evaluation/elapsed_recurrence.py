from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from semoflow.evaluation.distribution import (
    jensen_shannon_divergence,
)

DEFINITION_VERSION = "elapsed_recurrence_v1"
DEFAULT_GRID_HOURS = 3.0
WEEK_HOURS = 168.0


@dataclasses.dataclass(frozen=True, slots=True)
class ReturnEvents:
    place_ids: np.ndarray
    hours: np.ndarray
    valid: np.ndarray


@dataclasses.dataclass(frozen=True, slots=True)
class _Side:
    returns: int
    decisions: int
    rows: int
    events: int
    truncated_rows: int
    dropped_events: int
    index_gaps: np.ndarray
    hour_gaps: np.ndarray
    adjacent: np.ndarray


def _collect(events: ReturnEvents, horizon_hours: float) -> _Side:
    valid = np.asarray(events.valid, dtype=bool)
    place_ids = np.asarray(events.place_ids)
    hours = np.asarray(events.hours, dtype=np.float64)
    index_gaps: list[int] = []
    hour_gaps: list[float] = []
    adjacent: list[bool] = []
    decisions = truncated_rows = dropped_events = n_events = 0
    for row_index in range(valid.shape[0]):
        mask = valid[row_index]
        ids = place_ids[row_index][mask]
        times = hours[row_index][mask]
        if not np.isfinite(times).all():
            raise ValueError(f"row {row_index} has a non-finite active time")
        if np.any(times < 0) or np.any(times > horizon_hours):
            raise ValueError(
                f"row {row_index} has a time outside [0, {horizon_hours}] h"
            )
        if np.any(np.diff(times) < 0):
            raise ValueError(f"row {row_index} has a backwards time")
        at_horizon = np.flatnonzero(times == horizon_hours)
        if at_horizon.size and at_horizon[0] + 1 < len(times):
            truncated_rows += 1
            dropped_events += int(len(times) - at_horizon[0] - 1)
            ids, times = ids[: at_horizon[0] + 1], times[: at_horizon[0] + 1]
        n_events += len(ids)
        decisions += max(len(ids) - 1, 0)
        previous: dict[Any, int] = {}
        for position, value in enumerate(ids.tolist()):
            if value in previous:
                earlier = previous[value]
                index_gaps.append(position - earlier)
                hour_gaps.append(float(times[position] - times[earlier]))
                adjacent.append(earlier == position - 1)
            previous[value] = position
    return _Side(
        returns=len(index_gaps),
        decisions=decisions,
        rows=int(valid.shape[0]),
        events=n_events,
        truncated_rows=truncated_rows,
        dropped_events=dropped_events,
        index_gaps=np.asarray(index_gaps, dtype=np.int64),
        hour_gaps=np.asarray(hour_gaps, dtype=np.float64),
        adjacent=np.asarray(adjacent, dtype=bool),
    )


def soft_time_histogram(
    gaps_hours: np.ndarray,
    *,
    grid_hours: float,
    n_bins: int,
    index_gaps: np.ndarray | None = None,
    n_index_bins: int = 1,
) -> np.ndarray:
    gaps = np.asarray(gaps_hours, dtype=np.float64)
    rows = (
        np.zeros(len(gaps), dtype=np.int64)
        if index_gaps is None
        else np.asarray(index_gaps, dtype=np.int64)
    )
    z = gaps / grid_hours
    lower = np.floor(z).astype(np.int64)
    upper_weight = z - lower
    counts = np.zeros((n_index_bins, n_bins), dtype=np.float64)
    np.add.at(counts, (rows, lower), 1.0 - upper_weight)
    np.add.at(counts, (rows, lower + 1), upper_weight)
    return counts if index_gaps is not None else counts[0]


def _rate(side: _Side) -> float:
    return side.returns / side.decisions if side.decisions else math.nan


def _composite(real: _Side, gen: _Side, q_real, q_gen) -> float:
    r_real, r_gen = _rate(real), _rate(gen)
    if math.isnan(r_real) or math.isnan(r_gen):
        return math.nan

    def vector(rate: float, q: np.ndarray) -> np.ndarray:
        total = q.sum()
        shape = q / total if total > 0 else np.zeros_like(q)
        return np.concatenate([[1.0 - rate], rate * shape])

    return jensen_shannon_divergence(
        vector(r_real, q_real), vector(r_gen, q_gen)
    )


def _summary(side: _Side, grid_hours: float) -> dict[str, Any]:
    gaps = side.hour_gaps
    day_multiple = np.round(gaps / 24.0)
    near_day = (day_multiple >= 1) & (
        np.abs(gaps - 24.0 * day_multiple) <= grid_hours
    )
    return {
        "rows": side.rows,
        "events": side.events,
        "returns": side.returns,
        "repeat_rate": _rate(side),
        "adjacent_return_share": (
            float(side.adjacent.mean()) if side.returns else math.nan
        ),
        "non_adjacent_returns": int((~side.adjacent).sum()),
        "zero_hour_returns": int((gaps == 0).sum()),
        "near_24h_multiple_share": (
            float(near_day.mean()) if side.returns else math.nan
        ),
        "mean_gap_hours": float(gaps.mean()) if side.returns else math.nan,
        "horizon_truncated_rows": side.truncated_rows,
        "horizon_dropped_events": side.dropped_events,
    }


def elapsed_recurrence_jsd(
    real: ReturnEvents,
    generated: ReturnEvents,
    *,
    grid_hours: float = DEFAULT_GRID_HOURS,
    horizon_hours: float = WEEK_HOURS,
) -> dict[str, Any]:
    real_side = _collect(real, horizon_hours)
    gen_side = _collect(generated, horizon_hours)
    n_bins = math.floor(horizon_hours / grid_hours) + 2
    q_real = soft_time_histogram(
        real_side.hour_gaps, grid_hours=grid_hours, n_bins=n_bins
    )
    q_gen = soft_time_histogram(
        gen_side.hour_gaps, grid_hours=grid_hours, n_bins=n_bins
    )
    n_index_bins = (
        max(
            int(real_side.index_gaps.max(initial=0)),
            int(gen_side.index_gaps.max(initial=0)),
        )
        + 1
    )
    joint_real, joint_gen = (
        soft_time_histogram(
            side.hour_gaps,
            grid_hours=grid_hours,
            n_bins=n_bins,
            index_gaps=side.index_gaps,
            n_index_bins=n_index_bins,
        )
        for side in (real_side, gen_side)
    )

    def non_adjacent(side: _Side) -> np.ndarray:
        return soft_time_histogram(
            side.hour_gaps[~side.adjacent],
            grid_hours=grid_hours,
            n_bins=n_bins,
        )

    return {
        "inter_return_time_jsd": jensen_shannon_divergence(q_real, q_gen),
        "return_index_time_joint_jsd": jensen_shannon_divergence(
            joint_real.ravel(), joint_gen.ravel()
        ),
        "rate_aware_return_time_jsd": _composite(
            real_side, gen_side, q_real, q_gen
        ),
        "non_adjacent_inter_return_time_jsd": jensen_shannon_divergence(
            non_adjacent(real_side), non_adjacent(gen_side)
        ),
        "real": _summary(real_side, grid_hours),
        "generated": _summary(gen_side, grid_hours),
        "protocol": {
            "definition_version": DEFINITION_VERSION,
            "grid_hours": grid_hours,
            "horizon_hours": horizon_hours,
            "time_bins": n_bins,
            "binning": "linear interpolation between adjacent centers",
            "horizon_rule": "keep events through the first at the horizon",
            "log_base": "e",
        },
    }


__all__ = [
    "ReturnEvents",
    "elapsed_recurrence_jsd",
    "soft_time_histogram",
]
