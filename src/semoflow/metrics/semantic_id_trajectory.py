from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, cast

import numpy as np
import polars as pl

from semoflow.metrics.divergence import (
    jensen_shannon_divergence,
)
from semoflow.metrics.semantic_id_uncertainty import (
    bootstrap_difference_ci,
    bootstrap_mean_ci,
)


def sample_stays_for_trajectory_eval(
    stays: pl.DataFrame,
    *,
    max_events: int,
    seed: int,
) -> pl.DataFrame:
    if max_events <= 0:
        return stays.head(0)
    if stays.height <= max_events or "device_key" not in stays.columns:
        return stays

    devices = stays.select("device_key").unique().sort("device_key")
    n_devices = devices.height
    if n_devices == 0:
        return stays.head(0)

    target_devices = max(
        1, int(np.floor(n_devices * max_events / stays.height))
    )
    target_devices = min(target_devices, n_devices)
    rng = np.random.RandomState(seed)
    selected_idx = np.sort(
        rng.choice(n_devices, size=target_devices, replace=False)
    )
    selected = devices[selected_idx, "device_key"].to_list()
    sampled = stays.filter(pl.col("device_key").is_in(selected))
    if sampled.height > max_events:
        sort_cols = ["device_key"]
        if "local_ts" in sampled.columns:
            sort_cols.append("local_ts")
        elif "timestamp" in sampled.columns:
            sort_cols.append("timestamp")
        sampled = sampled.sort(sort_cols).head(max_events)
    return sampled


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    return jensen_shannon_divergence(p, q)


def _hourly_distribution(hours: np.ndarray) -> np.ndarray:
    counts = np.bincount(hours.astype(np.int32), minlength=24)[:24]
    total = counts.sum()
    return np.zeros(24) if total == 0 else counts.astype(np.float64) / total


def _prepare_stays_with_sids(
    stays: pl.DataFrame,
    sid_lookup: pl.DataFrame,
    code_cols: list[str],
) -> pl.DataFrame:
    join_col = (
        "placekey"
        if "placekey" in stays.columns and "placekey" in sid_lookup.columns
        else "placekey_id"
    )
    working = stays.with_row_index("_trajectory_order")
    joined = working.join(
        sid_lookup.select([join_col, *code_cols]), on=join_col, how="inner"
    )
    poi_column = (
        "placekey_id" if "placekey_id" in joined.columns else "placekey"
    )
    extra_columns = [
        pl.concat_str(
            [pl.col(column).cast(pl.Utf8) for column in code_cols],
            separator="-",
        ).alias("sid_str"),
        pl.col(poi_column).cast(pl.Utf8).alias("poi_str"),
    ]
    if "hour" not in joined.columns and "local_ts" in joined.columns:
        extra_columns.append(pl.col("local_ts").dt.hour().alias("hour"))
    joined = joined.with_columns(extra_columns)
    sort_columns = ["device_key"] if "device_key" in joined.columns else []
    if "local_ts" in joined.columns:
        sort_columns.append("local_ts")
    elif "timestamp" in joined.columns:
        sort_columns.append("timestamp")
    sort_columns.append("_trajectory_order")
    if sort_columns:
        joined = joined.sort(sort_columns)
    return joined


def collapse_consecutive_pois(
    joined: pl.DataFrame,
    *,
    poi_column: str | None = None,
) -> tuple[pl.DataFrame, float, str]:
    resolved_poi_column = poi_column or (
        "poi_str"
        if "poi_str" in joined.columns
        else "placekey_id"
        if "placekey_id" in joined.columns
        else "placekey"
    )
    if (
        joined.is_empty()
        or "device_key" not in joined.columns
        or resolved_poi_column not in joined.columns
    ):
        return joined, 0.0, resolved_poi_column

    working = joined
    if "_trajectory_order" not in working.columns:
        working = working.with_row_index("_trajectory_order")
    sort_columns = ["device_key"]
    if "local_ts" in working.columns:
        sort_columns.append("local_ts")
    elif "timestamp" in working.columns:
        sort_columns.append("timestamp")
    sort_columns.append("_trajectory_order")
    working = working.sort(sort_columns)

    next_poi = pl.col(resolved_poi_column).shift(-1).over("device_key")
    adjacent = working.with_columns(next_poi.alias("_next_poi"))
    valid_pairs = adjacent.filter(pl.col("_next_poi").is_not_null())
    anchor_mean = (
        (valid_pairs[resolved_poi_column] == valid_pairs["_next_poi"])
        .cast(pl.Float64)
        .mean()
        if not valid_pairs.is_empty()
        else None
    )
    anchor_rate = (
        float(cast(Any, anchor_mean)) if anchor_mean is not None else 0.0
    )

    previous_poi = pl.col(resolved_poi_column).shift(1).over("device_key")
    collapsed = (
        working.with_columns(previous_poi.alias("_previous_poi"))
        .filter(
            pl.col("_previous_poi").is_null()
            | (pl.col(resolved_poi_column) != pl.col("_previous_poi"))
        )
        .drop("_previous_poi")
    )
    return collapsed, anchor_rate, resolved_poi_column


def _transition_frame(
    frame: pl.DataFrame,
    code_cols: list[str],
) -> pl.DataFrame:
    if frame.is_empty() or "device_key" not in frame.columns:
        return frame.head(0)
    expressions = [
        pl.col("poi_str").shift(-1).over("device_key").alias("next_poi_str")
    ]
    expressions.extend(
        pl.col(column).shift(-1).over("device_key").alias(f"next_{column}")
        for column in code_cols
    )
    return frame.with_columns(expressions).filter(
        pl.col("next_poi_str").is_not_null()
    )


def _prefix_key(
    row: dict[str, Any], columns: list[str], prefix: int
) -> tuple[int, ...]:
    return tuple(int(row[column]) for column in columns[:prefix])


def _prefix_transition_metrics(
    transitions: pl.DataFrame,
    code_cols: list[str],
    *,
    min_transitions: int,
) -> tuple[
    dict[int, float],
    dict[int, float],
    dict[int, float | None],
]:
    entropy: dict[int, float] = {}
    prediction_accuracy: dict[int, float] = {}
    next_poi_mi: dict[int, float | None] = {}
    if transitions.is_empty():
        return entropy, prediction_accuracy, next_poi_mi

    rows = transitions.select(
        [
            *code_cols,
            *[f"next_{column}" for column in code_cols],
            "next_poi_str",
        ]
    ).iter_rows(named=True)
    transition_rows = list(rows)
    for depth in range(1, len(code_cols) + 1):
        by_source: dict[tuple[int, ...], dict[tuple[int, ...], int]] = {}
        source_total: dict[tuple[int, ...], int] = {}
        for row in transition_rows:
            source = _prefix_key(row, code_cols, depth)
            target = tuple(
                int(row[f"next_{column}"]) for column in code_cols[:depth]
            )
            target_counts = by_source.setdefault(source, {})
            target_counts[target] = target_counts.get(target, 0) + 1
            source_total[source] = source_total.get(source, 0) + 1

        eligible = {
            source: counts
            for source, counts in by_source.items()
            if source_total[source] >= min_transitions
        }
        entropy_values: list[float] = []
        correct = 0
        eligible_total = 0
        for _source, counts in eligible.items():
            total = sum(counts.values())
            probabilities = np.asarray(list(counts.values()), dtype=np.float64)
            probabilities /= total
            if len(probabilities) <= 1:
                entropy_values.append(0.0)
            else:
                entropy_values.append(
                    float(
                        -np.sum(probabilities * np.log(probabilities))
                        / np.log(len(probabilities))
                    )
                )
            correct += max(counts.values())
            eligible_total += total
        entropy[depth] = (
            float(np.mean(entropy_values)) if entropy_values else 0.0
        )
        prediction_accuracy[depth] = (
            correct / eligible_total if eligible_total else 0.0
        )

        source_labels: list[tuple[int, ...]] = []
        next_poi_labels: list[str] = []
        for row in transition_rows:
            source_labels.append(_prefix_key(row, code_cols, depth))
            next_poi_labels.append(str(row["next_poi_str"]))
        try:
            from sklearn.metrics import mutual_info_score

            source_ids = {
                label: index for index, label in enumerate(set(source_labels))
            }
            poi_ids = {
                label: index
                for index, label in enumerate(set(next_poi_labels))
            }
            next_poi_mi[depth] = float(
                mutual_info_score(
                    [source_ids[label] for label in source_labels],
                    [poi_ids[label] for label in next_poi_labels],
                )
            )
        except ImportError:
            next_poi_mi[depth] = None

    return entropy, prediction_accuracy, next_poi_mi


def _profile_coherence_by_prefix(
    frame: pl.DataFrame,
    code_cols: list[str],
    *,
    hour_col: str | None,
    min_visits: int,
    max_prefix_groups: int,
    seed: int,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
) -> tuple[dict[int, float], dict[int, float], dict[int, list[float]]]:
    if hour_col is None or frame.is_empty():
        return {}, {}, {}
    grouped = (
        frame.group_by("poi_str")
        .agg(
            *[
                pl.col(hour_col)
                .filter(pl.col(hour_col) == hour)
                .len()
                .alias(f"h{hour}")
                for hour in range(24)
            ],
            pl.len().alias("n_visits"),
            *[pl.col(column).first().alias(column) for column in code_cols],
        )
        .filter(pl.col("n_visits") >= min_visits)
    )
    if grouped.is_empty():
        return {}, {}, {}

    hour_columns = [f"h{hour}" for hour in range(24)]
    distributions = grouped.select(hour_columns).to_numpy().astype(np.float64)
    totals = distributions.sum(axis=1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    distributions /= totals
    prefixes = np.column_stack(
        [grouped[column].to_numpy() for column in code_cols]
    ).astype(np.int64, copy=False)
    rng = np.random.RandomState(seed)
    coherence: dict[int, float] = {}
    baseline: dict[int, float] = {}
    effect_ci: dict[int, list[float]] = {}
    for depth in range(1, len(code_cols) + 1):
        prefix_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index in range(prefixes.shape[0]):
            prefix = tuple(
                int(value) for value in prefixes[index, :depth].tolist()
            )
            prefix_groups[prefix].append(index)
        prefix_groups = {
            prefix: indices
            for prefix, indices in prefix_groups.items()
            if len(indices) >= 2
        }
        if not prefix_groups:
            continue
        selected_prefixes = list(prefix_groups)
        if len(selected_prefixes) > max_prefix_groups:
            selected_prefixes = [
                selected_prefixes[int(index)]
                for index in rng.choice(
                    len(selected_prefixes),
                    size=max_prefix_groups,
                    replace=False,
                )
            ]
        within: list[float] = []
        for prefix in selected_prefixes:
            indices = prefix_groups[prefix]
            if len(indices) > 50:
                indices = list(rng.choice(indices, size=50, replace=False))
            within.extend(
                _jsd(distributions[left], distributions[right])
                for left, right in combinations(indices, 2)
            )
        if within:
            coherence[depth] = float(np.mean(within))
        n_profiles = len(distributions)
        random_jsds: list[float] = []
        n_pairs = min(10000, n_profiles * (n_profiles - 1) // 2)
        if n_pairs > 0:
            left = rng.randint(0, n_profiles, size=n_pairs)
            right = rng.randint(0, n_profiles, size=n_pairs)
            valid = left != right
            if valid.any():
                random_jsds = [
                    _jsd(distributions[i], distributions[j])
                    for i, j in zip(left[valid], right[valid], strict=True)
                ]
                baseline[depth] = float(np.mean(random_jsds))
        if within and random_jsds and bootstrap_resamples > 0:
            interval = bootstrap_difference_ci(
                within,
                random_jsds,
                resamples=bootstrap_resamples,
                confidence=bootstrap_confidence,
                seed=seed + depth,
            )
            if interval is not None:
                effect_ci[depth] = interval
    return coherence, baseline, effect_ci


def _prefix_sharing_bootstrap_cis(
    transitions: pl.DataFrame,
    code_cols: list[str],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, dict[int, list[float]]]:
    intervals: dict[str, dict[int, list[float]]] = {
        "users": {},
        "trajectories": {},
        "time_periods": {},
    }
    if transitions.is_empty() or resamples <= 0:
        return intervals

    frame = transitions
    group_columns: dict[str, str] = {}
    if "device_key" in frame.columns:
        group_columns["users"] = "device_key"
        group_columns["trajectories"] = (
            "trajectory_id"
            if "trajectory_id" in frame.columns
            else "device_key"
        )
    timestamp_column = next(
        (
            column
            for column in ("local_ts", "timestamp")
            if column in frame.columns
        ),
        None,
    )
    if timestamp_column is not None:
        frame = frame.with_columns(
            pl.col(timestamp_column)
            .cast(pl.Datetime)
            .dt.date()
            .alias("_time_period")
        )
        group_columns["time_periods"] = "_time_period"

    for depth in range(1, len(code_cols) + 1):
        frame = frame.with_columns(
            pl.all_horizontal(
                [
                    pl.col(column) == pl.col(f"next_{column}")
                    for column in code_cols[:depth]
                ]
            )
            .cast(pl.Float64)
            .alias("_prefix_match")
        )
        for unit, group_column in group_columns.items():
            grouped = (
                frame.select([group_column, "_prefix_match"])
                .drop_nulls()
                .group_by(group_column)
                .agg(pl.col("_prefix_match").mean())
            )
            interval = bootstrap_mean_ci(
                grouped["_prefix_match"].to_numpy(),
                resamples=resamples,
                confidence=confidence,
                seed=seed + depth * 101 + len(unit),
            )
            if interval is not None:
                intervals[unit][depth] = interval
    return intervals


def trajectory_behavioral_metrics(
    stays: pl.DataFrame,
    sid_lookup: pl.DataFrame,
    code_cols: list[str],
    *,
    min_visits: int = 20,
    min_transitions: int = 10,
    max_prefix_groups: int = 200,
    seed: int = 42,
    bootstrap_resamples: int = 0,
    bootstrap_confidence: float = 0.95,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "temporal_distinctiveness": None,
        "transition_entropy": None,
        "transition_entropy_by_prefix": None,
        "prefix_sharing_rate": None,
        "prefix_temporal_coherence": None,
        "prefix_temporal_coherence_baseline": None,
        "prefix_temporal_coherence_by_depth": None,
        "prefix_temporal_coherence_baseline_by_depth": None,
        "prefix_temporal_coherence_effect_ci_by_depth": None,
        "next_prefix_prediction_accuracy": None,
        "next_poi_mutual_information": None,
        "anchor_repetition_rate": 0.0,
        "raw_event_count": 0,
        "collapsed_event_count": 0,
        "bootstrap_confidence_intervals": {
            "users": {},
            "trajectories": {},
            "time_periods": {},
        },
    }
    joined = _prepare_stays_with_sids(stays, sid_lookup, code_cols)
    result["raw_event_count"] = joined.height
    collapsed, anchor_rate, _poi_column = collapse_consecutive_pois(joined)
    result["anchor_repetition_rate"] = anchor_rate
    result["collapsed_event_count"] = collapsed.height
    if collapsed.is_empty():
        result["prefix_sharing_rate"] = {
            depth: 0.0 for depth in range(1, len(code_cols) + 1)
        }
        return result

    joined = collapsed
    hour_col = "hour" if "hour" in joined.columns else None

    if hour_col is not None:
        all_hours = joined[hour_col].drop_nulls().to_numpy()
        if len(all_hours) > 0:
            global_dist = _hourly_distribution(all_hours)
            hour_hist = (
                joined.group_by("sid_str")
                .agg(
                    *[
                        pl.col(hour_col)
                        .filter(pl.col(hour_col) == hour)
                        .len()
                        .alias(f"h{hour}")
                        for hour in range(24)
                    ],
                    pl.len().alias("n_visits"),
                )
                .filter(pl.col("n_visits") >= min_visits)
            )
            if not hour_hist.is_empty():
                hour_columns = [f"h{hour}" for hour in range(24)]
                counts = (
                    hour_hist.select(hour_columns)
                    .to_numpy()
                    .astype(np.float64)
                )
                totals = counts.sum(axis=1, keepdims=True)
                totals = np.where(totals > 0, totals, 1.0)
                result["temporal_distinctiveness"] = float(
                    np.mean(
                        [
                            _jsd(row / total, global_dist)
                            for row, total in zip(
                                counts, totals[:, 0], strict=True
                            )
                        ]
                    )
                )

    transitions = _transition_frame(joined, code_cols)
    (
        transition_entropy,
        next_prefix_accuracy,
        next_poi_mi,
    ) = _prefix_transition_metrics(
        transitions,
        code_cols,
        min_transitions=min_transitions,
    )
    result["transition_entropy_by_prefix"] = transition_entropy
    result["next_prefix_prediction_accuracy"] = next_prefix_accuracy
    result["next_poi_mutual_information"] = next_poi_mi
    if transition_entropy:
        result["transition_entropy"] = transition_entropy[len(code_cols)]

    prefix_rates: dict[int, float] = {}
    for depth in range(1, len(code_cols) + 1):
        frame = transitions.with_columns(
            pl.all_horizontal(
                [
                    pl.col(column) == pl.col(f"next_{column}")
                    for column in code_cols[:depth]
                ]
            ).alias("_match")
        )
        prefix_rates[depth] = (
            0.0
            if frame.is_empty()
            else float(cast(Any, frame["_match"].cast(pl.Float64).mean()))
        )
    result["prefix_sharing_rate"] = prefix_rates if prefix_rates else None

    coherence, coherence_baseline, coherence_effect_ci = (
        _profile_coherence_by_prefix(
            joined,
            code_cols,
            hour_col=hour_col,
            min_visits=min_visits,
            max_prefix_groups=max_prefix_groups,
            seed=seed,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_confidence=bootstrap_confidence,
        )
    )
    result["prefix_temporal_coherence_by_depth"] = coherence or None
    result["prefix_temporal_coherence_baseline_by_depth"] = (
        coherence_baseline or None
    )
    result["prefix_temporal_coherence_effect_ci_by_depth"] = (
        coherence_effect_ci or None
    )
    result["prefix_temporal_coherence"] = coherence.get(1)
    result["prefix_temporal_coherence_baseline"] = coherence_baseline.get(1)
    result["bootstrap_confidence_intervals"] = _prefix_sharing_bootstrap_cis(
        transitions,
        code_cols,
        resamples=bootstrap_resamples,
        confidence=bootstrap_confidence,
        seed=seed,
    )
    return result


__all__ = [
    "collapse_consecutive_pois",
    "sample_stays_for_trajectory_eval",
    "trajectory_behavioral_metrics",
]
