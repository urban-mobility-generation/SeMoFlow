from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Sequence
from typing import Any

import numpy as np

from semoflow.config.constants import PAD_ID
from semoflow.metrics.divergence import jensen_shannon_divergence

type MotifSignature = tuple[int, tuple[tuple[int, int], ...]]


def _as_rows(values: Any, *, name: str) -> list[list[Any]]:
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        rows = [
            list(row) if not isinstance(row, (str, bytes)) else [row]
            for row in values
        ]
    except TypeError as exc:
        raise ValueError(f"{name} must be a two-dimensional sequence") from exc
    if any(not isinstance(row, list) for row in rows):
        raise ValueError(f"{name} must be a two-dimensional sequence")
    return rows


def _scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _is_padding(value: Any, pad_id: Any) -> bool:
    value = _scalar(value)
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return bool(value == pad_id)


def _motif_for_day(
    locations: Iterable[Any],
    *,
    pad_id: Any,
) -> MotifSignature | None:
    node_ids: dict[Hashable, int] = {}
    edges: set[tuple[int, int]] = set()
    previous: int | None = None
    previous_location: Hashable | None = None

    for raw_location in locations:
        location = _scalar(raw_location)
        if _is_padding(location, pad_id):
            continue
        try:
            hash(location)
        except TypeError as exc:
            raise TypeError("mobility locations must be hashable") from exc
        if previous_location == location:
            continue
        current = node_ids.setdefault(location, len(node_ids))
        if previous is not None:
            edges.add((previous, current))
        previous = current
        previous_location = location

    if not node_ids:
        return None
    return len(node_ids), tuple(sorted(edges))


def daily_mobility_motifs(
    location_sequences: Sequence[Sequence[Any]] | Any,
    day_keys: Sequence[Sequence[Any]] | Any | None = None,
    *,
    pad_id: Any = PAD_ID,
) -> list[MotifSignature]:
    locations = _as_rows(location_sequences, name="location_sequences")
    keys = (
        _as_rows(day_keys, name="day_keys") if day_keys is not None else None
    )
    if keys is not None and len(keys) != len(locations):
        raise ValueError("day_keys must have the same number of rows")

    motifs: list[MotifSignature] = []
    for row_index, row_locations in enumerate(locations):
        if keys is None:
            motif = _motif_for_day(row_locations, pad_id=pad_id)
            if motif is not None:
                motifs.append(motif)
            continue
        row_keys = keys[row_index]
        if len(row_keys) != len(row_locations):
            raise ValueError(
                "day_keys rows must align with location_sequences rows"
            )
        grouped: dict[Hashable, list[Any]] = defaultdict(list)
        for location, raw_day in zip(row_locations, row_keys, strict=True):
            if _is_padding(location, pad_id):
                continue
            day = _scalar(raw_day)
            try:
                hash(day)
            except TypeError as exc:
                raise TypeError("mobility day keys must be hashable") from exc
            grouped[day].append(location)
        for day_locations in grouped.values():
            motif = _motif_for_day(day_locations, pad_id=pad_id)
            if motif is not None:
                motifs.append(motif)
    return motifs


def motif_signature_key(signature: MotifSignature) -> str:
    node_count, edges = signature
    edge_text = ",".join(f"{source}>{target}" for source, target in edges)
    return f"n{node_count}|e{edge_text}"


def _sorted_signatures(
    signatures: Iterable[MotifSignature],
) -> list[MotifSignature]:
    return sorted(signatures, key=motif_signature_key)


def mobility_motif_metrics(
    locations_real: Sequence[Sequence[Any]] | Any,
    locations_generated: Sequence[Sequence[Any]] | Any,
    *,
    day_keys_real: Sequence[Sequence[Any]] | Any | None = None,
    day_keys_generated: Sequence[Sequence[Any]] | Any | None = None,
    frequency_threshold: float = 0.005,
    pad_id: Any = PAD_ID,
) -> dict[str, Any]:
    if not 0.0 <= frequency_threshold <= 1.0:
        raise ValueError("frequency_threshold must be in [0, 1]")
    if (day_keys_real is None) != (day_keys_generated is None):
        raise ValueError(
            "day_keys_real and day_keys_generated must be provided together"
        )

    real_motifs = daily_mobility_motifs(
        locations_real, day_keys_real, pad_id=pad_id
    )
    generated_motifs = daily_mobility_motifs(
        locations_generated, day_keys_generated, pad_id=pad_id
    )
    real_counts = Counter(real_motifs)
    generated_counts = Counter(generated_motifs)
    n_real = len(real_motifs)
    n_generated = len(generated_motifs)

    frequent = (
        {
            motif
            for motif, count in real_counts.items()
            if count / n_real > frequency_threshold
        }
        if n_real
        else set()
    )
    ordered = _sorted_signatures(frequent)
    real_hist = np.asarray(
        [real_counts[motif] for motif in ordered]
        + [
            sum(
                count
                for motif, count in real_counts.items()
                if motif not in frequent
            )
        ],
        dtype=np.float64,
    )
    generated_hist = np.asarray(
        [generated_counts[motif] for motif in ordered]
        + [
            sum(
                count
                for motif, count in generated_counts.items()
                if motif not in frequent
            )
        ],
        dtype=np.float64,
    )

    motif_jsd = jensen_shannon_divergence(real_hist, generated_hist)
    real_frequencies = {
        motif_signature_key(motif): real_counts[motif] / n_real
        for motif in ordered
    }
    generated_frequencies = {
        motif_signature_key(motif): (
            generated_counts[motif] / n_generated
            if n_generated
            else float("nan")
        )
        for motif in ordered
    }
    real_coverage = (
        float(real_hist[:-1].sum() / n_real) if n_real else float("nan")
    )
    generated_coverage = (
        float(generated_hist[:-1].sum() / n_generated)
        if n_generated
        else float("nan")
    )
    frequency_errors = [
        abs(real_frequencies[key] - generated_frequencies[key])
        for key in real_frequencies
    ]

    return {
        "mobility_motif_jsd": motif_jsd,
        "real_motif_frequencies": real_frequencies,
        "generated_motif_frequencies": generated_frequencies,
        "real_other_frequency": (
            float(real_hist[-1] / n_real) if n_real else float("nan")
        ),
        "generated_other_frequency": (
            float(generated_hist[-1] / n_generated)
            if n_generated
            else float("nan")
        ),
        "real_frequent_motif_coverage": real_coverage,
        "generated_frequent_motif_coverage": generated_coverage,
        "frequent_motif_mean_abs_frequency_error": (
            float(np.mean(frequency_errors))
            if frequency_errors
            else float("nan")
        ),
        "reference_frequent_motif_count": len(ordered),
        "real_day_count": n_real,
        "generated_day_count": n_generated,
        "frequency_threshold": frequency_threshold,
        "frequency_threshold_comparison": ">",
        "signature": "first_visit_order_directed_unweighted_graph_v1",
        "day_key_source": (
            "explicit"
            if day_keys_real is not None or day_keys_generated is not None
            else "trajectory_row"
        ),
    }


__all__ = [
    "MotifSignature",
    "daily_mobility_motifs",
    "mobility_motif_metrics",
    "motif_signature_key",
]
