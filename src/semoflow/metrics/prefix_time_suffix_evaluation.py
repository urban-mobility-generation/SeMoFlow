from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from semoflow.metrics.trajectory.elapsed_recurrence import (
    ReturnEvents,
    elapsed_recurrence_jsd,
)
from semoflow.metrics.trajectory.grounding import (
    ground_sid_codes,
    grounded_poi_metrics,
)
from semoflow.metrics.trajectory.temporal_spatial import (
    events_from_place_ids,
    temporal_spatial_jsd,
)
from semoflow.realizer.bank import bank_segments, bank_split_window
from semoflow.realizer.time_tokens import (
    N_PART_OF_DAY_BINS,
    PART_OF_DAY_START_MINUTES,
)
from semoflow.tokenization.decoding.grounding_index import (
    SemanticIDGroundingIndex,
)


def load_places(
    path: Path,
) -> tuple[dict[int, tuple[float, float]], dict[int, str], int]:
    """In-range POI coordinates, geohashes and the placekey vocab size."""
    frame = pl.read_parquet(
        path, columns=["placekey_id", "latitude", "longitude", "geohash_5"]
    )
    located = frame.filter(
        pl.col("latitude").is_between(-90.0, 90.0)
        & pl.col("longitude").is_between(-180.0, 180.0)
    )
    coordinates = {
        int(placekey_id): (float(latitude), float(longitude))
        for placekey_id, latitude, longitude, _ in located.iter_rows()
    }
    geohashes = {
        int(placekey_id): str(geohash)
        for placekey_id, *_, geohash in frame.iter_rows()
        if geohash is not None
    }
    vocab_size = int(frame["placekey_id"].to_numpy().max(initial=0)) + 1
    return coordinates, geohashes, vocab_size


def load_reference(
    payload: dict[str, Any], *, split: str, max_samples: int
) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, torch.Tensor]:
    """The leading ``max_samples`` bank rows of a split, with their POIs."""
    offset, count = bank_split_window(payload, split)
    selected = min(count, max_samples)
    if selected == 0:
        raise ValueError("reference selection is empty")
    rows = slice(offset, offset + selected)
    valid_mask = payload["valid_mask"][rows].bool()
    frame = bank_segments(payload, split, ["placekey_id"], rows=selected)
    placekey_ids = torch.zeros_like(valid_mask, dtype=torch.long)
    flat = frame["placekey_id"].explode().to_numpy().astype(np.int64)
    placekey_ids[valid_mask] = torch.from_numpy(flat)
    return (
        payload["sid_codes"][rows].long(),
        payload["time_tokens"][rows].long(),
        valid_mask,
        placekey_ids,
    )


def aligned_reference_event_hours(
    payload: dict[str, Any], *, split: str, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Observed event times in nominal local hours from each row's week start.

    Segments are naive local timestamps grouped into Monday-truncated weeks,
    so each row's origin is Monday 00:00 of its first event's week and its
    times lie in [0, 168) h. Rows are the first ``valid_mask.size(0)`` rows of
    the split. Inactive positions are NaN.
    """
    frame = bank_segments(
        payload, split, ["_timestamps"], rows=int(valid_mask.size(0))
    )
    exploded = frame.with_columns(
        pl.col("_timestamps").list.first().dt.truncate("1w").alias("week")
    ).explode("_timestamps")
    hours = (
        exploded["_timestamps"] - exploded["week"]
    ).dt.total_microseconds().to_numpy() / 3.6e9
    out = torch.full(valid_mask.shape, float("nan"), dtype=torch.float64)
    out[valid_mask.bool()] = torch.from_numpy(hours).double()
    return out


def elapsed_recurrence_block(
    real_poi: torch.Tensor,
    real_valid: torch.Tensor,
    real_hours: torch.Tensor,
    generated_poi: torch.Tensor,
    generated_valid: torch.Tensor,
    generated_hours: torch.Tensor,
) -> dict[str, Any]:
    """Elapsed-time recurrence on grounded POIs.

    Events that ground to no POI (ID 0) are dropped, as the index
    recurrence metric drops them.
    """

    def events(poi, valid, hours) -> ReturnEvents:
        if hours.shape != valid.shape:
            raise ValueError("event hours do not align with the valid mask")
        poi = poi.cpu().numpy()
        return ReturnEvents(
            place_ids=poi,
            hours=hours.cpu().double().numpy(),
            valid=valid.cpu().numpy().astype(bool) & (poi != 0),
        )

    return elapsed_recurrence_jsd(
        events(real_poi, real_valid, real_hours),
        events(generated_poi, generated_valid, generated_hours),
    )


@dataclass(frozen=True)
class SampledPlan:
    """Planner samples laid out on the realizer's event grid."""

    prefix: torch.Tensor
    time_tokens: torch.Tensor
    valid: torch.Tensor
    event_hours: torch.Tensor
    city_id: torch.Tensor | None


def plan_from_planner_samples(
    samples: Mapping[str, Any],
    *,
    width: int,
    prefix_depth: int,
    max_rows: int,
) -> SampledPlan:
    """The leading ``max_rows`` planner samples as a realizer plan.

    Continuous arrival minutes become the realizer's weekly time tokens
    through its unequal circadian bins, and also give the continuous clock
    (in hours) for elapsed-time recurrence. Off-plan positions get time
    token ``-1`` and hour NaN.
    """
    prefix = samples["prefix_codes"][:max_rows].long()
    lengths = samples["lengths"][:max_rows].long()
    minutes = samples["event_time_minutes"][:max_rows].float()
    if prefix.size(-1) != prefix_depth:
        raise ValueError(
            f"planner samples have prefix depth {prefix.size(-1)}; the "
            f"realizer expects {prefix_depth}"
        )
    if bool((lengths > width).any()):
        raise ValueError(
            "planner samples contain a trajectory longer than the "
            f"realizer's {width} events"
        )
    padding = width - prefix.size(1)
    prefix = F.pad(prefix[:, :width], (0, 0, 0, max(padding, 0)))
    minutes = F.pad(minutes[:, :width], (0, max(padding, 0)))
    valid = torch.arange(width)[None, :] < lengths[:, None]
    clamped = minutes.clamp(0.0, 10079.999)
    part_of_day = torch.bucketize(
        clamped % 1440.0,
        torch.tensor(PART_OF_DAY_START_MINUTES[1:], dtype=clamped.dtype),
        right=True,
    )
    day = (clamped / 1440.0).long()
    city = samples.get("city_ids")
    return SampledPlan(
        prefix=prefix,
        time_tokens=(day * N_PART_OF_DAY_BINS + part_of_day).masked_fill(
            ~valid, -1
        ),
        valid=valid,
        event_hours=(minutes.double() / 60.0).masked_fill(
            ~valid, float("nan")
        ),
        city_id=None if city is None else city[:max_rows].long(),
    )


def load_candidate_prior(path: Path) -> dict[int, float]:
    """Per-POI train visit counts written by ``build_prior``."""
    counts = json.loads(Path(path).read_text())["counts"]
    return {int(key): float(value) for key, value in counts.items()}


def metric_block(
    real_codes: torch.Tensor,
    real_valid: torch.BoolTensor,
    generated_codes: torch.Tensor,
    generated_valid: torch.BoolTensor,
    *,
    grounding: SemanticIDGroundingIndex,
    coordinates: dict[int, tuple[float, float]],
    geohashes: dict[int, str],
    placekey_vocab_size: int,
    real_time_tokens: torch.Tensor,
    generated_time_tokens: torch.Tensor,
    real_event_hours: torch.Tensor,
    generated_event_hours: torch.Tensor,
    reference_placekey_ids: torch.Tensor | None = None,
    candidate_selection: str = "first",
    candidate_weights: Mapping[int, float] | None = None,
    compute_device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Ground both arms' SIDs to POIs and compare them.

    ``real_time_tokens``/``generated_time_tokens`` are the 42 weekly time
    tokens aligned with the codes. They give the motif day keys and the joint
    temporal-spatial JSDs. ``real_event_hours``/``generated_event_hours`` are
    continuous event times in hours for the elapsed-time recurrence JSDs;
    times are never derived from tokens.
    """
    real_poi = ground_sid_codes(
        real_codes,
        real_valid,
        grounding,
        preferred_placekey_ids=reference_placekey_ids,
    )
    generated_poi = ground_sid_codes(
        generated_codes,
        generated_valid,
        grounding,
        candidate_selection=candidate_selection,
        candidate_weights=candidate_weights,
    )
    metrics: dict[str, Any] = grounded_poi_metrics(
        real_poi,
        generated_poi,
        placekey_vocab_size=placekey_vocab_size,
        coordinates=coordinates,
        geohashes=geohashes,
        day_keys_real=real_time_tokens // N_PART_OF_DAY_BINS,
        day_keys_generated=generated_time_tokens // N_PART_OF_DAY_BINS,
        compute_device=compute_device,
    )
    metrics["temporal_spatial"] = temporal_spatial_jsd(
        *(
            events_from_place_ids(
                poi.cpu().numpy(),
                times.cpu().numpy(),
                valid.cpu().numpy(),
                coordinates=coordinates,
                geohashes=geohashes,
            )
            for poi, times, valid in (
                (real_poi, real_time_tokens, real_valid),
                (generated_poi, generated_time_tokens, generated_valid),
            )
        )
    )
    metrics["elapsed_recurrence"] = elapsed_recurrence_block(
        real_poi,
        real_valid,
        real_event_hours,
        generated_poi,
        generated_valid,
        generated_event_hours,
    )
    return metrics


__all__ = [
    "SampledPlan",
    "aligned_reference_event_hours",
    "elapsed_recurrence_block",
    "load_candidate_prior",
    "load_places",
    "load_reference",
    "metric_block",
    "plan_from_planner_samples",
]
