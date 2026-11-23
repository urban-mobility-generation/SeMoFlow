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

from semoflow.data import (
    N_PART_OF_DAY_BINS,
    bank_segments,
    bank_split_window,
    time_tokens_from_week_minutes,
)
from semoflow.evaluation.elapsed_recurrence import (
    ReturnEvents,
    elapsed_recurrence_jsd,
)
from semoflow.evaluation.grounding import (
    ground_sid_codes,
    grounded_poi_metrics,
)
from semoflow.evaluation.temporal_spatial import (
    events_from_place_ids,
    temporal_spatial_jsd,
)
from semoflow.poi.sids import (
    SemanticIDGroundingIndex,
)


def load_places(
    path: Path,
) -> tuple[dict[int, tuple[float, float]], dict[int, str], int]:
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
    offset, count = bank_split_window(payload, split)
    selected = min(count, max_samples)
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

    def events(poi, valid, hours) -> ReturnEvents:
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
    prefix: torch.Tensor
    time_tokens: torch.Tensor
    valid: torch.Tensor
    event_hours: torch.Tensor
    city_id: torch.Tensor


def plan_from_planner_samples(
    samples: Mapping[str, Any],
    *,
    width: int,
    max_rows: int,
) -> SampledPlan:
    prefix = samples["prefix_codes"][:max_rows].long()
    lengths = samples["lengths"][:max_rows].long()
    minutes = samples["event_time_minutes"][:max_rows].float()
    if bool((lengths > width).any()):
        raise ValueError(
            "planner samples contain a trajectory longer than the "
            f"realizer's {width} events"
        )
    padding = width - prefix.size(1)
    prefix = F.pad(prefix[:, :width], (0, 0, 0, max(padding, 0)))
    minutes = F.pad(minutes[:, :width], (0, max(padding, 0)))
    valid = torch.arange(width)[None, :] < lengths[:, None]
    return SampledPlan(
        prefix=prefix,
        time_tokens=time_tokens_from_week_minutes(
            minutes.clamp(0.0, 10079.999)
        ).masked_fill(~valid, -1),
        valid=valid,
        event_hours=(minutes.double() / 60.0).masked_fill(
            ~valid, float("nan")
        ),
        city_id=samples["city_ids"][:max_rows].long(),
    )


def load_candidate_prior(path: Path) -> dict[int, float]:
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
    reference_placekey_ids: torch.Tensor | None,
    candidate_selection: str,
    candidate_weights: Mapping[int, float] | None,
    compute_device: torch.device | str | None,
) -> dict[str, Any]:
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
    "load_candidate_prior",
    "load_places",
    "load_reference",
    "metric_block",
    "plan_from_planner_samples",
]
