from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from semoflow.metrics.trajectory.elapsed_recurrence import (
    ReturnEvents,
    elapsed_recurrence_jsd,
)
from semoflow.metrics.trajectory.grounding import (
    ground_sid_tokens_to_poi_tokens,
    grounded_poi_metrics,
)
from semoflow.metrics.trajectory.temporal_spatial import (
    events_from_place_ids,
    temporal_spatial_jsd,
)
from semoflow.realizer.bank import (
    load_bank_payload,
    resolve_bank_split_window,
)
from semoflow.realizer.time_tokens import (
    N_PART_OF_DAY_BINS,
    PART_OF_DAY_START_MINUTES,
)
from semoflow.tokenization.decoding.grounding_index import (
    SemanticIDGroundingIndex,
)
from semoflow.tokenization.token_space import (
    apply_trajectory_valid_mask,
    sid_code_columns,
    to_token_fields,
)


def load_lookup(path: Path) -> np.ndarray:
    columns = list(
        sid_code_columns(pl.scan_parquet(path).collect_schema().names())
    )
    if not columns:
        raise ValueError(f"lookup carries no sem_code_* columns: {path}")
    frame = pl.read_parquet(path, columns=columns)
    return np.stack(
        [frame[name].to_numpy() for name in columns],
        axis=1,
    ).astype(np.int64)


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
    bank_path: Path,
    *,
    max_samples: int,
    split: str = "val",
    payload: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, torch.Tensor]:
    """The leading ``max_samples`` bank rows of a split, with their POIs."""
    if payload is None:
        payload = load_bank_payload(bank_path)
    if split not in {"val", "test"}:
        raise ValueError("reference split must be 'val' or 'test'")
    offset, count = resolve_bank_split_window(payload, split, path=bank_path)
    selected_count = min(int(count), max_samples)
    if selected_count == 0:
        raise ValueError("reference selection is empty")
    rows = slice(offset, offset + selected_count)
    sid_codes = payload["sid_codes"][rows].long()
    time_tokens = payload["time_tokens"][rows].long()
    valid_mask = payload["valid_mask"][rows].bool()

    source = payload["source"]
    segments_value = source.get("segments_path")
    segments_path = Path(segments_value)
    skipped = source.get("skipped_missing_sid", {})
    if isinstance(skipped, dict) and int(skipped.get(split, 0)):
        raise ValueError(
            "cannot reconstruct aligned reference POIs when the bank "
            f"skipped {split} segments with missing SIDs"
        )
    max_events = int(source.get("max_events", sid_codes.size(1)))
    min_length = int(source.get("min_length", 1))
    frame = (
        pl.scan_parquet(segments_path)
        .filter(pl.col("_split") == split)
        .filter(
            pl.col("placekey_id").list.len().is_between(min_length, max_events)
        )
        .head(selected_count)
        .collect()
    )
    if frame.height != selected_count:
        raise ValueError(
            f"reference segment metadata has {frame.height} rows; "
            f"expected {selected_count}"
        )
    placekey_ids = torch.zeros_like(valid_mask, dtype=torch.long)
    for row_index, row in enumerate(frame.iter_rows(named=True)):
        ids = [int(value) for value in row["placekey_id"]]
        length = int(valid_mask[row_index].sum().item())
        if len(ids) != length:
            raise ValueError(
                "reference segment and bank valid lengths do not align at "
                f"row {row_index}: {len(ids)} != {length}"
            )
        placekey_ids[row_index, :length] = torch.tensor(ids, dtype=torch.long)
    return sid_codes, time_tokens, valid_mask, placekey_ids


def aligned_reference_event_hours(
    payload: dict[str, Any],
    *,
    split: str,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Observed event times in nominal local hours from each row's week start.

    Segments are naive local timestamps grouped into Monday-truncated weeks,
    so each row's origin is Monday 00:00 of its first event's week and its
    times lie in [0, 168) h. Rows are the first ``valid_mask.size(0)`` rows of
    the split, selected exactly as :func:`load_reference` selects them.
    Inactive positions are NaN.
    """
    source = payload["source"]
    max_events = int(source.get("max_events", valid_mask.size(1)))
    min_length = int(source.get("min_length", 1))
    frame = cast(
        pl.DataFrame,
        pl.scan_parquet(Path(source["segments_path"]))
        .filter(pl.col("_split") == split)
        .filter(
            pl.col("placekey_id").list.len().is_between(min_length, max_events)
        )
        .head(int(valid_mask.size(0)))
        .select("_timestamps")
        .collect(),
    )
    lengths = torch.from_numpy(
        frame["_timestamps"].list.len().to_numpy().astype(np.int64)
    )
    if not torch.equal(lengths, valid_mask.sum(dim=1).long()):
        raise ValueError("reference timestamps do not align with bank rows")
    exploded = frame.with_columns(
        pl.col("_timestamps").list.first().dt.truncate("1w").alias("week")
    ).explode("_timestamps")
    hours = (
        exploded["_timestamps"] - exploded["week"]
    ).dt.total_microseconds().to_numpy() / 3.6e9
    out = torch.full(valid_mask.shape, float("nan"), dtype=torch.float64)
    rows = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    starts = torch.cumsum(lengths, 0) - lengths
    columns = torch.arange(int(lengths.sum())) - torch.repeat_interleave(
        starts, lengths
    )
    out[rows, columns] = torch.from_numpy(hours).double()
    return out


def elapsed_recurrence_block(
    real_poi: torch.Tensor,
    real_valid: torch.Tensor,
    real_hours: torch.Tensor | None,
    generated_poi: torch.Tensor,
    generated_valid: torch.Tensor,
    generated_hours: torch.Tensor | None,
) -> dict[str, Any]:
    """Elapsed-time recurrence on grounded POIs, or why it is unavailable.

    Events that ground to no POI (ID 0) are dropped, as the index
    recurrence metric drops them.
    """
    missing = [
        side
        for side, hours in (
            ("real", real_hours),
            ("generated", generated_hours),
        )
        if hours is None
    ]
    if missing:
        return {"unavailable": f"no continuous clock: {', '.join(missing)}"}
    assert real_hours is not None and generated_hours is not None

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


CANDIDATE_PRIOR_FORMAT = "grounding_candidate_prior_v1"


def load_candidate_prior(path: Path) -> dict[int, float]:
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != CANDIDATE_PRIOR_FORMAT:
        raise ValueError(
            f"{path} is not a {CANDIDATE_PRIOR_FORMAT} artifact; "
            f"got {payload.get('format')!r}"
        )
    if payload.get("split") != "train":
        raise ValueError(
            f"candidate prior must be fitted on the train split; "
            f"{path} declares {payload.get('split')!r}"
        )
    counts = payload.get("counts")
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
    real_event_hours: torch.Tensor | None,
    generated_event_hours: torch.Tensor | None,
    reference_placekey_ids: torch.Tensor | None = None,
    candidate_selection: str = "first",
    candidate_weights: Mapping[int, float] | None = None,
    compute_device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Ground both arms' SIDs to POIs and compare them.

    ``real_time_tokens``/``generated_time_tokens`` are the 42 weekly time
    tokens aligned with the codes. They give the motif day keys and the joint
    temporal-spatial JSDs. ``real_event_hours``/``generated_event_hours`` are
    continuous event times in hours; when both are given the elapsed-time
    recurrence JSDs are scored, otherwise that section records why it is
    unavailable. Times are never derived from tokens.
    """
    fields = [f"sem_code_{level}" for level in range(real_codes.size(-1))]
    real_poi = ground_sid_tokens_to_poi_tokens(
        apply_trajectory_valid_mask(
            to_token_fields(real_codes, fields), real_valid
        ),
        grounding,
        preferred_placekey_ids=reference_placekey_ids,
    )
    generated_poi = ground_sid_tokens_to_poi_tokens(
        apply_trajectory_valid_mask(
            to_token_fields(generated_codes, fields), generated_valid
        ),
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
    "load_lookup",
    "load_places",
    "load_reference",
    "metric_block",
    "plan_from_planner_samples",
]
