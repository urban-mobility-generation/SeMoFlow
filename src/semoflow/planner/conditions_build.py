from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl
import torch

from semoflow.planner.artifacts import (
    CONDITIONS_ARTIFACT_TYPE,
    load_occurrence_planner_checkpoint,
    save_torch_atomic,
)
from semoflow.planner.conditioning import (
    CONTEXT_SCHEMA_CITY_CALENDAR_ORIGIN_V1,
    city_calendar_origin_features,
)


@dataclass(frozen=True, slots=True)
class OccurrencePlannerConditionsBuildResult:
    output_path: Path
    num_samples: int
    lengths_conditioned: bool

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _require_week_start(value: object) -> date | datetime:
    if not isinstance(value, date):
        raise TypeError("week_start values must be dates or datetimes")
    return value


def build_occurrence_planner_conditions(
    *,
    checkpoint_path: Path,
    rows: pl.DataFrame,
    output_path: Path,
    force: bool = False,
) -> OccurrencePlannerConditionsBuildResult:
    """Save one condition per row of ``rows`` (city_id, optional length)."""
    checkpoint = load_occurrence_planner_checkpoint(checkpoint_path)
    city_only = (
        checkpoint.config.city_conditioning
        and checkpoint.config.context_schema == ()
    )
    city_calendar_origin = (
        checkpoint.config.city_conditioning
        and checkpoint.config.context_schema
        == CONTEXT_SCHEMA_CITY_CALENDAR_ORIGIN_V1
    )
    if not city_only and not city_calendar_origin:
        raise ValueError("checkpoint does not use supported city conditioning")

    schema = rows.schema
    required = {"city_id"}
    if city_calendar_origin:
        required.update(("week_start", "origin_geohash_5"))
    missing = sorted(required.difference(schema))
    if missing:
        raise ValueError(
            "conditions input is missing required columns: "
            + ", ".join(missing)
        )
    columns = ["city_id"]
    if city_calendar_origin:
        columns.extend(("week_start", "origin_geohash_5"))
    has_lengths = "length" in schema
    if has_lengths:
        columns.append("length")
    frame = rows.select(columns)
    if frame.height < 1:
        raise ValueError("conditions input must contain at least one row")

    city_ids = torch.empty(frame.height, dtype=torch.long)
    context_features = (
        torch.empty(
            (frame.height, len(CONTEXT_SCHEMA_CITY_CALENDAR_ORIGIN_V1)),
            dtype=torch.float32,
        )
        if city_calendar_origin
        else None
    )
    length_values: list[int] = []
    for row_index, row in enumerate(frame.iter_rows(named=True)):
        raw_city_id = row["city_id"]
        if not isinstance(raw_city_id, int) or isinstance(raw_city_id, bool):
            raise TypeError("city_id values must be integers")
        city_ids[row_index] = raw_city_id
        if context_features is not None:
            raw_geohash = row["origin_geohash_5"]
            if not isinstance(raw_geohash, str):
                raise TypeError("origin_geohash_5 values must be strings")
            context_features[row_index] = city_calendar_origin_features(
                week_start=_require_week_start(row["week_start"]),
                origin_geohash=raw_geohash,
            )
        if has_lengths:
            raw_length = row["length"]
            if not isinstance(raw_length, int) or isinstance(raw_length, bool):
                raise TypeError("length values must be integers")
            length_values.append(raw_length)

    if not bool(torch.isin(city_ids, checkpoint.catalog_city_ids).all()):
        raise ValueError(
            "conditions input contains a city absent from checkpoint"
        )
    lengths = (
        torch.tensor(length_values, dtype=torch.long) if has_lengths else None
    )
    if lengths is not None and (
        bool((lengths < 1).any())
        or bool((lengths > checkpoint.config.max_occurrences).any())
    ):
        raise ValueError("conditioned length is outside checkpoint bounds")

    save_torch_atomic(
        output_path,
        {
            "artifact_type": CONDITIONS_ARTIFACT_TYPE,
            "lengths": lengths,
            "context_features": context_features,
            "context_schema": checkpoint.config.context_schema,
            "city_ids": city_ids,
        },
        force=force,
    )
    return OccurrencePlannerConditionsBuildResult(
        output_path=Path(output_path),
        num_samples=frame.height,
        lengths_conditioned=has_lengths,
    )


__all__ = [
    "OccurrencePlannerConditionsBuildResult",
    "build_occurrence_planner_conditions",
]
