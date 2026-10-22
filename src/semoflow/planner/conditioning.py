from __future__ import annotations

import math
from datetime import date, datetime, time

import torch

from semoflow.utils.geohash import geohash_center

CONTEXT_MODE_NONE = "none"
CONTEXT_MODE_CITY = "city"
CONTEXT_MODE_CITY_CALENDAR_ORIGIN = "city_calendar_origin"
CONTEXT_SCHEMA_CITY_CALENDAR_ORIGIN_V1 = (
    "week_of_year_sin",
    "week_of_year_cos",
    "origin_latitude_scaled",
    "origin_longitude_scaled",
)
SUPPORTED_CONTEXT_MODES = {
    CONTEXT_MODE_NONE,
    CONTEXT_MODE_CITY,
    CONTEXT_MODE_CITY_CALENDAR_ORIGIN,
}


def context_schema_for_mode(mode: str) -> tuple[str, ...]:
    if mode == CONTEXT_MODE_NONE:
        return ()
    if mode == CONTEXT_MODE_CITY:
        return ()
    if mode == CONTEXT_MODE_CITY_CALENDAR_ORIGIN:
        return CONTEXT_SCHEMA_CITY_CALENDAR_ORIGIN_V1
    raise ValueError(
        "context_mode must be 'none', 'city', or 'city_calendar_origin'"
    )


def city_conditioning_for_mode(mode: str) -> bool:
    if mode not in SUPPORTED_CONTEXT_MODES:
        raise ValueError(
            "context_mode must be 'none', 'city', or 'city_calendar_origin'"
        )
    return mode != CONTEXT_MODE_NONE


def city_calendar_origin_features(
    *, week_start: date | datetime, origin_geohash: str
) -> torch.Tensor:
    if not isinstance(week_start, date):
        raise TypeError("week_start must be a date or datetime")
    if isinstance(week_start, datetime):
        if week_start.timetz().replace(tzinfo=None) != time():
            raise ValueError("week_start datetime must be at midnight")
        normalized_week_start = week_start
    else:
        normalized_week_start = datetime.combine(week_start, time())
    if normalized_week_start.weekday() != 0:
        raise ValueError("week_start must be a Monday")
    if not isinstance(origin_geohash, str) or len(origin_geohash) != 5:
        raise ValueError("origin_geohash must be a five-character geohash")
    year_start = normalized_week_start.replace(
        month=1, day=1, hour=0, minute=0, second=0, microsecond=0
    )
    day_of_year = (
        normalized_week_start - year_start
    ).total_seconds() / 86400.0
    annual_phase = 2.0 * math.pi * day_of_year / 365.2425
    latitude, longitude = geohash_center(origin_geohash)
    return torch.tensor(
        (
            math.sin(annual_phase),
            math.cos(annual_phase),
            latitude / 90.0,
            longitude / 180.0,
        ),
        dtype=torch.float32,
    )


__all__ = [
    "CONTEXT_MODE_CITY",
    "CONTEXT_MODE_CITY_CALENDAR_ORIGIN",
    "CONTEXT_MODE_NONE",
    "CONTEXT_SCHEMA_CITY_CALENDAR_ORIGIN_V1",
    "SUPPORTED_CONTEXT_MODES",
    "city_calendar_origin_features",
    "city_conditioning_for_mode",
    "context_schema_for_mode",
]
