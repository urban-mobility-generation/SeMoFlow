from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np

N_PART_OF_DAY_BINS = 6
N_TIME_TOKENS = 42
# Start minute of each unequal circadian part-of-day bin:
# [0,6), [6,10), [10,14), [14,18), [18,22), [22,24) hours.
PART_OF_DAY_START_MINUTES: tuple[int, ...] = (0, 360, 600, 840, 1080, 1320)


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, np.datetime64):
        value = str(value)
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise TypeError(f"unsupported timestamp type: {type(value).__name__}")


def part_of_day_bin(value: Any) -> int:
    stamp = _coerce_datetime(value)
    minute = stamp.hour * 60 + stamp.minute
    return sum(minute >= start for start in PART_OF_DAY_START_MINUTES[1:])


def time_token(value: Any) -> int:
    stamp = _coerce_datetime(value)
    return stamp.weekday() * N_PART_OF_DAY_BINS + part_of_day_bin(stamp)


def time_tokens_from_timestamps(
    values: list[Any] | tuple[Any, ...],
) -> np.ndarray:
    return np.asarray([time_token(value) for value in values], dtype=np.int64)
