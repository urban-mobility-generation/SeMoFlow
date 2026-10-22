from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np

N_PART_OF_DAY_BINS = 6
N_TIME_TOKENS = 42


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
    hour = _coerce_datetime(value).hour
    if hour < 6:
        return 0
    if hour < 10:
        return 1
    if hour < 14:
        return 2
    if hour < 18:
        return 3
    if hour < 22:
        return 4
    return 5


def time_token(value: Any) -> int:
    stamp = _coerce_datetime(value)
    return stamp.weekday() * N_PART_OF_DAY_BINS + part_of_day_bin(stamp)


def time_tokens_from_timestamps(
    values: list[Any] | tuple[Any, ...],
) -> np.ndarray:
    return np.asarray([time_token(value) for value in values], dtype=np.int64)
