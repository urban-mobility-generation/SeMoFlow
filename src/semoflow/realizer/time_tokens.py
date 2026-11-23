from __future__ import annotations

from datetime import datetime

import numpy as np

N_PART_OF_DAY_BINS = 6
N_TIME_TOKENS = 42
PART_OF_DAY_START_MINUTES: tuple[int, ...] = (0, 360, 600, 840, 1080, 1320)


def time_token(stamp: datetime) -> int:
    minute = stamp.hour * 60 + stamp.minute
    part = sum(minute >= start for start in PART_OF_DAY_START_MINUTES[1:])
    return stamp.weekday() * N_PART_OF_DAY_BINS + part


def time_tokens_from_timestamps(values: list[datetime]) -> np.ndarray:
    return np.asarray([time_token(value) for value in values], dtype=np.int64)
