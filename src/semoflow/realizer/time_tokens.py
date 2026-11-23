from __future__ import annotations

import torch

N_PART_OF_DAY_BINS = 6
N_TIME_TOKENS = 42
PART_OF_DAY_START_MINUTES: tuple[int, ...] = (0, 360, 600, 840, 1080, 1320)


def time_tokens_from_week_minutes(minutes: torch.Tensor) -> torch.Tensor:
    part_of_day = torch.bucketize(
        minutes % 1440.0,
        torch.tensor(PART_OF_DAY_START_MINUTES[1:], dtype=minutes.dtype),
        right=True,
    )
    return (minutes / 1440.0).long() * N_PART_OF_DAY_BINS + part_of_day
