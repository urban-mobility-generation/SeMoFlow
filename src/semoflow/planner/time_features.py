from __future__ import annotations

import math

import torch

_DAY_MINUTES = 1440.0
_WEEK_MINUTES = 10080.0
_DAILY_HARMONICS = 4
_WEEKLY_HARMONICS = 2

TIME_FEATURE_DIM = 2 * (_DAILY_HARMONICS + _WEEKLY_HARMONICS) + 2


def time_features(
    event_time_minutes: torch.Tensor,
    *,
    observation_window_minutes: float,
) -> torch.Tensor:
    if observation_window_minutes <= 0.0 or not math.isfinite(
        observation_window_minutes
    ):
        raise ValueError(
            "observation_window_minutes must be finite and positive"
        )
    minutes = event_time_minutes.float()
    day_phase = (
        2.0 * math.pi * torch.remainder(minutes, _DAY_MINUTES) / _DAY_MINUTES
    )
    week_phase = (
        2.0 * math.pi * torch.remainder(minutes, _WEEK_MINUTES) / _WEEK_MINUTES
    )
    features: list[torch.Tensor] = []
    for harmonic in range(1, _DAILY_HARMONICS + 1):
        features.append(torch.sin(harmonic * day_phase))
        features.append(torch.cos(harmonic * day_phase))
    for harmonic in range(1, _WEEKLY_HARMONICS + 1):
        features.append(torch.sin(harmonic * week_phase))
        features.append(torch.cos(harmonic * week_phase))
    elapsed_fraction = torch.clamp(
        minutes / observation_window_minutes, 0.0, 1.0
    )
    features.append(elapsed_fraction)
    features.append(1.0 - elapsed_fraction)
    return torch.stack(features, dim=-1)


__all__ = ["TIME_FEATURE_DIM", "time_features"]
