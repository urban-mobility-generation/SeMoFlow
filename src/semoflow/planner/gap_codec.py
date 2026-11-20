from __future__ import annotations

from collections.abc import Sequence

import torch

PLANNER_GAP_CODEC_VERSION = "positive_duration_v1"

PLANNER_GAP_BUCKET_EDGES_MINUTES = (
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    45.0,
    90.0,
    180.0,
    360.0,
    720.0,
    1080.0,
    1440.0,
    2160.0,
    2880.0,
    4320.0,
)

PLANNER_GAP_REPRESENTATIVES_MINUTES = (
    0.5,
    1.5,
    3.5,
    7.5,
    15.0,
    32.5,
    67.5,
    135.0,
    270.0,
    540.0,
    900.0,
    1260.0,
    1800.0,
    2520.0,
    3600.0,
    5040.0,
)


def exact_gap_duration_minutes(
    event_time_minutes: torch.Tensor,
) -> torch.Tensor:
    times = torch.as_tensor(event_time_minutes, dtype=torch.float32)
    durations = torch.empty_like(times)
    durations[..., 0] = times[..., 0]
    if times.shape[-1] > 1:
        durations[..., 1:] = times[..., 1:] - times[..., :-1]
    return durations


def reconstruct_event_time_minutes(
    gap_tokens: torch.Tensor,
    *,
    representative_minutes: tuple[float, ...],
    observation_window_minutes: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    tokens = torch.as_tensor(gap_tokens, dtype=torch.long)
    representatives = torch.tensor(
        representative_minutes,
        dtype=torch.float32,
        device=tokens.device,
    )
    durations = representatives[tokens]
    if valid_mask is not None:
        mask = torch.as_tensor(
            valid_mask, dtype=torch.bool, device=tokens.device
        )
        durations = torch.where(mask, durations, 0.0)
    return torch.cumsum(durations, dim=-1).clamp(
        max=float(observation_window_minutes)
    )


def bucketize_gap_durations(
    durations: torch.Tensor, bucket_edges_minutes: Sequence[float]
) -> torch.Tensor:
    boundaries = torch.tensor(
        bucket_edges_minutes, dtype=durations.dtype, device=durations.device
    )
    return torch.bucketize(durations, boundaries, right=False).to(torch.long)


def gap_tokens_from_event_time_minutes(
    event_time_minutes: torch.Tensor,
    *,
    bucket_edges_minutes: tuple[float, ...],
) -> torch.Tensor:
    times = torch.as_tensor(event_time_minutes, dtype=torch.float32)
    return bucketize_gap_durations(
        exact_gap_duration_minutes(times), bucket_edges_minutes
    )


__all__ = [
    "PLANNER_GAP_BUCKET_EDGES_MINUTES",
    "PLANNER_GAP_CODEC_VERSION",
    "PLANNER_GAP_REPRESENTATIVES_MINUTES",
    "bucketize_gap_durations",
    "exact_gap_duration_minutes",
    "gap_tokens_from_event_time_minutes",
    "reconstruct_event_time_minutes",
]
