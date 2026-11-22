from __future__ import annotations

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


def reconstruct_event_time_minutes(
    gap_tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    representative_minutes: tuple[float, ...],
    observation_window_minutes: float,
) -> torch.Tensor:
    """Event times as cumulative representative gaps, capped at the window."""
    representatives = torch.tensor(
        representative_minutes, dtype=torch.float32, device=gap_tokens.device
    )
    durations = torch.where(valid_mask, representatives[gap_tokens], 0.0)
    return torch.cumsum(durations, dim=-1).clamp(
        max=float(observation_window_minutes)
    )


def gap_tokens_from_event_time_minutes(
    event_time_minutes: torch.Tensor,
) -> torch.Tensor:
    """Bucket each event's gap since the previous event (the first event's
    since the week start)."""
    durations = torch.diff(event_time_minutes, prepend=torch.zeros(1))
    edges = torch.tensor(
        PLANNER_GAP_BUCKET_EDGES_MINUTES, dtype=durations.dtype
    )
    return torch.bucketize(durations, edges, right=False).to(torch.long)


__all__ = [
    "PLANNER_GAP_BUCKET_EDGES_MINUTES",
    "PLANNER_GAP_CODEC_VERSION",
    "PLANNER_GAP_REPRESENTATIVES_MINUTES",
    "gap_tokens_from_event_time_minutes",
    "reconstruct_event_time_minutes",
]
