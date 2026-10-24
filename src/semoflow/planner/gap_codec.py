from __future__ import annotations

import math
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


def validate_gap_codec_contract(
    *,
    version: str | None,
    bucket_edges_minutes: tuple[float, ...] | None,
    representative_minutes: tuple[float, ...] | None,
) -> None:
    values = (version, bucket_edges_minutes, representative_minutes)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError(
            "gap_codec_version, gap_bucket_edges_minutes, and "
            "gap_duration_minutes must be set together"
        )
    assert version is not None
    assert bucket_edges_minutes is not None
    assert representative_minutes is not None
    if not version:
        raise ValueError("gap_codec_version must not be empty")
    if len(representative_minutes) < 1:
        raise ValueError("gap codec must contain at least one bucket")
    if len(bucket_edges_minutes) != len(representative_minutes) - 1:
        raise ValueError(
            "gap_bucket_edges_minutes must contain gap_vocab - 1 boundaries"
        )
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in representative_minutes
    ):
        raise ValueError(
            "gap_duration_minutes must be finite and strictly positive"
        )
    previous = 0.0
    for index, edge in enumerate(bucket_edges_minutes):
        if not math.isfinite(edge) or edge <= previous:
            raise ValueError(
                "gap_bucket_edges_minutes must be finite, positive, and "
                "strictly increasing"
            )
        representative = representative_minutes[index]
        above_lower = (
            representative >= previous
            if index == 0
            else representative > previous
        )
        if not above_lower or representative > edge:
            raise ValueError(
                "each finite gap bucket representative must lie inside its bucket"
            )
        previous = edge
    if representative_minutes[-1] <= previous:
        raise ValueError(
            "the final gap representative must be above the last edge"
        )


def exact_gap_duration_minutes(
    event_time_minutes: torch.Tensor,
) -> torch.Tensor:
    times = torch.as_tensor(event_time_minutes, dtype=torch.float32)
    if times.ndim < 1 or times.shape[-1] < 1:
        raise ValueError("event_time_minutes must have a non-empty last axis")
    durations = torch.empty_like(times)
    durations[..., 0] = times[..., 0]
    if times.shape[-1] > 1:
        durations[..., 1:] = times[..., 1:] - times[..., :-1]
    if bool((durations < 0).any()):
        raise ValueError("event_time_minutes must be non-decreasing")
    return durations


def reconstruct_event_time_minutes(
    gap_tokens: torch.Tensor,
    *,
    representative_minutes: tuple[float, ...],
    observation_window_minutes: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    tokens = torch.as_tensor(gap_tokens, dtype=torch.long)
    if tokens.ndim not in {1, 2} or tokens.shape[-1] < 1:
        raise ValueError("gap_tokens must have shape [T] or [B,T]")
    if observation_window_minutes <= 0.0 or not math.isfinite(
        observation_window_minutes
    ):
        raise ValueError(
            "observation_window_minutes must be finite and positive"
        )
    representatives = torch.tensor(
        representative_minutes,
        dtype=torch.float32,
        device=tokens.device,
    )
    if representatives.numel() < 1:
        raise ValueError("representative_minutes must not be empty")
    if bool((tokens < 0).any()) or bool(
        (tokens >= representatives.numel()).any()
    ):
        raise ValueError("gap token is outside the codec vocabulary")
    durations = representatives[tokens]
    if valid_mask is not None:
        mask = torch.as_tensor(
            valid_mask, dtype=torch.bool, device=tokens.device
        )
        if mask.shape != tokens.shape:
            raise ValueError("valid_mask must align with gap_tokens")
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
    if times.ndim != 1 or times.numel() < 1:
        raise ValueError("event_time_minutes must be a non-empty vector")
    if not bool(torch.isfinite(times).all()) or bool((times < 0).any()):
        raise ValueError("event_time_minutes must be finite and non-negative")
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
    "validate_gap_codec_contract",
]
