from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class OccurrenceBatch:
    previous_prefix_codes: torch.Tensor
    previous_gap_tokens: torch.Tensor
    previous_event_time_minutes: torch.Tensor
    target_prefix_codes: torch.Tensor
    target_gap_tokens: torch.Tensor
    valid_mask: torch.Tensor
    lengths: torch.Tensor
    city_ids: torch.Tensor

    def _map_tensors(
        self, transform: Callable[[torch.Tensor], torch.Tensor]
    ) -> OccurrenceBatch:
        return OccurrenceBatch(
            **{
                field.name: transform(getattr(self, field.name))
                for field in dataclasses.fields(self)
            }
        )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> OccurrenceBatch:
        return self._map_tensors(
            lambda tensor: tensor.to(device, non_blocking=non_blocking)
        )

    def pin_memory(self) -> OccurrenceBatch:
        return self._map_tensors(torch.Tensor.pin_memory)


@dataclass(frozen=True, slots=True)
class OccurrencePlannerOutput:
    length_logits: torch.Tensor
    gap_logits: torch.Tensor
    prefix_log_probability: torch.Tensor


@dataclass(frozen=True, slots=True)
class OccurrenceGenerationCache:
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class OccurrenceStepSample:
    gap_tokens: torch.Tensor
    prefix_codes: torch.Tensor
    arrival_time_minutes: torch.Tensor


@dataclass(frozen=True, slots=True)
class OccurrencePlannerLoss:
    total: torch.Tensor
    length: torch.Tensor
    gap: torch.Tensor
    prefix: torch.Tensor
