from __future__ import annotations

import random
from collections.abc import Iterator
from functools import partial
from typing import Any

import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    Sampler,
    SequentialSampler,
)

from semoflow.planner.artifacts import OccurrencePlannerSplit
from semoflow.planner.config import OccurrencePlannerConfig
from semoflow.planner.gap_codec import reconstruct_event_time_minutes
from semoflow.planner.legality import LegalPrefixIndex
from semoflow.planner.types import OccurrenceBatch


def _shift_right(
    values: torch.Tensor, valid_mask: torch.Tensor, fill: float
) -> torch.Tensor:
    """Each step's previous valid value, or ``fill`` at the first step."""
    shifted = torch.full_like(values, fill)
    previous_valid = valid_mask[:, 1:]
    if values.ndim == 3:
        previous_valid = previous_valid[..., None]
    shifted[:, 1:] = torch.where(previous_valid, values[:, :-1], fill)
    return shifted


def collate_occurrence_tensors(
    tensors: tuple[torch.Tensor, ...],
    *,
    config: OccurrencePlannerConfig,
    legality: LegalPrefixIndex,
) -> OccurrenceBatch:
    """Teacher-forcing inputs and targets of one batch of split rows."""
    prefix_codes, gap_tokens, lengths, city_ids = tensors
    batch_size = prefix_codes.shape[0]
    device = prefix_codes.device
    max_len = int(lengths.max().item())

    sub_prefix = prefix_codes[:, :max_len]
    sub_gaps = gap_tokens[:, :max_len]
    valid_mask = (
        torch.arange(max_len, device=device)[None, :] < lengths[:, None]
    )
    default_prefix = (
        legality.codes[0]
        .to(device)
        .view(1, 1, -1)
        .expand(batch_size, max_len, -1)
    )
    target_prefixes = torch.where(
        valid_mask[..., None], sub_prefix, default_prefix
    )
    target_gaps = torch.where(valid_mask, sub_gaps, 0)
    event_times = reconstruct_event_time_minutes(
        target_gaps,
        valid_mask,
        representative_minutes=config.gap_duration_minutes,
        observation_window_minutes=config.observation_window_minutes,
    )
    return OccurrenceBatch(
        previous_prefix_codes=_shift_right(sub_prefix, valid_mask, -1),
        previous_gap_tokens=_shift_right(sub_gaps, valid_mask, -1),
        previous_event_time_minutes=_shift_right(event_times, valid_mask, 0.0),
        target_prefix_codes=target_prefixes,
        target_gap_tokens=target_gaps,
        valid_mask=valid_mask,
        lengths=lengths,
        city_ids=city_ids,
    )


class OccurrencePlannerTensorDataset(Dataset[Any]):
    """A planner split; indexing with a batch of indices slices every
    tensor at once."""

    def __init__(self, split: OccurrencePlannerSplit) -> None:
        self.tensors = (
            split.prefix_codes,
            split.gap_tokens,
            split.lengths,
            split.city_ids,
        )

    def __len__(self) -> int:
        return int(self.tensors[2].shape[0])

    def __getitem__(self, indices: Any) -> tuple[torch.Tensor, ...]:
        return tuple(tensor[indices] for tensor in self.tensors)


class _SeededShuffleSampler(Sampler[int]):
    def __init__(self, size: int, *, seed: int | None) -> None:
        self._size = int(size)
        self._random = random.Random(seed)

    def __iter__(self) -> Iterator[int]:
        order = list(range(self._size))
        self._random.shuffle(order)
        return iter(order)

    def __len__(self) -> int:
        return self._size


def build_occurrence_tensor_dataloader(
    dataset: OccurrencePlannerTensorDataset,
    *,
    config: OccurrencePlannerConfig,
    legality: LegalPrefixIndex,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int | None = None,
    pin_memory: bool = True,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    if sampler is None:
        sampler = (
            _SeededShuffleSampler(len(dataset), seed=seed)
            if shuffle
            else SequentialSampler(dataset)
        )
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=False),
        collate_fn=partial(
            collate_occurrence_tensors, config=config, legality=legality
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
