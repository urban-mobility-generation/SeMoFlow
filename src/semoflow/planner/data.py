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

from semoflow.planner.config import OccurrencePlannerConfig
from semoflow.planner.gap_codec import (
    bucketize_gap_durations,
    reconstruct_event_time_minutes,
)
from semoflow.planner.legality import LegalPrefixIndex
from semoflow.planner.types import (
    OccurrenceBatch,
)


def _check_collation_inputs(
    prefix_codes: torch.Tensor,
    gap_tokens: torch.Tensor,
    lengths: torch.Tensor,
    *,
    config: OccurrencePlannerConfig,
    legality: LegalPrefixIndex,
    event_time_minutes: torch.Tensor | None,
    context_features: torch.Tensor | None,
    city_ids: torch.Tensor | None,
) -> None:
    batch_size = prefix_codes.shape[0]
    if batch_size < 1:
        raise ValueError("batch must not be empty")
    if legality.codebook_size != config.codebook_size:
        raise ValueError("legality and config codebook sizes differ")
    if legality.prefix_depth != config.prefix_depth:
        raise ValueError("legality and config prefix depths differ")
    if (
        legality.codes.device.type != "cpu"
        and prefix_codes.device.type == "cpu"
    ):
        raise ValueError(
            "collation requires a separate CPU legality index; do not pass "
            "the model-device legality index"
        )
    if city_ids is not None:
        if city_ids.dtype != torch.long:
            raise ValueError("city_ids must use torch.long")
        if city_ids.shape != (batch_size,):
            raise ValueError("city_ids must have shape [batch]")
        if city_ids.device != prefix_codes.device:
            raise ValueError(
                "city_ids must share the occurrence tensor device"
            )
    if context_features is not None and not config.context_dim:
        raise ValueError("context supplied to a context-free planner")
    if context_features is not None:
        if context_features.shape != (batch_size, config.context_dim):
            raise ValueError("context_features have the wrong shape")
    elif config.context_dim:
        raise ValueError("context_features are required by the config")

    if config.temporal_state_enabled and event_time_minutes is None:
        raise ValueError(
            "event_time_minutes is required by a planner with a time channel"
        )
    if not config.temporal_state_enabled and event_time_minutes is not None:
        raise ValueError(
            "event_time_minutes supplied to a planner without a time channel"
        )

    if prefix_codes.ndim != 3:
        raise ValueError("prefix_codes must have shape [B, S, P]")
    if gap_tokens.ndim != 2 or lengths.ndim != 1:
        raise ValueError("gap_tokens/lengths have invalid rank")
    rows, width, depth = prefix_codes.shape
    if rows != batch_size or depth != config.prefix_depth:
        raise ValueError("prefix_codes shape disagrees with config")
    if gap_tokens.shape != (batch_size, width):
        raise ValueError("gap_tokens do not align with prefix_codes")
    if lengths.shape != (batch_size,):
        raise ValueError("lengths must have shape [batch]")
    if event_time_minutes is not None:
        if event_time_minutes.shape != (batch_size, width):
            raise ValueError(
                "event_time_minutes do not align with prefix_codes"
            )
        if event_time_minutes.device != prefix_codes.device:
            raise ValueError(
                "event_time_minutes must share the occurrence tensor device"
            )
        if not event_time_minutes.is_floating_point():
            raise ValueError("event_time_minutes must use a floating dtype")
    if (
        context_features is not None
        and context_features.device != prefix_codes.device
    ):
        raise ValueError(
            "context_features must share the occurrence tensor device"
        )
    if (
        prefix_codes.dtype != torch.long
        or gap_tokens.dtype != torch.long
        or lengths.dtype != torch.long
    ):
        raise ValueError("occurrence tensors must use torch.long")
    if (
        gap_tokens.device != prefix_codes.device
        or lengths.device != prefix_codes.device
    ):
        raise ValueError("occurrence tensors must share one device")
    if bool((lengths < 1).any()) or bool((lengths > width).any()):
        raise ValueError("lengths are outside the padded tensor width")


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


def _event_time_channels(
    event_time_minutes: torch.Tensor,
    target_gaps: torch.Tensor,
    valid_mask: torch.Tensor,
    config: OccurrencePlannerConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check raw event times and return (previous, target) codec times."""
    window = config.observation_window_minutes
    edges = config.gap_bucket_edges_minutes
    representatives = config.gap_duration_minutes
    assert window is not None and edges is not None
    assert representatives is not None
    raw_times = event_time_minutes[:, : valid_mask.shape[1]].float()
    raw_durations = raw_times.clone()
    raw_durations[:, 1:] -= raw_times[:, :-1]
    if bool((raw_durations[valid_mask] < 0).any()):
        raise ValueError("event_time_minutes must be non-decreasing")
    expected_gaps = bucketize_gap_durations(raw_durations, edges)
    if not torch.equal(target_gaps[valid_mask], expected_gaps[valid_mask]):
        raise ValueError("gap_tokens disagree with raw event_time_minutes")
    reconstructed = reconstruct_event_time_minutes(
        target_gaps,
        representative_minutes=representatives,
        observation_window_minutes=window,
        valid_mask=valid_mask,
    )
    return (
        _shift_right(reconstructed, valid_mask, 0.0),
        torch.where(valid_mask, reconstructed, 0.0),
    )


def collate_occurrence_tensors(
    prefix_codes: torch.Tensor,
    gap_tokens: torch.Tensor,
    lengths: torch.Tensor,
    *,
    config: OccurrencePlannerConfig,
    legality: LegalPrefixIndex,
    event_time_minutes: torch.Tensor | None = None,
    context_features: torch.Tensor | None = None,
    city_ids: torch.Tensor | None = None,
) -> OccurrenceBatch:
    if city_ids is not None:
        city_ids = torch.as_tensor(city_ids)
    if context_features is not None:
        context_features = torch.as_tensor(
            context_features, dtype=torch.float32
        )
    _check_collation_inputs(
        prefix_codes,
        gap_tokens,
        lengths,
        config=config,
        legality=legality,
        event_time_minutes=event_time_minutes,
        context_features=context_features,
        city_ids=city_ids,
    )
    batch_size = prefix_codes.shape[0]
    device = prefix_codes.device
    max_len = int(lengths.max().item())
    if max_len > config.max_occurrences:
        raise ValueError("sequence length exceeds max_occurrences")

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

    previous_times: torch.Tensor | None = None
    target_times: torch.Tensor | None = None
    if event_time_minutes is not None:
        previous_times, target_times = _event_time_channels(
            event_time_minutes, target_gaps, valid_mask, config
        )

    valid_prefixes = target_prefixes[valid_mask]
    legality_device = legality.codes.device
    if valid_prefixes.numel() and not bool(
        legality.contains(valid_prefixes.to(legality_device)).all()
    ):
        raise ValueError("tensor batch contains an illegal prefix")
    if bool((target_gaps[valid_mask] < 0).any()) or bool(
        (target_gaps[valid_mask] >= config.gap_vocab).any()
    ):
        raise ValueError("tensor batch contains an invalid gap token")

    batch = OccurrenceBatch(
        previous_prefix_codes=_shift_right(sub_prefix, valid_mask, -1),
        previous_gap_tokens=_shift_right(sub_gaps, valid_mask, -1),
        target_prefix_codes=target_prefixes,
        target_gap_tokens=target_gaps,
        valid_mask=valid_mask,
        lengths=lengths,
        city_ids=city_ids,
        previous_event_time_minutes=previous_times,
        target_event_time_minutes=target_times,
        context_features=context_features,
    )
    batch.validate(config)
    return batch


class OccurrencePlannerTensorDataset(Dataset[Any]):
    def __init__(
        self,
        prefix_codes: torch.Tensor,
        gap_tokens: torch.Tensor,
        lengths: torch.Tensor,
        *,
        event_time_minutes: torch.Tensor | None = None,
        context_features: torch.Tensor | None = None,
        city_ids: torch.Tensor | None = None,
    ) -> None:
        self.prefix_codes = prefix_codes
        self.gap_tokens = gap_tokens
        self.lengths = lengths
        self.event_time_minutes = event_time_minutes
        self.context_features = context_features
        self.city_ids = city_ids
        self._size = int(lengths.shape[0])

    def __len__(self) -> int:
        return self._size

    def __getitem__(
        self, indices: Any
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        return (
            self.prefix_codes[indices],
            self.gap_tokens[indices],
            self.lengths[indices],
            (
                None
                if self.event_time_minutes is None
                else self.event_time_minutes[indices]
            ),
            (
                None
                if self.context_features is None
                else self.context_features[indices]
            ),
            None if self.city_ids is None else self.city_ids[indices],
        )


def _collate_occurrence_tensor_batch(
    batch_tensors: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ],
    *,
    config: OccurrencePlannerConfig,
    legality: LegalPrefixIndex,
) -> OccurrenceBatch:
    prefixes, gaps, lengths, times, context, city_ids = batch_tensors
    return collate_occurrence_tensors(
        prefixes,
        gaps,
        lengths,
        config=config,
        legality=legality,
        event_time_minutes=times,
        context_features=context,
        city_ids=city_ids,
    )


class _SeededShuffleSampler(Sampler[int]):
    def __init__(self, size: int, *, seed: int | None) -> None:
        if size < 1:
            raise ValueError("sampler size must be positive")
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
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if sampler is not None and shuffle:
        raise ValueError(
            "explicit sampler and shuffle=True are mutually exclusive"
        )
    if sampler is None:
        if shuffle:
            active_sampler: Sampler[int] = _SeededShuffleSampler(
                len(dataset), seed=seed
            )
        else:
            active_sampler = SequentialSampler(dataset)
    else:
        active_sampler = sampler
    batch_sampler = BatchSampler(
        active_sampler, batch_size=batch_size, drop_last=False
    )

    return DataLoader(
        dataset,
        batch_size=None,
        sampler=batch_sampler,
        collate_fn=partial(
            _collate_occurrence_tensor_batch,
            config=config,
            legality=legality,
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
