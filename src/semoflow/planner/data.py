from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
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
    gap_tokens_from_event_time_minutes,
    reconstruct_event_time_minutes,
)
from semoflow.planner.legality import LegalPrefixIndex
from semoflow.planner.types import (
    OccurrenceBatch,
    OccurrenceSequence,
)


def collate_occurrence_sequences(
    sequences: Sequence[OccurrenceSequence],
    *,
    config: OccurrencePlannerConfig,
    legality: LegalPrefixIndex,
    context_features: torch.Tensor | None = None,
    city_ids: torch.Tensor | None = None,
) -> OccurrenceBatch:
    if not sequences:
        raise ValueError("sequences must not be empty")
    if legality.codebook_size != config.codebook_size:
        raise ValueError("legality and config codebook sizes differ")
    if legality.prefix_depth != config.prefix_depth:
        raise ValueError("legality and config prefix depths differ")
    if legality.codes.device.type != "cpu":
        raise ValueError(
            "collation requires a separate CPU legality index; do not pass "
            "the model-device legality index"
        )
    batch_size = len(sequences)
    if city_ids is None:
        sequence_city_ids = [sequence.city_id for sequence in sequences]
        if any(value is not None for value in sequence_city_ids):
            if any(value is None for value in sequence_city_ids):
                raise ValueError(
                    "sequence city_id must be present for every row"
                )
            city_ids = torch.tensor(sequence_city_ids, dtype=torch.long)
    else:
        city_ids = torch.as_tensor(city_ids)
        if city_ids.dtype != torch.long:
            raise ValueError("city_ids must use torch.long")
        if city_ids.shape != (batch_size,):
            raise ValueError("city_ids must have shape [batch]")
        for row, sequence in enumerate(sequences):
            if (
                sequence.city_id is not None
                and int(city_ids[row]) != sequence.city_id
            ):
                raise ValueError(
                    "explicit city_ids disagree with sequence city_id"
                )
    max_length = max(sequence.length for sequence in sequences)
    if max_length > config.max_occurrences:
        raise ValueError("sequence length exceeds max_occurrences")
    if context_features is not None and not config.context_dim:
        raise ValueError("context supplied to a context-free planner")
    if context_features is not None:
        context_features = torch.as_tensor(
            context_features, dtype=torch.float32
        )
        if context_features.shape != (batch_size, config.context_dim):
            raise ValueError("context_features have the wrong shape")
    elif config.context_dim:
        raise ValueError("context_features are required by the config")

    previous_prefixes = torch.full(
        (batch_size, max_length, config.prefix_depth), -1, dtype=torch.long
    )
    previous_gaps = torch.full((batch_size, max_length), -1, dtype=torch.long)
    target_prefixes = (
        legality.codes[0]
        .cpu()
        .view(1, 1, -1)
        .expand(batch_size, max_length, -1)
        .clone()
    )
    target_gaps = torch.zeros_like(previous_gaps)
    valid = torch.zeros((batch_size, max_length), dtype=torch.bool)
    lengths = torch.empty(batch_size, dtype=torch.long)
    previous_times: torch.Tensor | None = None
    target_times: torch.Tensor | None = None
    if config.temporal_state_enabled:
        previous_times = torch.zeros(
            (batch_size, max_length), dtype=torch.float32
        )
        target_times = torch.zeros(
            (batch_size, max_length), dtype=torch.float32
        )

    for row, sequence in enumerate(sequences):
        if sequence.prefix_codes.shape[1] != config.prefix_depth:
            raise ValueError("sequence prefix depth differs from config")
        if bool((sequence.prefix_codes >= config.codebook_size).any()):
            raise ValueError("sequence prefix code exceeds the codebook")
        if bool((sequence.gap_tokens >= config.gap_vocab).any()):
            raise ValueError("sequence gap token exceeds the vocabulary")
        if not bool(legality.contains(sequence.prefix_codes).all()):
            raise ValueError("sequence contains an illegal prefix")
        if (
            config.temporal_state_enabled
            and sequence.event_time_minutes is None
        ):
            raise ValueError(
                "sequence.event_time_minutes is required by a planner with a "
                "time channel"
            )
        if (
            not config.temporal_state_enabled
            and sequence.event_time_minutes is not None
        ):
            raise ValueError(
                "sequence.event_time_minutes supplied to a planner without a "
                "time channel"
            )
        length = sequence.length
        lengths[row] = length
        valid[row, :length] = True
        target_prefixes[row, :length] = sequence.prefix_codes
        target_gaps[row, :length] = sequence.gap_tokens
        if length > 1:
            previous_prefixes[row, 1:length] = sequence.prefix_codes[:-1]
            previous_gaps[row, 1:length] = sequence.gap_tokens[:-1]
        if config.temporal_state_enabled:
            assert previous_times is not None and target_times is not None
            event_time = sequence.event_time_minutes
            assert event_time is not None
            window = config.observation_window_minutes
            edges = config.gap_bucket_edges_minutes
            assert window is not None and edges is not None
            if bool((event_time > window + 1e-3).any()):
                raise ValueError(
                    "sequence.event_time_minutes exceeds the observation window"
                )
            expected_gap_tokens = gap_tokens_from_event_time_minutes(
                event_time,
                bucket_edges_minutes=edges,
            )
            if not torch.equal(sequence.gap_tokens, expected_gap_tokens):
                raise ValueError(
                    "sequence gap_tokens disagree with event_time_minutes"
                )
            representatives = config.gap_duration_minutes
            assert representatives is not None
            reconstructed = reconstruct_event_time_minutes(
                sequence.gap_tokens,
                representative_minutes=representatives,
                observation_window_minutes=window,
            )
            target_times[row, :length] = reconstructed
            if length > 1:
                previous_times[row, 1:length] = reconstructed[:-1]

    return OccurrenceBatch(
        previous_prefix_codes=previous_prefixes,
        previous_gap_tokens=previous_gaps,
        target_prefix_codes=target_prefixes,
        target_gap_tokens=target_gaps,
        valid_mask=valid,
        lengths=lengths,
        city_ids=city_ids,
        previous_event_time_minutes=previous_times,
        target_event_time_minutes=target_times,
        context_features=context_features,
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
        city_ids = torch.as_tensor(city_ids)
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
        context_features = torch.as_tensor(
            context_features, dtype=torch.float32
        )
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

    device = prefix_codes.device
    max_len = int(lengths.max().item())
    if max_len > config.max_occurrences:
        raise ValueError("sequence length exceeds max_occurrences")

    sub_prefix = prefix_codes[:, :max_len]
    sub_gaps = gap_tokens[:, :max_len]

    steps_arange = torch.arange(max_len, device=device)
    valid_mask = steps_arange[None, :] < lengths[:, None]

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

    previous_prefixes = torch.full(
        (batch_size, max_len, config.prefix_depth),
        -1,
        dtype=torch.long,
        device=device,
    )
    previous_gaps = torch.full(
        (batch_size, max_len), -1, dtype=torch.long, device=device
    )
    if max_len > 1:
        valid_prev = valid_mask[:, 1:]
        previous_prefixes[:, 1:] = torch.where(
            valid_prev[..., None],
            sub_prefix[:, : max_len - 1],
            -1,
        )
        previous_gaps[:, 1:] = torch.where(
            valid_prev,
            sub_gaps[:, : max_len - 1],
            -1,
        )

    previous_times: torch.Tensor | None = None
    target_times: torch.Tensor | None = None
    if config.temporal_state_enabled:
        assert event_time_minutes is not None
        window = config.observation_window_minutes
        edges = config.gap_bucket_edges_minutes
        representatives = config.gap_duration_minutes
        assert window is not None and edges is not None
        assert representatives is not None
        raw_times = event_time_minutes[:, :max_len].float()
        raw_previous = torch.zeros_like(raw_times)
        if max_len > 1:
            raw_previous[:, 1:] = raw_times[:, : max_len - 1]
        raw_durations = raw_times - raw_previous
        if bool((raw_durations[valid_mask] < 0).any()):
            raise ValueError("event_time_minutes must be non-decreasing")
        boundaries = torch.tensor(edges, dtype=raw_times.dtype, device=device)
        expected_gaps = torch.bucketize(
            raw_durations, boundaries, right=False
        ).to(torch.long)
        if not torch.equal(target_gaps[valid_mask], expected_gaps[valid_mask]):
            raise ValueError("gap_tokens disagree with raw event_time_minutes")
        reconstructed = reconstruct_event_time_minutes(
            target_gaps,
            representative_minutes=representatives,
            observation_window_minutes=window,
            valid_mask=valid_mask,
        )
        target_times = torch.where(valid_mask, reconstructed, 0.0)
        previous_times = torch.zeros(
            (batch_size, max_len), dtype=torch.float32, device=device
        )
        if max_len > 1:
            previous_times[:, 1:] = torch.where(
                valid_mask[:, 1:],
                reconstructed[:, : max_len - 1],
                0.0,
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
        previous_prefix_codes=previous_prefixes,
        previous_gap_tokens=previous_gaps,
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
