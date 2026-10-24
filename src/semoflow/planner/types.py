from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from semoflow.planner.config import OccurrencePlannerConfig


@dataclass(frozen=True, slots=True)
class OccurrenceSequence:
    prefix_codes: torch.Tensor
    gap_tokens: torch.Tensor
    event_time_minutes: torch.Tensor | None = None
    city_id: int | None = None

    def __post_init__(self) -> None:
        prefixes = torch.as_tensor(self.prefix_codes, dtype=torch.long)
        gaps = torch.as_tensor(self.gap_tokens, dtype=torch.long)
        if prefixes.ndim != 2 or prefixes.shape[0] < 1:
            raise ValueError("prefix_codes must have shape [length, depth]")
        if gaps.shape != (prefixes.shape[0],):
            raise ValueError("gap_tokens must align with prefix_codes")
        if bool((prefixes < 0).any()) or bool((gaps < 0).any()):
            raise ValueError("occurrence tokens must be non-negative")
        object.__setattr__(self, "prefix_codes", prefixes.clone())
        object.__setattr__(self, "gap_tokens", gaps.clone())
        if self.event_time_minutes is not None:
            event_time = torch.as_tensor(
                self.event_time_minutes, dtype=torch.float32
            )
            if event_time.shape != (prefixes.shape[0],):
                raise ValueError(
                    "event_time_minutes must align with prefix_codes"
                )
            if not bool(torch.isfinite(event_time).all()):
                raise ValueError("event_time_minutes must be finite")
            if bool((event_time < 0).any()):
                raise ValueError("event_time_minutes must be non-negative")
            if event_time.shape[0] > 1 and bool(
                (event_time[1:] < event_time[:-1]).any()
            ):
                raise ValueError("event_time_minutes must be non-decreasing")
            object.__setattr__(self, "event_time_minutes", event_time.clone())
        if self.city_id is not None and (
            not isinstance(self.city_id, int) or isinstance(self.city_id, bool)
        ):
            raise ValueError("city_id must be an integer when supplied")

    @property
    def length(self) -> int:
        return int(self.prefix_codes.shape[0])


@dataclass(frozen=True, slots=True)
class OccurrenceBatch:
    previous_prefix_codes: torch.Tensor
    previous_gap_tokens: torch.Tensor
    target_prefix_codes: torch.Tensor
    target_gap_tokens: torch.Tensor
    valid_mask: torch.Tensor
    lengths: torch.Tensor
    city_ids: torch.Tensor | None = None
    previous_event_time_minutes: torch.Tensor | None = None
    target_event_time_minutes: torch.Tensor | None = None
    context_features: torch.Tensor | None = None

    def validate(self, config: OccurrencePlannerConfig) -> None:
        batch_size, steps = self._validate_layout(config)
        self._validate_tokens(config, steps)
        self._validate_context(config, batch_size)
        self._validate_event_times(config, batch_size, steps)

    def _validate_layout(
        self, config: OccurrencePlannerConfig
    ) -> tuple[int, int]:
        """Shapes, dtypes and devices; returns ``(batch_size, steps)``."""
        prefixes = self.previous_prefix_codes
        if prefixes.ndim != 3:
            raise ValueError("previous prefixes must have shape [B, S, P]")
        batch_size, steps, depth = prefixes.shape
        if batch_size < 1:
            raise ValueError("occurrence batch must not be empty")
        if depth != config.prefix_depth:
            raise ValueError("history prefix depth differs from config")
        if steps < 1 or steps > config.max_occurrences:
            raise ValueError("history length is outside max_occurrences")
        if self.previous_gap_tokens.shape != (batch_size, steps):
            raise ValueError("previous gaps do not align with prefixes")
        if self.target_prefix_codes.shape != (batch_size, steps, depth):
            raise ValueError("target prefixes do not align with history")
        if self.target_gap_tokens.shape != (batch_size, steps):
            raise ValueError("target gaps do not align with history")
        if self.valid_mask.shape != (batch_size, steps):
            raise ValueError("valid_mask does not align with history")
        if self.lengths.shape != (batch_size,):
            raise ValueError("lengths must have shape [batch]")
        if self.city_ids is not None:
            if self.city_ids.shape != (batch_size,):
                raise ValueError("city_ids must have shape [batch]")
            if self.city_ids.dtype != torch.long:
                raise ValueError("city_ids must use torch.long")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean")
        integer_tensors = [
            prefixes,
            self.previous_gap_tokens,
            self.target_prefix_codes,
            self.target_gap_tokens,
            self.lengths,
        ]
        if self.city_ids is not None:
            integer_tensors.append(self.city_ids)
        if any(tensor.dtype != torch.long for tensor in integer_tensors):
            raise ValueError(
                "occurrence tokens and lengths must use torch.long"
            )
        devices = {tensor.device for tensor in integer_tensors}
        devices.add(self.valid_mask.device)
        if self.context_features is not None:
            devices.add(self.context_features.device)
        if len(devices) != 1:
            raise ValueError(
                "all occurrence batch tensors must share one device"
            )
        return batch_size, steps

    def _validate_tokens(
        self, config: OccurrencePlannerConfig, steps: int
    ) -> None:
        """Lengths, vocabulary ranges and the BOS/padding sentinels."""
        if bool((self.lengths < 1).any()) or bool(
            (self.lengths > config.max_occurrences).any()
        ):
            raise ValueError("length is outside the configured vocabulary")
        if bool((self.lengths > steps).any()):
            raise ValueError(
                "valid_mask cannot represent the declared lengths"
            )
        expected_mask = (
            torch.arange(steps, device=self.lengths.device)[None, :]
            < self.lengths[:, None]
        )
        if not torch.equal(self.valid_mask, expected_mask):
            raise ValueError(
                "valid_mask must be left-contiguous and match lengths"
            )
        prefixes = self.previous_prefix_codes
        previous_prefix_in_range = (prefixes >= 0) & (
            prefixes < config.codebook_size
        )
        previous_prefix_sentinel = prefixes == -1
        if not bool(
            (previous_prefix_in_range | previous_prefix_sentinel).all()
        ):
            raise ValueError("previous prefix code is outside the codebook")
        whole_tuple_sentinel = previous_prefix_sentinel.all(dim=-1)
        whole_tuple_code = previous_prefix_in_range.all(dim=-1)
        if not bool((whole_tuple_sentinel | whole_tuple_code).all()):
            raise ValueError(
                "previous prefix sentinel must cover a whole tuple"
            )
        if bool((self.previous_gap_tokens < -1).any()) or bool(
            (self.previous_gap_tokens >= config.gap_vocab).any()
        ):
            raise ValueError("previous gap token is outside the vocabulary")
        gap_sentinel = self.previous_gap_tokens == -1
        if not torch.equal(gap_sentinel, whole_tuple_sentinel):
            raise ValueError("previous gap and prefix sentinels must align")
        if bool((self.target_prefix_codes < 0).any()) or bool(
            (self.target_prefix_codes >= config.codebook_size).any()
        ):
            raise ValueError("target prefix code is outside the codebook")
        if bool((self.target_gap_tokens < 0).any()) or bool(
            (self.target_gap_tokens >= config.gap_vocab).any()
        ):
            raise ValueError("target gap token is outside the vocabulary")
        if not bool(whole_tuple_sentinel[:, 0].all()):
            raise ValueError(
                "the first history position must use the BOS sentinel"
            )
        observed_history = self.valid_mask.clone()
        observed_history[:, 0] = False
        if bool(whole_tuple_sentinel[observed_history].any()):
            raise ValueError("valid shifted history must not use the sentinel")
        if bool((~whole_tuple_sentinel[~self.valid_mask]).any()):
            raise ValueError("padded history must use the sentinel")

    def _validate_context(
        self, config: OccurrencePlannerConfig, batch_size: int
    ) -> None:
        context = self.context_features
        if config.context_dim == 0:
            if context is not None:
                raise ValueError("context supplied to a context-free planner")
        elif context is None:
            raise ValueError("context_features are required")
        elif context.shape != (batch_size, config.context_dim):
            raise ValueError("context_features have the wrong shape")
        elif not context.is_floating_point():
            raise ValueError("context_features must use a floating dtype")
        elif not bool(torch.isfinite(context).all()):
            raise ValueError("context_features must be finite")

    def _validate_event_times(
        self, config: OccurrencePlannerConfig, batch_size: int, steps: int
    ) -> None:
        """Event-time state must replay the gap tokens inside the window."""
        previous_time = self.previous_event_time_minutes
        target_time = self.target_event_time_minutes
        if not config.temporal_state_enabled:
            if previous_time is not None or target_time is not None:
                raise ValueError(
                    "event-time tensors supplied to a planner without a time channel"
                )
            return
        if previous_time is None or target_time is None:
            raise ValueError(
                "a planner with a time channel requires event-time tensors"
            )
        expected_shape = (batch_size, steps)
        if (
            previous_time.shape != expected_shape
            or target_time.shape != expected_shape
        ):
            raise ValueError("event-time tensors do not align with history")
        if (
            not previous_time.is_floating_point()
            or not target_time.is_floating_point()
        ):
            raise ValueError("event-time tensors must use a floating dtype")
        device = self.previous_prefix_codes.device
        if previous_time.device != device or target_time.device != device:
            raise ValueError(
                "all occurrence batch tensors must share one device"
            )
        if not bool(torch.isfinite(previous_time).all()) or not bool(
            torch.isfinite(target_time).all()
        ):
            raise ValueError("event-time tensors must be finite")
        if bool((previous_time < 0).any()) or bool((target_time < 0).any()):
            raise ValueError("event-time tensors must be non-negative")
        if not bool((previous_time[:, 0] == 0).all()):
            raise ValueError(
                "previous event time must start at the observation window"
            )
        window = config.observation_window_minutes
        assert window is not None
        valid_target = target_time[self.valid_mask]
        if bool((valid_target > window + 1e-3).any()):
            raise ValueError(
                "target event time exceeds the observation window"
            )
        if bool((valid_target + 1e-6 < previous_time[self.valid_mask]).any()):
            raise ValueError(
                "target event time must not precede the previous event time"
            )
        representatives = config.gap_duration_minutes
        assert representatives is not None
        representative = torch.tensor(
            representatives, dtype=target_time.dtype, device=target_time.device
        )
        expected_target = (
            previous_time + representative[self.target_gap_tokens]
        ).clamp(max=window)
        if not torch.allclose(
            valid_target,
            expected_target[self.valid_mask],
            atol=1.0e-4,
            rtol=0.0,
        ):
            raise ValueError(
                "event-time state disagrees with gap reconstruction"
            )
        shifted = self.valid_mask[:, 1:]
        if steps > 1 and not torch.allclose(
            previous_time[:, 1:][shifted],
            target_time[:, :-1][shifted],
            atol=1.0e-4,
            rtol=0.0,
        ):
            raise ValueError(
                "previous event-time state must equal the prior target"
            )

    def _map_tensors(
        self, transform: Callable[[torch.Tensor], torch.Tensor]
    ) -> OccurrenceBatch:
        values = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
        }
        return OccurrenceBatch(
            **{
                name: None if value is None else transform(value)
                for name, value in values.items()
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
    hidden: torch.Tensor
    length_logits: torch.Tensor
    gap_logits: torch.Tensor
    prefix_logits: torch.Tensor
    prefix_targets: torch.Tensor
    duration_log_probability: torch.Tensor | None = None
    prefix_log_probability: torch.Tensor | None = None
    retrieve_probability: torch.Tensor | None = None
    retrieve_responsibility: torch.Tensor | None = None
    retrieval_mechanism_weights: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class OccurrenceGenerationCache:
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]

    @property
    def step(self) -> int:
        return 0 if not self.keys else int(self.keys[0].shape[2])


@dataclass(frozen=True, slots=True)
class OccurrenceLogProbability:
    length: float
    gap: float
    prefix: float
    duration: float = 0.0

    @property
    def total(self) -> float:
        return self.length + self.gap + self.duration + self.prefix


@dataclass(frozen=True, slots=True)
class OccurrenceStepSample:
    gap_tokens: torch.Tensor
    prefix_codes: torch.Tensor
    model_gap_log_probability: torch.Tensor
    model_prefix_log_probability: torch.Tensor
    sampling_gap_log_probability: torch.Tensor
    sampling_prefix_log_probability: torch.Tensor
    model_duration_log_probability: torch.Tensor | None = None
    sampling_duration_log_probability: torch.Tensor | None = None
    arrival_time_minutes: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class OccurrenceSample:
    sequence: OccurrenceSequence
    model_log_probability: OccurrenceLogProbability
    sampling_log_probability: OccurrenceLogProbability
    length_was_sampled: bool


@dataclass(frozen=True, slots=True)
class OccurrencePlannerLoss:
    total: torch.Tensor
    length: torch.Tensor
    gap: torch.Tensor
    prefix: torch.Tensor
    duration: torch.Tensor
    recurrence: torch.Tensor
