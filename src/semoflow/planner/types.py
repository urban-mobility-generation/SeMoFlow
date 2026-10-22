from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from semoflow.planner.config import (
        OccurrencePlannerConfig,
    )


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

    def validate(
        self,
        config: OccurrencePlannerConfig,
        *,
        validate_temporal_targets: bool = True,
    ) -> None:
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
        integer_tensors = {
            "previous prefixes": prefixes,
            "previous gaps": self.previous_gap_tokens,
            "target prefixes": self.target_prefix_codes,
            "target gaps": self.target_gap_tokens,
            "lengths": self.lengths,
        }
        if self.city_ids is not None:
            integer_tensors["city_ids"] = self.city_ids
        if any(
            tensor.dtype != torch.long for tensor in integer_tensors.values()
        ):
            raise ValueError(
                "occurrence tokens and lengths must use torch.long"
            )
        devices = {tensor.device for tensor in integer_tensors.values()}
        devices.add(self.valid_mask.device)
        if self.context_features is not None:
            devices.add(self.context_features.device)
        if len(devices) != 1:
            raise ValueError(
                "all occurrence batch tensors must share one device"
            )
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
        if config.context_dim == 0:
            if self.context_features is not None:
                raise ValueError("context supplied to a context-free planner")
        elif self.context_features is None:
            raise ValueError("context_features are required")
        elif self.context_features.shape != (batch_size, config.context_dim):
            raise ValueError("context_features have the wrong shape")
        elif not self.context_features.is_floating_point():
            raise ValueError("context_features must use a floating dtype")
        elif not bool(torch.isfinite(self.context_features).all()):
            raise ValueError("context_features must be finite")
        if config.temporal_state_enabled:
            previous_time = self.previous_event_time_minutes
            target_time = self.target_event_time_minutes
            if previous_time is None or target_time is None:
                raise ValueError(
                    "a planner with a time channel requires event-time tensors"
                )
            if previous_time.shape != (
                batch_size,
                steps,
            ) or target_time.shape != (
                batch_size,
                steps,
            ):
                raise ValueError(
                    "event-time tensors do not align with history"
                )
            if (
                not previous_time.is_floating_point()
                or not target_time.is_floating_point()
            ):
                raise ValueError(
                    "event-time tensors must use a floating dtype"
                )
            if (
                previous_time.device != prefixes.device
                or target_time.device != prefixes.device
            ):
                raise ValueError(
                    "all occurrence batch tensors must share one device"
                )
            if not bool(torch.isfinite(previous_time).all()) or not bool(
                torch.isfinite(target_time).all()
            ):
                raise ValueError("event-time tensors must be finite")
            if bool((previous_time < 0).any()) or bool(
                (target_time < 0).any()
            ):
                raise ValueError("event-time tensors must be non-negative")
            if not bool((previous_time[:, 0] == 0).all()):
                raise ValueError(
                    "previous event time must start at the observation window"
                )
            window = config.observation_window_minutes
            assert window is not None
            if bool((target_time[self.valid_mask] > window + 1e-3).any()):
                raise ValueError(
                    "target event time exceeds the observation window"
                )
            if bool(
                (
                    target_time[self.valid_mask] + 1e-6
                    < previous_time[self.valid_mask]
                ).any()
            ):
                raise ValueError(
                    "target event time must not precede the previous event time"
                )
            if validate_temporal_targets:
                representatives = config.gap_duration_minutes
                assert representatives is not None
                representative = torch.tensor(
                    representatives,
                    dtype=target_time.dtype,
                    device=target_time.device,
                )
                expected_target = (
                    previous_time + representative[self.target_gap_tokens]
                )
                expected_target = expected_target.clamp(max=window)
                if not torch.allclose(
                    target_time[self.valid_mask],
                    expected_target[self.valid_mask],
                    atol=1.0e-4,
                    rtol=0.0,
                ):
                    raise ValueError(
                        "event-time state disagrees with gap reconstruction"
                    )
                if steps > 1:
                    shifted = self.valid_mask[:, 1:]
                    if not torch.allclose(
                        previous_time[:, 1:][shifted],
                        target_time[:, :-1][shifted],
                        atol=1.0e-4,
                        rtol=0.0,
                    ):
                        raise ValueError(
                            "previous event-time state must equal the prior target"
                        )
        elif (
            self.previous_event_time_minutes is not None
            or self.target_event_time_minutes is not None
        ):
            raise ValueError(
                "event-time tensors supplied to a planner without a time channel"
            )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> OccurrenceBatch:
        def move(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.to(device, non_blocking=non_blocking)

        return OccurrenceBatch(
            previous_prefix_codes=move(self.previous_prefix_codes),
            previous_gap_tokens=move(self.previous_gap_tokens),
            target_prefix_codes=move(self.target_prefix_codes),
            target_gap_tokens=move(self.target_gap_tokens),
            valid_mask=move(self.valid_mask),
            lengths=move(self.lengths),
            city_ids=move(self.city_ids),
            previous_event_time_minutes=move(self.previous_event_time_minutes),
            target_event_time_minutes=move(self.target_event_time_minutes),
            context_features=move(self.context_features),
        )

    def pin_memory(self) -> OccurrenceBatch:
        def pin(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.pin_memory()

        return OccurrenceBatch(
            previous_prefix_codes=pin(self.previous_prefix_codes),
            previous_gap_tokens=pin(self.previous_gap_tokens),
            target_prefix_codes=pin(self.target_prefix_codes),
            target_gap_tokens=pin(self.target_gap_tokens),
            valid_mask=pin(self.valid_mask),
            lengths=pin(self.lengths),
            city_ids=pin(self.city_ids),
            previous_event_time_minutes=pin(self.previous_event_time_minutes),
            target_event_time_minutes=pin(self.target_event_time_minutes),
            context_features=pin(self.context_features),
        )


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
