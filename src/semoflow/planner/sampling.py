from __future__ import annotations

from collections.abc import Sequence

import torch

from semoflow.planner.decoders import (
    sample_categorical_logits,
)
from semoflow.planner.model import OccurrenceARPlanner
from semoflow.planner.types import (
    OccurrenceBatch,
    OccurrenceLogProbability,
    OccurrenceSample,
    OccurrenceSequence,
)


@torch.no_grad()
def sample_occurrence_sequences(
    model: OccurrenceARPlanner,
    *,
    batch_size: int = 1,
    lengths: Sequence[int] | torch.Tensor | None = None,
    context_features: torch.Tensor | None = None,
    city_ids: Sequence[int] | torch.Tensor | None = None,
    temperature: float = 1.0,
    length_temperature: float | None = None,
    gap_temperature: float | None = None,
    prefix_temperature: float | None = None,
    route_temperature: float | None = None,
    seed: int = 0,
) -> tuple[OccurrenceSample, ...]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    length_temperature = (
        temperature if length_temperature is None else length_temperature
    )
    gap_temperature = (
        temperature if gap_temperature is None else gap_temperature
    )
    prefix_temperature = (
        temperature if prefix_temperature is None else prefix_temperature
    )
    route_temperature = (
        temperature if route_temperature is None else route_temperature
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    context = (
        None
        if context_features is None
        else torch.as_tensor(context_features, dtype=dtype, device=device)
    )
    city = (
        None if city_ids is None else torch.as_tensor(city_ids, device=device)
    )
    if city is not None and city.dtype != torch.long:
        raise ValueError("city_ids must use torch.long")
    if city is None and model.catalog_city_ids.numel() == 1:
        city = model.catalog_city_ids[0].expand(batch_size)
    if city is None:
        raise ValueError("city_ids are required for a multi-city planner")
    if city.shape != (batch_size,):
        raise ValueError("city_ids must have shape [batch_size]")
    model._catalog_rows_for_city_ids(city, batch_size=batch_size)
    length_logits = model.predict_length(
        batch_size=batch_size, context_features=context, city_ids=city
    )
    length_was_sampled = lengths is None
    if lengths is None:
        length_sample = sample_categorical_logits(
            length_logits,
            temperature=length_temperature,
            generator=generator,
        )
        sampled_lengths = length_sample.choices + 1
        model_length_log_probability = length_sample.model_log_probability
        sampling_length_log_probability = (
            length_sample.sampling_log_probability
        )
    else:
        sampled_lengths = torch.as_tensor(
            lengths, dtype=torch.long, device=device
        )
        if sampled_lengths.shape != (batch_size,):
            raise ValueError("lengths must have shape [batch_size]")
        if bool((sampled_lengths < 1).any()) or bool(
            (sampled_lengths > model.config.max_occurrences).any()
        ):
            raise ValueError("requested length is outside the vocabulary")
        selected_length_logits = length_logits.gather(
            -1, (sampled_lengths - 1)[:, None]
        ).squeeze(-1)
        if bool(torch.isneginf(selected_length_logits).any()):
            raise ValueError(
                "requested length is masked by planner constraints"
            )
        model_length_log_probability = (
            torch.log_softmax(length_logits.float(), dim=-1)
            .gather(-1, (sampled_lengths - 1)[:, None])
            .squeeze(-1)
        )
        sampling_length_log_probability = torch.zeros_like(
            model_length_log_probability
        )

    max_length = int(sampled_lengths.max())
    prefixes = torch.zeros(
        (batch_size, max_length, model.config.prefix_depth),
        dtype=torch.long,
        device=device,
    )
    gaps = torch.zeros(
        (batch_size, max_length), dtype=torch.long, device=device
    )
    previous_prefixes = torch.full_like(prefixes, -1)
    previous_gaps = torch.full_like(gaps, -1)
    temporal_state_enabled = model.config.temporal_state_enabled
    event_times = (
        torch.zeros(
            (batch_size, max_length), dtype=torch.float32, device=device
        )
        if temporal_state_enabled
        else None
    )
    previous_times = (
        torch.zeros(
            (batch_size, max_length), dtype=torch.float32, device=device
        )
        if temporal_state_enabled
        else None
    )
    valid = (
        torch.arange(max_length, device=device)[None, :]
        < sampled_lengths[:, None]
    )
    model_gap_log_probability = torch.zeros(
        batch_size, dtype=torch.float32, device=device
    )
    model_prefix_log_probability = torch.zeros_like(model_gap_log_probability)
    model_duration_log_probability = torch.zeros_like(
        model_gap_log_probability
    )
    sampling_gap_log_probability = torch.zeros_like(model_gap_log_probability)
    sampling_prefix_log_probability = torch.zeros_like(
        model_gap_log_probability
    )
    sampling_duration_log_probability = torch.zeros_like(
        model_gap_log_probability
    )
    was_training = model.training
    model.eval()
    try:
        cache = model.initialize_generation_cache(
            batch_size=batch_size, device=device, dtype=dtype
        )
        for step in range(max_length):
            active = sampled_lengths > step
            batch = OccurrenceBatch(
                previous_prefix_codes=previous_prefixes,
                previous_gap_tokens=previous_gaps,
                target_prefix_codes=prefixes,
                target_gap_tokens=gaps,
                valid_mask=valid,
                lengths=sampled_lengths,
                city_ids=city,
                previous_event_time_minutes=previous_times,
                target_event_time_minutes=previous_times,
                context_features=context,
            )
            occurrence, cache = model._sample_next_occurrence_cached(
                batch,
                step=step,
                active=active,
                cache=cache,
                gap_temperature=gap_temperature,
                prefix_temperature=prefix_temperature,
                route_temperature=route_temperature,
                generator=generator,
            )
            gaps[active, step] = occurrence.gap_tokens
            prefixes[active, step] = occurrence.prefix_codes
            model_gap_log_probability[active] += (
                occurrence.model_gap_log_probability
            )
            model_prefix_log_probability[active] += (
                occurrence.model_prefix_log_probability
            )
            if occurrence.model_duration_log_probability is not None:
                model_duration_log_probability[active] += (
                    occurrence.model_duration_log_probability
                )
            sampling_gap_log_probability[active] += (
                occurrence.sampling_gap_log_probability
            )
            sampling_prefix_log_probability[active] += (
                occurrence.sampling_prefix_log_probability
            )
            if occurrence.sampling_duration_log_probability is not None:
                sampling_duration_log_probability[active] += (
                    occurrence.sampling_duration_log_probability
                )
            if temporal_state_enabled:
                assert (
                    event_times is not None
                    and occurrence.arrival_time_minutes is not None
                )
                window = model.config.observation_window_minutes
                assert window is not None
                if bool(
                    (occurrence.arrival_time_minutes > window + 1.0e-3).any()
                ):
                    raise RuntimeError(
                        "sampler produced an out-of-window arrival time"
                    )
                event_times[active, step] = occurrence.arrival_time_minutes
            if step + 1 < max_length:
                continuing = sampled_lengths > step + 1
                previous_gaps[continuing, step + 1] = gaps[continuing, step]
                previous_prefixes[continuing, step + 1] = prefixes[
                    continuing, step
                ]
                if temporal_state_enabled:
                    assert (
                        event_times is not None and previous_times is not None
                    )
                    previous_times[continuing, step + 1] = event_times[
                        continuing, step
                    ]
    finally:
        model.train(was_training)

    return tuple(
        OccurrenceSample(
            sequence=OccurrenceSequence(
                prefix_codes=prefixes[row, : int(sampled_lengths[row])].cpu(),
                gap_tokens=gaps[row, : int(sampled_lengths[row])].cpu(),
                event_time_minutes=(
                    None
                    if event_times is None
                    else event_times[row, : int(sampled_lengths[row])].cpu()
                ),
                city_id=int(city[row].cpu()),
            ),
            model_log_probability=OccurrenceLogProbability(
                length=float(model_length_log_probability[row].cpu()),
                gap=float(model_gap_log_probability[row].cpu()),
                prefix=float(model_prefix_log_probability[row].cpu()),
                duration=float(model_duration_log_probability[row].cpu()),
            ),
            sampling_log_probability=OccurrenceLogProbability(
                length=float(sampling_length_log_probability[row].cpu()),
                gap=float(sampling_gap_log_probability[row].cpu()),
                prefix=float(sampling_prefix_log_probability[row].cpu()),
                duration=float(sampling_duration_log_probability[row].cpu()),
            ),
            length_was_sampled=length_was_sampled,
        )
        for row in range(batch_size)
    )
