from __future__ import annotations

from collections.abc import Sequence

import torch

from semoflow.planner.decoders import sample_categorical_logits
from semoflow.planner.model import OccurrenceARPlanner
from semoflow.planner.types import (
    OccurrenceBatch,
    OccurrenceLogProbability,
    OccurrenceSample,
    OccurrenceSequence,
)

# Per-step log-probability terms accumulated over a sampled sequence.
_STEP_TERMS = ("gap", "prefix", "duration")


def _resolve_city_ids(
    model: OccurrenceARPlanner,
    city_ids: Sequence[int] | torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
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
    return city


def _sequence_lengths(
    model: OccurrenceARPlanner,
    length_logits: torch.Tensor,
    lengths: Sequence[int] | torch.Tensor | None,
    *,
    temperature: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sampled (or validated requested) lengths with their model and
    sampling log-probabilities; requested lengths have sampling log-prob 0."""
    if lengths is None:
        sample = sample_categorical_logits(
            length_logits, temperature=temperature, generator=generator
        )
        return (
            sample.choices + 1,
            sample.model_log_probability,
            sample.sampling_log_probability,
        )
    batch_size = length_logits.shape[0]
    requested = torch.as_tensor(
        lengths, dtype=torch.long, device=length_logits.device
    )
    if requested.shape != (batch_size,):
        raise ValueError("lengths must have shape [batch_size]")
    if bool((requested < 1).any()) or bool(
        (requested > model.config.max_occurrences).any()
    ):
        raise ValueError("requested length is outside the vocabulary")
    choice = (requested - 1)[:, None]
    if bool(torch.isneginf(length_logits.gather(-1, choice)).any()):
        raise ValueError("requested length is masked by planner constraints")
    model_log_probability = (
        torch.log_softmax(length_logits.float(), dim=-1)
        .gather(-1, choice)
        .squeeze(-1)
    )
    return (
        requested,
        model_log_probability,
        torch.zeros_like(model_log_probability),
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
    """Autoregressively sample occurrence plans; unset per-head
    temperatures fall back to ``temperature``."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    def resolve(value: float | None) -> float:
        return temperature if value is None else value

    generator = torch.Generator(device=device).manual_seed(seed)
    context = (
        None
        if context_features is None
        else torch.as_tensor(context_features, dtype=dtype, device=device)
    )
    city = _resolve_city_ids(
        model, city_ids, batch_size=batch_size, device=device
    )
    length_logits = model.predict_length(
        batch_size=batch_size, context_features=context, city_ids=city
    )
    sampled_lengths, model_length_lp, sampling_length_lp = _sequence_lengths(
        model,
        length_logits,
        lengths,
        temperature=resolve(length_temperature),
        generator=generator,
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
    temporal = model.config.temporal_state_enabled
    window = model.config.observation_window_minutes
    event_times = previous_times = None
    if temporal:
        event_times = torch.zeros(
            (batch_size, max_length), dtype=torch.float32, device=device
        )
        previous_times = torch.zeros_like(event_times)
    valid = (
        torch.arange(max_length, device=device)[None, :]
        < sampled_lengths[:, None]
    )
    totals = {
        (kind, term): torch.zeros(
            batch_size, dtype=torch.float32, device=device
        )
        for kind in ("model", "sampling")
        for term in _STEP_TERMS
    }
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
                gap_temperature=resolve(gap_temperature),
                prefix_temperature=resolve(prefix_temperature),
                route_temperature=resolve(route_temperature),
                generator=generator,
            )
            gaps[active, step] = occurrence.gap_tokens
            prefixes[active, step] = occurrence.prefix_codes
            for (kind, term), total in totals.items():
                value = getattr(occurrence, f"{kind}_{term}_log_probability")
                if value is not None:
                    total[active] += value
            if event_times is not None:
                arrival = occurrence.arrival_time_minutes
                assert arrival is not None and window is not None
                if bool((arrival > window + 1.0e-3).any()):
                    raise RuntimeError(
                        "sampler produced an out-of-window arrival time"
                    )
                event_times[active, step] = arrival
            if step + 1 < max_length:
                continuing = sampled_lengths > step + 1
                previous_gaps[continuing, step + 1] = gaps[continuing, step]
                previous_prefixes[continuing, step + 1] = prefixes[
                    continuing, step
                ]
                if previous_times is not None and event_times is not None:
                    previous_times[continuing, step + 1] = event_times[
                        continuing, step
                    ]
    finally:
        model.train(was_training)

    def log_probability(
        kind: str, length: torch.Tensor, row: int
    ) -> OccurrenceLogProbability:
        return OccurrenceLogProbability(
            length=float(length[row].cpu()),
            **{
                term: float(totals[kind, term][row].cpu())
                for term in _STEP_TERMS
            },
        )

    samples = []
    for row in range(batch_size):
        length = int(sampled_lengths[row])
        samples.append(
            OccurrenceSample(
                sequence=OccurrenceSequence(
                    prefix_codes=prefixes[row, :length].cpu(),
                    gap_tokens=gaps[row, :length].cpu(),
                    event_time_minutes=(
                        None
                        if event_times is None
                        else event_times[row, :length].cpu()
                    ),
                    city_id=int(city[row].cpu()),
                ),
                model_log_probability=log_probability(
                    "model", model_length_lp, row
                ),
                sampling_log_probability=log_probability(
                    "sampling", sampling_length_lp, row
                ),
                length_was_sampled=lengths is None,
            )
        )
    return tuple(samples)
