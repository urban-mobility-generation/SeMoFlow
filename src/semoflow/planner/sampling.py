from __future__ import annotations

import torch

from semoflow.planner.decoders import sample_categorical_logits
from semoflow.planner.model import OccurrenceARPlanner
from semoflow.planner.types import OccurrenceBatch


@torch.no_grad()
def sample_occurrence_sequences(
    model: OccurrenceARPlanner,
    *,
    city_ids: torch.Tensor,
    temperature: float = 1.0,
    route_temperature: float | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Autoregressively sample one occurrence plan per city id.

    Returns zero-padded ``prefix_codes`` [B, L, P], ``gap_tokens`` and
    ``event_time_minutes`` [B, L], ``lengths`` and ``city_ids`` [B]. The Explore/Retrieve gate uses
    ``route_temperature`` when set, else ``temperature``.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if route_temperature is None:
        route_temperature = temperature
    generator = torch.Generator(device=device).manual_seed(seed)
    city = city_ids.to(device)
    batch_size = int(city.shape[0])
    lengths = (
        sample_categorical_logits(
            model.predict_length(city),
            temperature=temperature,
            generator=generator,
        )
        + 1
    )
    max_length = int(lengths.max())
    prefixes = torch.zeros(
        (batch_size, max_length, model.config.prefix_depth),
        dtype=torch.long,
        device=device,
    )
    gaps = torch.zeros(
        (batch_size, max_length), dtype=torch.long, device=device
    )
    event_times = torch.zeros(
        (batch_size, max_length), dtype=torch.float32, device=device
    )
    previous_prefixes = torch.full_like(prefixes, -1)
    previous_gaps = torch.full_like(gaps, -1)
    previous_times = torch.zeros_like(event_times)
    valid = torch.arange(max_length, device=device)[None, :] < lengths[:, None]
    was_training = model.training
    model.eval()
    try:
        cache = model.initialize_generation_cache(
            batch_size=batch_size, device=device, dtype=dtype
        )
        for step in range(max_length):
            active = lengths > step
            batch = OccurrenceBatch(
                previous_prefix_codes=previous_prefixes,
                previous_gap_tokens=previous_gaps,
                previous_event_time_minutes=previous_times,
                target_prefix_codes=prefixes,
                target_gap_tokens=gaps,
                valid_mask=valid,
                lengths=lengths,
                city_ids=city,
            )
            occurrence, cache = model.sample_next_occurrence(
                batch,
                step=step,
                active=active,
                cache=cache,
                gap_temperature=temperature,
                prefix_temperature=temperature,
                route_temperature=route_temperature,
                generator=generator,
            )
            gaps[active, step] = occurrence.gap_tokens
            prefixes[active, step] = occurrence.prefix_codes
            event_times[active, step] = occurrence.arrival_time_minutes
            if step + 1 < max_length:
                continuing = lengths > step + 1
                for previous, current in (
                    (previous_gaps, gaps),
                    (previous_prefixes, prefixes),
                    (previous_times, event_times),
                ):
                    previous[continuing, step + 1] = current[continuing, step]
    finally:
        model.train(was_training)
    return {
        "prefix_codes": prefixes.cpu(),
        "gap_tokens": gaps.cpu(),
        "event_time_minutes": event_times.cpu(),
        "lengths": lengths.cpu(),
        "city_ids": city.cpu(),
    }
