from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from semoflow.planner.artifacts import (
    load_occurrence_planner_checkpoint,
    save_occurrence_planner_samples,
)
from semoflow.planner.decoders import sample_categorical_logits
from semoflow.planner.model import OccurrenceARPlanner
from semoflow.planner.types import OccurrenceBatch

logger = logging.getLogger(__name__)


@torch.no_grad()
def sample_occurrence_sequences(
    model: OccurrenceARPlanner,
    *,
    city_ids: torch.Tensor,
    temperature: float,
    route_temperature: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Autoregressively sample one occurrence plan per city id with an
    eval-mode ``model``.

    Returns zero-padded ``prefix_codes`` [B, L, P], ``gap_tokens`` and
    ``event_time_minutes`` [B, L], ``lengths`` and ``city_ids`` [B]. The
    Explore/Retrieve gate samples at ``route_temperature``.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
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
    return {
        "prefix_codes": prefixes.cpu(),
        "gap_tokens": gaps.cpu(),
        "event_time_minutes": event_times.cpu(),
        "lengths": lengths.cpu(),
        "city_ids": city.cpu(),
    }


def _concat_padded(
    chunks: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Concatenate sample chunks, zero-padding the step axis."""
    width = max(int(chunk["lengths"].max()) for chunk in chunks)
    merged: dict[str, torch.Tensor] = {}
    for name in chunks[0]:
        parts = []
        for chunk in chunks:
            value = chunk[name]
            if value.ndim > 1:
                pad = [0, 0] * (value.ndim - 2) + [0, width - value.shape[1]]
                value = F.pad(value, pad)
            parts.append(value)
        merged[name] = torch.cat(parts)
    return merged


def sample_occurrence_planner(
    *,
    checkpoint_path: Path,
    output_path: Path,
    num_samples: int,
    city_id: int | None = None,
    batch_size: int = 1024,
    device: str = "cpu",
    seed: int = 0,
    temperature: float = 1.0,
    route_temperature: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Sample ``num_samples`` plans for one city into ``output_path``;
    ``city_id`` may be omitted for a single-city checkpoint."""
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing artifact: {output_path}"
        )
    model, payload = load_occurrence_planner_checkpoint(checkpoint_path)
    cities = model.catalog_city_ids.tolist()
    if city_id is None:
        if len(cities) != 1:
            raise ValueError(
                f"the planner covers cities {cities}; pass a city id"
            )
        city_id = cities[0]
    elif city_id not in cities:
        raise ValueError(
            f"city {city_id} is not in the planner catalog {cities}"
        )
    logger.info(
        "sampling  checkpoint=%s  epoch=%d  num_samples=%d  city_id=%d  "
        "batch_size=%d  device=%s  seed=%d  temperature=%g",
        checkpoint_path,
        payload["epoch"],
        num_samples,
        city_id,
        batch_size,
        device,
        seed,
        temperature,
    )
    model = model.to(torch.device(device))
    if route_temperature is None:
        route_temperature = temperature
    chunks = []
    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        chunks.append(
            sample_occurrence_sequences(
                model,
                city_ids=torch.full((stop - start,), city_id),
                temperature=temperature,
                route_temperature=route_temperature,
                seed=seed + start,
            )
        )
        logger.info("sampling progress samples=%d/%d", stop, num_samples)
    save_occurrence_planner_samples(
        output_path,
        **_concat_padded(chunks),
        seed=seed,
        temperature=temperature,
        route_temperature=route_temperature,
        force=force,
    )
    return {"output_path": output_path, "num_samples": num_samples}
