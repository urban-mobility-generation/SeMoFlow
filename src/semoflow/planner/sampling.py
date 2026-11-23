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
from semoflow.planner.layers import sample_categorical_logits
from semoflow.planner.model import OccurrenceARPlanner

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
    prefixes = torch.full(
        (batch_size, max_length + 1, model.config.prefix_depth),
        -1,
        dtype=torch.long,
        device=device,
    )
    gaps = torch.full(
        (batch_size, max_length + 1), -1, dtype=torch.long, device=device
    )
    times = torch.zeros(
        (batch_size, max_length + 1), dtype=torch.float32, device=device
    )
    cache = model.initialize_generation_cache(
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        dtype=dtype,
    )
    for step in range(max_length):
        active = lengths > step
        gap, prefix, time = model.sample_step(
            prefixes,
            gaps,
            times,
            lengths=lengths,
            city_ids=city,
            step=step,
            active=active,
            cache=cache,
            temperature=temperature,
            route_temperature=route_temperature,
            generator=generator,
        )
        gaps[active, step + 1] = gap
        prefixes[active, step + 1] = prefix
        times[active, step + 1] = time
    padding = torch.arange(max_length, device=device) >= lengths[:, None]
    return {
        "prefix_codes": prefixes[:, 1:]
        .masked_fill(padding[..., None], 0)
        .cpu(),
        "gap_tokens": gaps[:, 1:].masked_fill(padding, 0).cpu(),
        "event_time_minutes": times[:, 1:].cpu(),
        "lengths": lengths.cpu(),
        "city_ids": city.cpu(),
    }


def _concat_padded(
    chunks: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
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
