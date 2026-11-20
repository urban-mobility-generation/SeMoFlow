from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from semoflow.planner.artifacts import (
    load_occurrence_planner_checkpoint,
    save_occurrence_planner_samples,
)
from semoflow.planner.sampling import sample_occurrence_sequences

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OccurrencePlannerSampleResult:
    output_path: Path
    num_samples: int
    batch_size: int


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
    length_temperature: float | None = None,
    gap_temperature: float | None = None,
    prefix_temperature: float | None = None,
    route_temperature: float | None = None,
    force: bool = False,
) -> OccurrencePlannerSampleResult:
    """Sample ``num_samples`` plans for one city; ``city_id`` may be omitted
    for a single-city checkpoint."""
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing artifact: {output_path}"
        )
    checkpoint = load_occurrence_planner_checkpoint(checkpoint_path)
    catalog_city_ids = checkpoint.catalog_city_ids.tolist()
    if city_id is None:
        if len(catalog_city_ids) != 1:
            raise ValueError(
                f"the planner covers cities {catalog_city_ids}; pass a city id"
            )
        city_id = catalog_city_ids[0]
    elif city_id not in catalog_city_ids:
        raise ValueError(
            f"city {city_id} is not in the planner catalog {catalog_city_ids}"
        )
    logger.info(
        "loaded planner checkpoint  path=%s  epoch=%d  validation_metric=%.4f",
        checkpoint_path,
        checkpoint.epoch,
        checkpoint.validation_metric,
    )
    model = checkpoint.model.to(torch.device(device))
    logger.info(
        "sampling  num_samples=%d  city_id=%d  batch_size=%d  device=%s  "
        "seed=%d  temperature=%g",
        num_samples,
        city_id,
        batch_size,
        device,
        seed,
        temperature,
    )
    chunks = []
    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        chunks.append(
            sample_occurrence_sequences(
                model,
                city_ids=torch.full((stop - start,), city_id),
                temperature=temperature,
                length_temperature=length_temperature,
                gap_temperature=gap_temperature,
                prefix_temperature=prefix_temperature,
                route_temperature=route_temperature,
                seed=seed + start,
            )
        )
        logger.info("sampling progress samples=%d/%d", stop, num_samples)

    temperatures = {
        "length_temperature": length_temperature,
        "gap_temperature": gap_temperature,
        "prefix_temperature": prefix_temperature,
        "route_temperature": route_temperature,
    }
    save_occurrence_planner_samples(
        output_path,
        **_concat_padded(chunks),
        seed=seed,
        temperature=temperature,
        **{
            name: temperature if value is None else value
            for name, value in temperatures.items()
        },
        force=force,
    )
    logger.info(
        "sampling complete  output=%s  num_samples=%d",
        output_path,
        num_samples,
    )
    return OccurrencePlannerSampleResult(
        output_path=output_path,
        num_samples=num_samples,
        batch_size=batch_size,
    )


__all__ = ["OccurrencePlannerSampleResult", "sample_occurrence_planner"]
