from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from semoflow.planner.artifacts import (
    load_occurrence_planner_checkpoint,
    load_occurrence_planner_conditions,
    save_occurrence_planner_samples,
)
from semoflow.planner.sampling import sample_occurrence_sequences
from semoflow.planner.types import OccurrenceSample

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OccurrencePlannerSampleResult:
    output_path: Path
    num_samples: int
    batch_size: int


# Column order of the stored [N, 4] log-probability tensors.
_LOG_PROBABILITY_TERMS = ("length", "gap", "duration", "prefix")


def _pack_samples(
    samples: list[OccurrenceSample], *, prefix_depth: int, temporal: bool
) -> dict[str, Any]:
    """Stack sampled sequences into zero-padded sample-artifact tensors."""
    num_samples = len(samples)
    max_length = max(sample.sequence.length for sample in samples)
    prefix_codes = torch.zeros(
        (num_samples, max_length, prefix_depth), dtype=torch.long
    )
    gap_tokens = torch.zeros((num_samples, max_length), dtype=torch.long)
    event_time_minutes = (
        torch.zeros((num_samples, max_length), dtype=torch.float32)
        if temporal
        else None
    )
    city_ids = torch.empty(num_samples, dtype=torch.long)
    for row, sample in enumerate(samples):
        sequence = sample.sequence
        prefix_codes[row, : sequence.length] = sequence.prefix_codes
        gap_tokens[row, : sequence.length] = sequence.gap_tokens
        if event_time_minutes is not None:
            assert sequence.event_time_minutes is not None
            event_time_minutes[row, : sequence.length] = (
                sequence.event_time_minutes
            )
        if sequence.city_id is None:
            raise RuntimeError("sampled planner sequence is missing city_id")
        city_ids[row] = sequence.city_id

    def log_probabilities(kind: str) -> torch.Tensor:
        return torch.tensor(
            [
                [
                    getattr(getattr(sample, kind), term)
                    for term in _LOG_PROBABILITY_TERMS
                ]
                for sample in samples
            ],
            dtype=torch.float32,
        )

    return {
        "prefix_codes": prefix_codes,
        "gap_tokens": gap_tokens,
        "lengths": torch.tensor(
            [sample.sequence.length for sample in samples], dtype=torch.long
        ),
        "model_log_probability": log_probabilities("model_log_probability"),
        "sampling_log_probability": log_probabilities(
            "sampling_log_probability"
        ),
        "length_was_sampled": torch.tensor(
            [sample.length_was_sampled for sample in samples],
            dtype=torch.bool,
        ),
        "city_ids": city_ids,
        "event_time_minutes": event_time_minutes,
    }


def sample_occurrence_planner(
    *,
    checkpoint_path: Path,
    output_path: Path,
    num_samples: int,
    batch_size: int = 1024,
    conditions_path: Path | None = None,
    device: str = "cpu",
    seed: int = 0,
    temperature: float = 1.0,
    length_temperature: float | None = None,
    gap_temperature: float | None = None,
    prefix_temperature: float | None = None,
    route_temperature: float | None = None,
    force: bool = False,
) -> OccurrencePlannerSampleResult:
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing artifact: {output_path}"
        )
    checkpoint = load_occurrence_planner_checkpoint(checkpoint_path)
    logger.info(
        "loaded planner checkpoint  path=%s  epoch=%d  validation_metric=%.4f  "
        "temporal_state=%s",
        checkpoint_path,
        checkpoint.epoch,
        checkpoint.validation_metric,
        checkpoint.config.temporal_state_enabled,
    )

    lengths: torch.Tensor | None = None
    context_features: torch.Tensor | None = None
    city_ids: torch.Tensor | None = None
    if conditions_path is not None:
        conditions = load_occurrence_planner_conditions(
            conditions_path,
            num_samples=num_samples,
            context_dim=checkpoint.config.context_dim,
            context_schema=checkpoint.config.context_schema,
            max_occurrences=checkpoint.config.max_occurrences,
            catalog_city_ids=checkpoint.catalog_city_ids,
        )
        lengths = conditions.lengths
        context_features = conditions.context_features
        city_ids = conditions.city_ids
        logger.info("loaded planner conditions  path=%s", conditions_path)
    elif (
        checkpoint.config.context_dim > 0
        or checkpoint.catalog_city_ids.numel() > 1
    ):
        raise ValueError(
            "planner checkpoint requires per-sample conditions; --conditions is required"
        )

    device_obj = torch.device(device)
    model = checkpoint.model.to(device_obj)
    logger.info(
        "sampling  num_samples=%d  batch_size=%d  device=%s  seed=%d  "
        "temperature=%g",
        num_samples,
        batch_size,
        device,
        seed,
        temperature,
    )
    samples: list[OccurrenceSample] = []
    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        chunk_samples = sample_occurrence_sequences(
            model,
            batch_size=stop - start,
            lengths=(None if lengths is None else lengths[start:stop]),
            context_features=(
                None
                if context_features is None
                else context_features[start:stop]
            ),
            city_ids=None if city_ids is None else city_ids[start:stop],
            temperature=temperature,
            length_temperature=length_temperature,
            gap_temperature=gap_temperature,
            prefix_temperature=prefix_temperature,
            route_temperature=route_temperature,
            seed=seed + start,
        )
        samples.extend(chunk_samples)
        logger.info(
            "sampling progress samples=%d/%d",
            stop,
            num_samples,
        )

    temperatures = {
        "length_temperature": length_temperature,
        "gap_temperature": gap_temperature,
        "prefix_temperature": prefix_temperature,
        "route_temperature": route_temperature,
    }
    save_occurrence_planner_samples(
        output_path,
        **_pack_samples(
            samples,
            prefix_depth=checkpoint.config.prefix_depth,
            temporal=checkpoint.config.temporal_state_enabled,
        ),
        context_features=context_features,
        context_schema=checkpoint.config.context_schema,
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
