from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from semoflow.utils.serialization import normalize_mapping, write_json

logger = logging.getLogger(__name__)

_NESTED_SUMMARY_KEYS = {
    "time_location_joint_jsd": (
        "temporal_spatial",
        "time_location",
        "joint_jsd",
    ),
    "gap_distance_joint_jsd": (
        "temporal_spatial",
        "gap_distance",
        "joint_jsd",
    ),
    "inter_return_time_jsd": ("elapsed_recurrence", "inter_return_time_jsd"),
    "return_index_time_joint_jsd": (
        "elapsed_recurrence",
        "return_index_time_joint_jsd",
    ),
}


def realizer_max_events(checkpoint: Path) -> int:
    """The number of events per trajectory a realizer checkpoint models."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    position = payload["model_state_dict"]["event_position_embedding.weight"]
    return int(position.shape[0])


def summary(metrics: Mapping[str, Any]) -> dict[str, object]:
    """An arm's scalar metrics, with the nested joint JSDs lifted out.

    A metric the arm could not score (no continuous clock) is ``None``.
    """
    out: dict[str, object] = {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, dict)
    }
    for key, (section, *path) in _NESTED_SUMMARY_KEYS.items():
        node: Any = metrics[section]
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        out[key] = node
    return out


def evaluate_planner_realizer(
    *,
    realizer_checkpoint: Path,
    planner_samples: Path,
    bank: Path,
    codebooks: Path,
    lookup: Path,
    grounding_index: Path,
    places: Path,
    output: Path,
    max_samples: int,
    batch_size: int,
    device: str,
    num_steps: int,
    solver: str,
    candidate_selection: str,
    candidate_prior: Path | None = None,
    catalog_temperature: float | None = None,
    reference_split: str = "val",
    catalog_noise_coupling: str = "independent",
) -> dict[str, Any]:
    """Complete planner samples with the realizer and score both arms.

    The gold-prefix arm completes the reference split's own prefixes; the
    sampled-plan arm completes the planner's. Reports go to ``output``.
    """
    from semoflow.realizer.evaluation import evaluate

    started = time.perf_counter()
    logger.info(
        "planner+realizer evaluation start planner_samples=%s "
        "realizer_checkpoint=%s max_samples=%d",
        planner_samples,
        realizer_checkpoint,
        max_samples,
    )
    result = evaluate(
        checkpoint_path=realizer_checkpoint,
        bank_path=bank,
        planner_samples=torch.load(
            planner_samples, map_location="cpu", weights_only=False
        ),
        codebooks_path=codebooks,
        lookup_path=lookup,
        grounding_path=grounding_index,
        places_path=places,
        output_path=output,
        max_samples=max_samples,
        batch_size=batch_size,
        device=device,
        num_steps=num_steps,
        solver=solver,
        reference_split=reference_split,
        candidate_selection=candidate_selection,
        candidate_prior_path=candidate_prior,
        catalog_temperature_override=catalog_temperature,
        catalog_noise_coupling=catalog_noise_coupling,
    )
    result["planner_samples"] = str(planner_samples)
    result["composition_protocol"] = {
        "reference_plan_source": f"realizer_bank_{reference_split}_split",
        "generated_plan_source": "planner_samples",
        "time_source": "event_time_minutes_to_circadian_weekly_time_token",
        "reference_row_alignment": False,
    }
    result["summary"] = {
        arm: summary(metrics) for arm, metrics in result["evaluations"].items()
    }
    result = normalize_mapping(result)
    write_json(output, result)
    logger.info(
        "planner+realizer evaluation complete output=%s elapsed_seconds=%.1f",
        output,
        time.perf_counter() - started,
    )
    return result


__all__ = ["evaluate_planner_realizer", "realizer_max_events", "summary"]
