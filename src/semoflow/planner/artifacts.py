from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from semoflow.config import (
    PlannerTrainingConfig,
)
from semoflow.planner.model import OccurrenceARPlanner, OccurrencePlannerConfig
from semoflow.utils import load_artifact, save_torch_atomic

CHECKPOINT_ARTIFACT_TYPE = "occurrence_planner_checkpoint"
SAMPLES_ARTIFACT_TYPE = "occurrence_planner_samples"


def _cpu(
    tensor: torch.Tensor | None, dtype: torch.dtype
) -> torch.Tensor | None:
    return None if tensor is None else tensor.detach().to("cpu", dtype).clone()


def save_occurrence_planner_checkpoint(
    path: Path,
    *,
    config: OccurrencePlannerConfig,
    legal_prefixes: torch.Tensor,
    catalog_city_ids: torch.Tensor,
    city_legal_prefix_mask: torch.Tensor,
    model_state: dict[str, Any],
    epoch: int,
    validation_metric: float,
    seed: int,
    options: PlannerTrainingConfig,
    legal_prefix_centroids: torch.Tensor | None = None,
    legal_prefix_centroid_available: torch.Tensor | None = None,
    force: bool = False,
) -> None:
    payload = {
        "artifact_type": CHECKPOINT_ARTIFACT_TYPE,
        "config": asdict(config),
        "legal_prefixes": _cpu(legal_prefixes, torch.long),
        "catalog_city_ids": _cpu(catalog_city_ids, torch.long),
        "city_legal_prefix_mask": _cpu(city_legal_prefix_mask, torch.bool),
        "legal_prefix_centroids": _cpu(legal_prefix_centroids, torch.float32),
        "legal_prefix_centroid_available": _cpu(
            legal_prefix_centroid_available, torch.bool
        ),
        "model_state": model_state,
        "epoch": int(epoch),
        "validation_metric": float(validation_metric),
        "seed": int(seed),
        "options": asdict(options),
    }
    save_torch_atomic(path, payload, force=force)


_RETIRED_CONFIG = {
    "context_dim": 0,
    "context_schema": (),
    "prefix_decoder": "hierarchical",
    "mobility_router": "explore_retrieve",
    "use_tokenizer_latent": False,
}


def load_occurrence_planner_checkpoint(
    path: Path,
) -> tuple[OccurrenceARPlanner, dict[str, Any]]:
    payload = load_artifact(path, CHECKPOINT_ARTIFACT_TYPE)
    config = dict(payload["config"])
    for key, supported in _RETIRED_CONFIG.items():
        if key in config and config.pop(key) != supported:
            raise ValueError(
                f"unsupported planner checkpoint: {key} != {supported!r}"
            )
    model = OccurrenceARPlanner(
        OccurrencePlannerConfig(**config),
        payload["legal_prefixes"],
        catalog_city_ids=payload["catalog_city_ids"],
        city_legal_prefix_mask=payload["city_legal_prefix_mask"],
        legal_prefix_centroids=payload["legal_prefix_centroids"],
        legal_prefix_centroid_available=payload[
            "legal_prefix_centroid_available"
        ],
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def save_occurrence_planner_samples(
    path: Path,
    *,
    prefix_codes: torch.Tensor,
    gap_tokens: torch.Tensor,
    event_time_minutes: torch.Tensor,
    lengths: torch.Tensor,
    city_ids: torch.Tensor,
    seed: int,
    temperature: float,
    route_temperature: float,
    force: bool = False,
) -> None:
    payload = {
        "artifact_type": SAMPLES_ARTIFACT_TYPE,
        "prefix_codes": prefix_codes.long(),
        "gap_tokens": gap_tokens.long(),
        "event_time_minutes": event_time_minutes.float(),
        "lengths": lengths.long(),
        "city_ids": city_ids.long(),
        "seed": int(seed),
        "temperature": float(temperature),
        "route_temperature": float(route_temperature),
    }
    save_torch_atomic(path, payload, force=force)


__all__ = [
    "CHECKPOINT_ARTIFACT_TYPE",
    "SAMPLES_ARTIFACT_TYPE",
    "load_occurrence_planner_checkpoint",
    "save_occurrence_planner_checkpoint",
    "save_occurrence_planner_samples",
]
