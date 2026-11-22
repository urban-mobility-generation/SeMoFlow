from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from semoflow.config.schema import (
    PlannerTrainingConfig,
)
from semoflow.planner.config import OccurrencePlannerConfig
from semoflow.planner.model import OccurrenceARPlanner

DATASET_ARTIFACT_TYPE = "occurrence_planner_dataset"

CHECKPOINT_ARTIFACT_TYPE = "occurrence_planner_checkpoint"

SAMPLES_ARTIFACT_TYPE = "occurrence_planner_samples"


@dataclass(frozen=True, slots=True)
class OccurrencePlannerDataConfig:
    codebook_size: int
    gap_vocab: int
    prefix_depth: int
    max_occurrences: int
    city_conditioning: bool
    gap_codec_version: str
    gap_bucket_edges_minutes: tuple[float, ...]
    gap_duration_minutes: tuple[float, ...]
    observation_window_minutes: float


@dataclass(frozen=True, slots=True)
class OccurrencePlannerSplit:
    prefix_codes: torch.Tensor
    gap_tokens: torch.Tensor
    lengths: torch.Tensor
    city_ids: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.lengths.shape[0])


@dataclass(frozen=True, slots=True)
class OccurrencePlannerDataset:
    config: OccurrencePlannerDataConfig
    legal_prefixes: torch.Tensor
    catalog_city_ids: torch.Tensor
    city_legal_prefix_mask: torch.Tensor
    train: OccurrencePlannerSplit
    validation: OccurrencePlannerSplit
    source_path: Path
    legal_prefix_centroids: torch.Tensor | None = None
    legal_prefix_centroid_available: torch.Tensor | None = None


def _publish_atomic(path: Path, write: Any, *, force: bool) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing artifact: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_torch_atomic(
    path: Path, payload: object, *, force: bool = False
) -> None:
    _publish_atomic(path, lambda tmp: torch.save(payload, tmp), force=force)


def _load_artifact(path: Path, artifact_type: str) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("artifact_type") != artifact_type:
        raise ValueError(f"{path} is not an {artifact_type} artifact")
    return payload


def load_occurrence_planner_dataset(path: Path) -> OccurrencePlannerDataset:
    payload = _load_artifact(path, DATASET_ARTIFACT_TYPE)
    return OccurrencePlannerDataset(
        config=OccurrencePlannerDataConfig(**payload["data_config"]),
        legal_prefixes=payload["legal_prefixes"],
        legal_prefix_centroids=payload["legal_prefix_centroids"],
        legal_prefix_centroid_available=payload[
            "legal_prefix_centroid_available"
        ],
        catalog_city_ids=payload["catalog_city_ids"],
        city_legal_prefix_mask=payload["city_legal_prefix_mask"],
        train=OccurrencePlannerSplit(**payload["train"]),
        validation=OccurrencePlannerSplit(**payload["validation"]),
        source_path=Path(path).resolve(),
    )


@dataclass(frozen=True, slots=True)
class OccurrencePlannerCheckpoint:
    config: OccurrencePlannerConfig
    legal_prefixes: torch.Tensor
    catalog_city_ids: torch.Tensor
    city_legal_prefix_mask: torch.Tensor
    model: OccurrenceARPlanner
    epoch: int
    validation_metric: float
    legal_prefix_centroids: torch.Tensor | None
    legal_prefix_centroid_available: torch.Tensor | None


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
        "legal_prefixes": legal_prefixes.detach()
        .to("cpu", torch.long)
        .clone(),
        "catalog_city_ids": catalog_city_ids.detach()
        .to("cpu", torch.long)
        .clone(),
        "city_legal_prefix_mask": city_legal_prefix_mask.detach()
        .to("cpu", torch.bool)
        .clone(),
        "legal_prefix_centroids": (
            None
            if legal_prefix_centroids is None
            else legal_prefix_centroids.detach()
            .to("cpu", torch.float32)
            .clone()
        ),
        "legal_prefix_centroid_available": (
            None
            if legal_prefix_centroid_available is None
            else legal_prefix_centroid_available.detach()
            .to("cpu", torch.bool)
            .clone()
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


def _checkpoint_config(raw: dict[str, Any]) -> OccurrencePlannerConfig:
    config = dict(raw)
    for key, supported in _RETIRED_CONFIG.items():
        if key in config and config.pop(key) != supported:
            raise ValueError(
                f"unsupported planner checkpoint: {key} != {supported!r}"
            )
    return OccurrencePlannerConfig(**config)


def load_occurrence_planner_checkpoint(
    path: Path,
) -> OccurrencePlannerCheckpoint:
    payload = _load_artifact(path, CHECKPOINT_ARTIFACT_TYPE)
    config = _checkpoint_config(payload["config"])
    centroids = payload["legal_prefix_centroids"]
    centroid_available = payload["legal_prefix_centroid_available"]
    model = OccurrenceARPlanner(
        config,
        payload["legal_prefixes"],
        catalog_city_ids=payload["catalog_city_ids"],
        city_legal_prefix_mask=payload["city_legal_prefix_mask"],
        legal_prefix_centroids=centroids,
        legal_prefix_centroid_available=centroid_available,
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return OccurrencePlannerCheckpoint(
        config=config,
        legal_prefixes=payload["legal_prefixes"],
        catalog_city_ids=payload["catalog_city_ids"],
        city_legal_prefix_mask=payload["city_legal_prefix_mask"],
        model=model,
        epoch=payload["epoch"],
        validation_metric=payload["validation_metric"],
        legal_prefix_centroids=centroids,
        legal_prefix_centroid_available=centroid_available,
    )


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
    "DATASET_ARTIFACT_TYPE",
    "SAMPLES_ARTIFACT_TYPE",
    "OccurrencePlannerCheckpoint",
    "OccurrencePlannerDataConfig",
    "OccurrencePlannerDataset",
    "OccurrencePlannerSplit",
    "load_occurrence_planner_checkpoint",
    "load_occurrence_planner_dataset",
    "save_occurrence_planner_checkpoint",
    "save_occurrence_planner_samples",
    "save_torch_atomic",
]
