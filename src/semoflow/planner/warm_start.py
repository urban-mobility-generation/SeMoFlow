from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from semoflow.planner.artifacts import (
    load_occurrence_planner_checkpoint,
)
from semoflow.planner.model import OccurrenceARPlanner


@dataclass(frozen=True, slots=True)
class PlannerWarmStart:
    checkpoint: Path

    def __post_init__(self) -> None:
        if not Path(self.checkpoint).is_file():
            raise ValueError(
                f"warm-start planner checkpoint not found: {self.checkpoint}"
            )


@dataclass(frozen=True, slots=True)
class PlannerWarmStartReport:
    checkpoint: str
    source_epoch: int
    source_validation_metric: float
    source_city_ids: tuple[int, ...]
    target_city_ids: tuple[int, ...]
    inherited_city_ids: tuple[int, ...]
    zero_initialized_city_ids: tuple[int, ...]
    source_legal_prefix_count: int
    target_legal_prefix_count: int
    catalog_changed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint": self.checkpoint,
            "source_epoch": self.source_epoch,
            "source_validation_metric": self.source_validation_metric,
            "source_city_ids": list(self.source_city_ids),
            "target_city_ids": list(self.target_city_ids),
            "inherited_city_ids": list(self.inherited_city_ids),
            "zero_initialized_city_ids": list(self.zero_initialized_city_ids),
            "source_legal_prefix_count": self.source_legal_prefix_count,
            "target_legal_prefix_count": self.target_legal_prefix_count,
            "catalog_changed": self.catalog_changed,
        }


def _city_embedding_for_target(
    *,
    source_state: dict[str, torch.Tensor],
    source_city_ids: list[int],
    target_city_ids: list[int],
    target_weight: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    source_weight = source_state["city_embedding.weight"]
    if source_weight.shape[1:] != target_weight.shape[1:]:
        raise ValueError(
            "warm-start city embedding width does not match the target"
        )
    source_row = {city_id: row for row, city_id in enumerate(source_city_ids)}
    transferred = torch.zeros_like(target_weight)
    inherited: list[int] = []
    zeroed: list[int] = []
    for row, city_id in enumerate(target_city_ids):
        if city_id in source_row:
            transferred[row] = source_weight[source_row[city_id]]
            inherited.append(city_id)
        else:
            zeroed.append(city_id)
    return transferred, tuple(inherited), tuple(zeroed)


def apply_planner_warm_start(
    model: OccurrenceARPlanner,
    warm_start: PlannerWarmStart,
) -> PlannerWarmStartReport:
    checkpoint_path = Path(warm_start.checkpoint)
    source = load_occurrence_planner_checkpoint(checkpoint_path)
    if source.config != model.config:
        raise ValueError(
            "warm-start planner architecture is incompatible with the target"
        )
    state = dict(source.model.state_dict())
    source_city_ids = [int(value) for value in source.catalog_city_ids]
    target_city_ids = [int(value) for value in model.catalog_city_ids]

    inherited: tuple[int, ...] = ()
    zeroed: tuple[int, ...] = ()
    if model.config.city_conditioning:
        assert model.city_embedding is not None
        transferred, inherited, zeroed = _city_embedding_for_target(
            source_state=state,
            source_city_ids=source_city_ids,
            target_city_ids=target_city_ids,
            target_weight=model.city_embedding.weight,
        )
        state["city_embedding.weight"] = transferred

    source_metadata = state["_extra_state"]
    target_metadata = model.get_extra_state()
    catalog_changed = (
        source_metadata["catalog_fingerprint"]
        != target_metadata["catalog_fingerprint"]
        or source_metadata["catalog_city_ids"]
        != target_metadata["catalog_city_ids"]
        or source_metadata["city_legality_fingerprint"]
        != target_metadata["city_legality_fingerprint"]
    )
    model.load_neural_state_dict(
        state, allow_catalog_mismatch=True, strict=True
    )
    return PlannerWarmStartReport(
        checkpoint=str(checkpoint_path),
        source_epoch=source.epoch,
        source_validation_metric=source.validation_metric,
        source_city_ids=tuple(source_city_ids),
        target_city_ids=tuple(target_city_ids),
        inherited_city_ids=inherited,
        zero_initialized_city_ids=zeroed,
        source_legal_prefix_count=int(source.legal_prefixes.shape[0]),
        target_legal_prefix_count=model.legality.vocabulary_size,
        catalog_changed=catalog_changed,
    )


__all__ = [
    "PlannerWarmStart",
    "PlannerWarmStartReport",
    "apply_planner_warm_start",
]
