from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PlanSamplingOptions:
    batch_size: int = 1024
    temperature: float = 1.0
    # None samples the Explore/Retrieve gate at ``temperature``.
    route_temperature: float | None = None

    def __post_init__(self) -> None:
        if self.temperature <= 0 or (
            self.route_temperature is not None and self.route_temperature <= 0
        ):
            raise ValueError("sampling temperatures must be positive")


@dataclass(frozen=True, slots=True)
class CompositionOptions:
    batch_size: int = 64
    solver: str = "heun"
    num_steps: int = 32
    # None keeps the realizer's calibrated catalog temperature.
    catalog_temperature: float | None = None
    candidate_selection: str = "first"
    # Score only the comparison metrics and POI coverage.
    brief_metrics: bool = False


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Per-city sampling and composition for ``semoflow-eval evaluate``.

    Each city samples one plan per week in ``reference_split`` and writes
    its outputs to ``output_dir/<city name>``.
    """

    planner_checkpoint: Path
    realizer_checkpoint: Path
    tokenization_dir: Path
    segments_path: Path
    places_path: Path
    output_dir: Path
    # City name -> city_id, in evaluation order.
    cities: dict[str, int]
    reference_split: str = "test"
    device: str = "cuda:0"
    seed: int = 0
    sampling: PlanSamplingOptions = field(default_factory=PlanSamplingOptions)
    composition: CompositionOptions = field(default_factory=CompositionOptions)

    def __post_init__(self) -> None:
        if not self.cities:
            raise ValueError("evaluation.cities must name at least one city")
        if self.reference_split not in {"val", "test"}:
            raise ValueError("reference_split must be 'val' or 'test'")


__all__ = ["CompositionOptions", "EvaluationConfig", "PlanSamplingOptions"]
