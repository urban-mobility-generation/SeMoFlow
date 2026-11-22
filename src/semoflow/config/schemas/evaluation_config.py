from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PlanSamplingOptions:
    batch_size: int = 1024
    temperature: float = 1.0
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
    catalog_temperature: float | None = None
    candidate_selection: str = "first"
    catalog_noise_coupling: str = "independent"

    def __post_init__(self) -> None:
        if self.catalog_noise_coupling not in {
            "independent",
            "trajectory_prefix",
        }:
            raise ValueError(
                "catalog_noise_coupling must be 'independent' or "
                "'trajectory_prefix'"
            )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Per-city sampling and composition for ``semoflow-eval evaluate``.

    Each city samples one plan per week in ``reference_split``. ``cities``
    defaults to every city in ``data.cities``.
    """

    cities: list[str] | None = None
    reference_split: str = "test"
    device: str = "cuda:0"
    seed: int = 0
    sampling: PlanSamplingOptions = field(default_factory=PlanSamplingOptions)
    composition: CompositionOptions = field(default_factory=CompositionOptions)

    def __post_init__(self) -> None:
        if self.cities is not None and not self.cities:
            raise ValueError("evaluation.cities must name at least one city")
        if self.reference_split not in {"val", "test"}:
            raise ValueError("reference_split must be 'val' or 'test'")


__all__ = ["CompositionOptions", "EvaluationConfig", "PlanSamplingOptions"]
