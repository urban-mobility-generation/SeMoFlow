from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class OccurrencePlannerArchitecture:
    model_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    mobility_gate_prior: str = "log_distinct"
    use_distance_prior: bool = False

    def __post_init__(self) -> None:
        if (
            self.model_dim % self.n_heads
            or (self.model_dim // self.n_heads) % 2
        ):
            raise ValueError(
                "model_dim must split into n_heads even-sized heads"
            )
        if self.mobility_gate_prior not in {"log_distinct", "epr"}:
            raise ValueError(
                "mobility_gate_prior must be 'log_distinct' or 'epr'"
            )


@dataclass(frozen=True, slots=True)
class OccurrencePlannerTrainingOptions:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    device: str = "cuda:0"
    lr_scheduler: str = "constant"
    warmup_steps: int = 0
    min_lr: float = 0.0
    num_workers: int = 0
    early_stopping_patience: int = 0
    distributed: bool = False
    compile_model: bool = False
    compile_backend: str = "inductor"
    compile_mode: str = "default"

    def __post_init__(self) -> None:
        if self.lr_scheduler not in {"constant", "cosine"}:
            raise ValueError("lr_scheduler must be 'constant' or 'cosine'")
        if self.lr_scheduler == "constant" and self.min_lr != 0.0:
            raise ValueError("min_lr requires the cosine scheduler")


@dataclass(frozen=True, slots=True)
class OccurrencePlannerDataOptions:
    """How ``semoflow planner-data`` derives the planner dataset from
    the tokenizer's visits; the codebook size comes from its codebooks."""

    min_active_days: int = 5
    max_occurrences: int = 30
    city_conditioning: bool = False


@dataclass(frozen=True, slots=True)
class PlannerTrainingConfig:
    data: OccurrencePlannerDataOptions = field(
        default_factory=OccurrencePlannerDataOptions
    )
    seed: int = 0
    architecture: OccurrencePlannerArchitecture = field(
        default_factory=OccurrencePlannerArchitecture
    )
    training: OccurrencePlannerTrainingOptions = field(
        default_factory=OccurrencePlannerTrainingOptions
    )


__all__ = [
    "OccurrencePlannerArchitecture",
    "OccurrencePlannerDataOptions",
    "OccurrencePlannerTrainingOptions",
    "PlannerTrainingConfig",
]
