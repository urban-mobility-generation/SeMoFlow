from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OccurrencePlannerArchitecture:
    model_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    prefix_decoder: str = "hierarchical"
    mobility_router: str = "none"
    mobility_gate_prior: str = "log_distinct"
    use_tokenizer_latent: bool = False
    use_distance_prior: bool = False


@dataclass(frozen=True, slots=True)
class OccurrencePlannerTrainingOptions:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    length_weight: float = 1.0
    gap_weight: float = 1.0
    prefix_weight: float = 1.0
    label_smoothing: float = 0.0
    # cpu or cuda:N; "[0,1,...]" spawns one DDP rank per listed GPU.
    device: str = "cpu"
    recurrence_weight: float = 0.0
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
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
            ("length_weight", self.length_weight),
            ("gap_weight", self.gap_weight),
            ("prefix_weight", self.prefix_weight),
            ("recurrence_weight", self.recurrence_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} must be a finite non-negative number"
                )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not self.device:
            raise ValueError("device must not be empty")
        if self.lr_scheduler not in {"constant", "cosine"}:
            raise ValueError(
                "lr_scheduler must be 'constant' or 'cosine', "
                f"got {self.lr_scheduler!r}"
            )
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not math.isfinite(self.min_lr) or self.min_lr < 0:
            raise ValueError("min_lr must be a finite non-negative number")
        if self.min_lr > self.learning_rate:
            raise ValueError("min_lr cannot exceed learning_rate")
        if self.lr_scheduler == "constant" and self.min_lr != 0.0:
            raise ValueError("min_lr requires the cosine scheduler")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if not isinstance(self.distributed, bool):
            raise ValueError("distributed must be boolean")
        if not isinstance(self.compile_model, bool):
            raise ValueError("compile_model must be boolean")
        if not self.compile_backend:
            raise ValueError("compile_backend must not be empty")
        if self.compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError(
                "compile_mode must be one of default, reduce-overhead, "
                "max-autotune, or max-autotune-no-cudagraphs"
            )


@dataclass(frozen=True, slots=True)
class OccurrencePlannerDataOptions:
    """How ``semoflow-artifacts build`` derives the planner dataset.

    The codebook size comes from the tokenizer's codebooks.
    """

    tokenization_dir: Path
    segments_path: Path
    min_active_days: int = 5
    max_occurrences: int = 30
    context_mode: str = "none"
    # POI coordinates used to build the distance prior's prefix centroids.
    places_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PlannerTrainingConfig:
    dataset_path: Path
    output_dir: Path
    data: OccurrencePlannerDataOptions | None = None
    seed: int = 0
    # Warm-start weights from a planner with the same architecture; the
    # optimizer, schedule, and epoch counters still start fresh.
    initialize_from: Path | None = None
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
