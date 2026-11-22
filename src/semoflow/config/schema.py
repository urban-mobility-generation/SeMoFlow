"""Every stage's settings, in the order of ``configs/foursquare.yaml``."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from semoflow.config.artifacts import RunLayout
from semoflow.config.overrides import apply_override, from_dict
from semoflow.utils.serialization import normalize_value, to_yaml_string

logger = logging.getLogger(__name__)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _choice(name: str, value: str, choices: set[str]) -> None:
    _require(value in choices, f"{name} must be one of {sorted(choices)}")


@dataclass(frozen=True)
class DataConfig:
    """The Foursquare global check-in release and the cities taken from it.

    ``raw_dir`` is relative to ``root`` unless absolute.
    """

    root: Path = Path("data")
    raw_dir: Path = Path("foursquare_global_2015")
    cities: list[str] = field(
        default_factory=lambda: [
            "los_angeles",
            "washington_dc",
            "atlanta",
            "san_francisco",
            "milwaukee",
        ]
    )

    def __post_init__(self) -> None:
        _require(bool(self.cities), "data.cities must name a city")

    @property
    def source_dir(self) -> Path:
        return Path(self.root).expanduser() / Path(self.raw_dir).expanduser()


@dataclass(frozen=True)
class TextEncoderConfig:
    model: str = "sentence-t5-xxl"
    device: str = "cuda:0"
    batch_size: int = 512
    precision: str = "float16"
    include_address_in_template: bool = True
    include_name_in_template: bool = True

    def __post_init__(self) -> None:
        _choice(
            "encoder.text.precision",
            self.precision,
            {"float32", "float16", "bfloat16"},
        )


@dataclass(frozen=True)
class GraphEncoderConfig:
    device: str = "cuda:0"
    hidden_dim: int = 256
    out_dim: int = 256
    num_layers: int = 2
    epochs: int = 5
    batch_size: int = 1024
    lr: float = 0.001
    num_workers: int = 0
    num_neighbors: list[int] | dict[str, list[int]] = field(
        default_factory=lambda: [10, 10]
    )
    hash_train_fraction: float = 0.70
    seed: int = 42
    temperature: float = 0.07
    max_negatives: int = 2048
    max_queries: int = 65536
    aggr: str = "sum"
    layer_norm: bool = False
    residual: bool = False
    dropout: float = 0.0
    id_embedding_dim: int = 0
    id_embedding_node_types: list[str] | None = None
    variance_weight: float = 0.0
    covariance_weight: float = 0.0
    variance_target: float = 1.0

    def __post_init__(self) -> None:
        hops = self.num_neighbors
        entries = list(hops.values()) if isinstance(hops, dict) else [hops]
        _require(
            not isinstance(hops, dict) or "default" in hops,
            "encoder.graph.num_neighbors needs a 'default' relation",
        )
        _require(
            len({len(entry) for entry in entries}) == 1
            and all(entry and min(entry) >= 1 for entry in entries),
            "encoder.graph.num_neighbors entries must be non-empty lists "
            "of positive counts with the same number of hops",
        )
        _require(self.temperature > 0, "encoder.graph.temperature must be > 0")
        _choice("encoder.graph.aggr", self.aggr, {"sum", "mean", "max", "min"})
        _require(
            0.0 <= self.dropout < 1.0,
            "encoder.graph.dropout must be in [0, 1)",
        )
        types = self.id_embedding_node_types
        _require(
            types is None or (bool(types) and len(set(types)) == len(types)),
            "encoder.graph.id_embedding_node_types must be null or "
            "distinct node types",
        )


@dataclass(frozen=True)
class EncoderConfig:
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    graph: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)


@dataclass(frozen=True)
class TokenizerConfig:
    n_codebooks: int = 6
    codebook_size: int = 32
    hidden_dim: int = 512
    latent_dim: int = 256
    n_encoder_layers: int = 4
    epochs: int = 50
    batch_size: int = 1024
    lr: float = 0.001
    beta: float = 3.0
    sk_epsilons: list[float] | None = None
    device: str = "cuda:0"
    num_workers: int = 0
    seed: int = 42
    pca_text_dim: int = 32
    pca_graph_dim: int = 32
    geo_relative_dim: int = 6
    geo_scale: float = 5.0
    text_graph_scale: float = 2.5

    def __post_init__(self) -> None:
        _require(
            self.n_encoder_layers >= 1,
            "tokenizer.n_encoder_layers must be >= 1",
        )
        _require(
            self.geo_relative_dim >= 2
            and (self.geo_relative_dim - 2) % 4 == 0,
            "tokenizer.geo_relative_dim must be 2 + 4F for F sin/cos "
            f"wavelength pairs, got {self.geo_relative_dim}",
        )


@dataclass(frozen=True)
class PlannerDataConfig:
    """How ``semoflow planner-data`` derives the planner dataset from the
    tokenizer's visits; the codebook size comes from its codebooks."""

    min_active_days: int = 5
    max_occurrences: int = 30
    city_conditioning: bool = False


@dataclass(frozen=True)
class PlannerArchitectureConfig:
    model_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    use_distance_prior: bool = False

    def __post_init__(self) -> None:
        head_dim, rest = divmod(self.model_dim, self.n_heads)
        _require(
            rest == 0 and head_dim % 2 == 0,
            "planner.architecture.model_dim must split into n_heads "
            "even-sized heads",
        )


@dataclass(frozen=True)
class PlannerTrainingConfig:
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
    compile_model: bool = False

    def __post_init__(self) -> None:
        _choice(
            "planner.training.lr_scheduler",
            self.lr_scheduler,
            {"constant", "cosine"},
        )
        _require(
            self.lr_scheduler == "cosine" or self.min_lr == 0.0,
            "planner.training.min_lr requires the cosine scheduler",
        )


@dataclass(frozen=True)
class PlannerConfig:
    data: PlannerDataConfig = field(default_factory=PlannerDataConfig)
    seed: int = 0
    architecture: PlannerArchitectureConfig = field(
        default_factory=PlannerArchitectureConfig
    )
    training: PlannerTrainingConfig = field(
        default_factory=PlannerTrainingConfig
    )


@dataclass(frozen=True)
class RealizerConfig:
    """Hard-prefix latent-flow realizer over whitened suffix tails, decoded
    by the calibrated legal-catalog posterior."""

    max_events: int = 32
    prefix_depth: int = 3
    city_vocab: int = 0
    model_dim: int = 256
    condition_depth: int = 4
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    catalog_loss_weight: float = 0.1
    catalog_temperature: float = 1.0
    catalog_candidate_count: int = 64
    endpoint_loss_weight: float = 0.0
    population_loss_weight: float = 0.0
    catalog_prior_weight: float = 0.25
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant"
    warmup_steps: int = 0
    min_lr: float = 0.0
    early_stopping_patience: int | None = None
    tail_whitening_chunk_size: int = 262144
    num_workers: int = 0
    device: str = "cuda:0"
    seed: int = 0
    compile: bool = False
    bf16: bool = False

    def __post_init__(self) -> None:
        _require(
            self.epochs > 0 and self.batch_size > 0,
            "realizer.epochs and batch_size must be positive",
        )
        _choice(
            "realizer.lr_scheduler", self.lr_scheduler, {"constant", "cosine"}
        )
        _require(
            0.0 <= self.min_lr <= self.learning_rate
            and (self.lr_scheduler == "cosine" or self.min_lr == 0.0),
            "realizer.min_lr must lie in [0, learning_rate] and requires "
            "the cosine scheduler",
        )
        _require(
            self.catalog_loss_weight > 0.0,
            "realizer.catalog_loss_weight must be positive; it trains the "
            "calibrated catalog decoder",
        )


@dataclass(frozen=True)
class SamplingConfig:
    batch_size: int = 1024
    temperature: float = 1.0
    route_temperature: float | None = None

    def __post_init__(self) -> None:
        _require(
            self.temperature > 0
            and (self.route_temperature is None or self.route_temperature > 0),
            "sampling temperatures must be positive",
        )


@dataclass(frozen=True)
class CompositionConfig:
    batch_size: int = 64
    solver: str = "heun"
    num_steps: int = 32
    catalog_temperature: float | None = None
    candidate_selection: str = "first"
    catalog_noise_coupling: str = "independent"

    def __post_init__(self) -> None:
        _choice(
            "evaluation.composition.catalog_noise_coupling",
            self.catalog_noise_coupling,
            {"independent", "trajectory_prefix"},
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """Per-city sampling and composition for ``semoflow evaluate``.

    Each city samples one plan per week in ``reference_split``. ``cities``
    defaults to every city in ``data.cities``.
    """

    cities: list[str] | None = None
    reference_split: str = "test"
    device: str = "cuda:0"
    seed: int = 0
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    composition: CompositionConfig = field(default_factory=CompositionConfig)

    def __post_init__(self) -> None:
        _require(
            self.cities is None or bool(self.cities),
            "evaluation.cities must name a city",
        )
        _choice(
            "evaluation.reference_split", self.reference_split, {"val", "test"}
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """One experiment: every stage's settings, with all artifacts under
    ``run_dir`` in the fixed :class:`RunLayout`."""

    name: str = "semoflow"
    run_dir: Path = Path("outputs")
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    realizer: RealizerConfig = field(default_factory=RealizerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        unknown = set(self.evaluation.cities or ()) - set(self.data.cities)
        _require(
            not unknown,
            f"evaluation.cities {sorted(unknown)} are not in data.cities",
        )

    @property
    def layout(self) -> RunLayout:
        return RunLayout(Path(self.run_dir))

    @property
    def evaluation_cities(self) -> list[str]:
        return list(self.evaluation.cities or self.data.cities)

    def to_yaml(self) -> str:
        return to_yaml_string(cast(dict[str, Any], normalize_value(self)))

    def save_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_yaml(), encoding="utf-8")


def load_config(
    path: Path | str | None = None, overrides: Sequence[str] = ()
) -> ExperimentConfig:
    """Load a YAML config (or defaults), apply ``--set`` overrides, validate."""
    raw: object = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML at {path} did not parse to a mapping")
    for override in overrides:
        apply_override(raw, override, ExperimentConfig)
    cfg = from_dict(ExperimentConfig, raw)
    logger.info(
        "config  path=%s  overrides=%s", path or "defaults", list(overrides)
    )
    return cfg


__all__ = [
    "CompositionConfig",
    "DataConfig",
    "EncoderConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "GraphEncoderConfig",
    "PlannerArchitectureConfig",
    "PlannerConfig",
    "PlannerDataConfig",
    "PlannerTrainingConfig",
    "RealizerConfig",
    "SamplingConfig",
    "TextEncoderConfig",
    "TokenizerConfig",
    "load_config",
]
