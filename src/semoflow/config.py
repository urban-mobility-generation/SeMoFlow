from __future__ import annotations

import dataclasses
import logging
import types
import typing
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

from semoflow.utils import to_yaml_string

logger = logging.getLogger(__name__)
PAD_ID: Final[int] = 0
CITY_ID: Final[dict[str, int]] = {
    "boston": 0,
    "chicago": 1,
    "dallas": 2,
    "new_york_city": 3,
    "pittsburgh": 4,
    "los_angeles": 7,
    "washington_dc": 8,
    "atlanta": 9,
    "san_francisco": 10,
    "milwaukee": 11,
}


@dataclass(frozen=True)
class PreprocessingArtifacts:
    run_dir: Path

    @property
    def processed_visits(self) -> Path:
        return self.run_dir / "processed_visits.parquet"

    @property
    def places_index(self) -> Path:
        return self.run_dir / "places_index.parquet"


@dataclass(frozen=True)
class TokenizerArtifacts:
    run_dir: Path

    @property
    def augmented_processed_visits(self) -> Path:
        return self.run_dir / "augmented_processed_visits.parquet"

    @property
    def placekey_lookup(self) -> Path:
        return self.run_dir / "placekey_lookup.parquet"

    @property
    def tokenizer_model(self) -> Path:
        return self.run_dir / "tokenizer_model.pt"

    @property
    def tokenizer_codebooks(self) -> Path:
        return self.run_dir / "tokenizer_codebooks.pt"

    @property
    def grounding_index(self) -> Path:
        return self.run_dir / "grounding_index.pkl"


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @property
    def preprocessing(self) -> PreprocessingArtifacts:
        return PreprocessingArtifacts(self.root / "preprocessing")

    @property
    def encoder_dir(self) -> Path:
        return self.root / "encoder"

    @property
    def text_embeddings(self) -> Path:
        return self.encoder_dir / "text" / "text_embeddings.npy"

    @property
    def graph_dir(self) -> Path:
        return self.encoder_dir / "graph"

    @property
    def graph_embeddings(self) -> Path:
        return self.graph_dir / "poi_graph_emb.pt"

    @property
    def tokenizer(self) -> TokenizerArtifacts:
        return TokenizerArtifacts(self.root / "tokenization")

    @property
    def segments(self) -> Path:
        return self.root / "segments.parquet"

    @property
    def planner_dataset(self) -> Path:
        return self.root / "planner_dataset.pt"

    @property
    def planner_dir(self) -> Path:
        return self.root / "planner"

    @property
    def bank(self) -> Path:
        return self.root / "bank.pt"

    @property
    def realizer_dir(self) -> Path:
        return self.root / "realizer"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"


_NULL = {"", "null", "none", "~"}
_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def _non_null_types(annotation: Any) -> tuple[list[Any], bool]:
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        members = typing.get_args(annotation)
        kinds = [kind for kind in members if kind is not type(None)]
        return kinds, len(kinds) < len(members)
    return [annotation], False


def _dataclass_type(annotation: Any) -> type | None:
    kinds, _ = _non_null_types(annotation)
    if len(kinds) == 1 and dataclasses.is_dataclass(kinds[0]):
        return kinds[0]
    return None


def _coerce(text: str, annotation: Any) -> object:
    kinds, optional = _non_null_types(annotation)
    if optional and text.strip().lower() in _NULL:
        return None
    kind = kinds[0] if len(kinds) == 1 else None
    if kind is str:
        return text
    if kind is Path:
        return Path(text)
    if kind is bool:
        if text.strip().lower() in _TRUE | _FALSE:
            return text.strip().lower() in _TRUE
        raise ValueError(f"expected a boolean, got {text!r}")
    if kind in (int, float):
        return kind(text)
    if typing.get_origin(kind) in (list, tuple) and not text.startswith("["):
        text = f"[{text}]"
    return yaml.safe_load(text)


def apply_override(data: dict[str, Any], override: str, cls: type) -> None:
    key, sep, raw_value = override.partition("=")
    parts = key.strip().split(".")
    if not sep or not key.strip():
        raise ValueError(f"Override must be field=value, got {override!r}")
    target, owner = data, cls
    for depth, part in enumerate(parts):
        types_by_name = typing.get_type_hints(owner)
        if part not in types_by_name:
            where = ".".join(parts[:depth]) or "config"
            raise ValueError(
                f"Unknown field {part!r} in {where!r}. "
                f"Valid fields: {sorted(types_by_name)}"
            )
        if depth == len(parts) - 1:
            try:
                target[part] = _coerce(raw_value.strip(), types_by_name[part])
            except ValueError as exc:
                raise ValueError(f"Cannot set {key.strip()}: {exc}") from exc
            return
        section = _dataclass_type(types_by_name[part])
        if section is None:
            raise ValueError(f"{'.'.join(parts[: depth + 1])!r} is a field")
        if not isinstance(target.get(part), dict):
            target[part] = {}
        target, owner = target[part], section


def from_dict[T](cls: type[T], data: dict[str, Any]) -> T:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a dataclass")
    types_by_name = typing.get_type_hints(cls)
    unknown = set(data) - set(types_by_name)
    if unknown:
        raise ValueError(
            f"Unknown keys for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(types_by_name)}"
        )
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        annotation = types_by_name[name]
        section = _dataclass_type(annotation)
        if section is not None and isinstance(value, dict):
            value = from_dict(section, value)
        elif isinstance(value, str):
            try:
                value = _coerce(value, annotation)
            except ValueError as exc:
                raise ValueError(f"{cls.__name__}.{name}: {exc}") from exc
        kwargs[name] = value
    return cls(**kwargs)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _choice(name: str, value: str, choices: set[str]) -> None:
    _require(value in choices, f"{name} must be one of {sorted(choices)}")


@dataclass(frozen=True)
class DataConfig:
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
    num_neighbors: list[int] = field(default_factory=lambda: [10, 10])
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
        _require(
            bool(self.num_neighbors) and min(self.num_neighbors) >= 1,
            "encoder.graph.num_neighbors must be positive per-hop counts",
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
    early_stopping_patience: int | None = None
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
            "evaluation.composition.solver", self.solver, {"euler", "heun"}
        )
        _choice(
            "evaluation.composition.candidate_selection",
            self.candidate_selection,
            {"first", "train_prior_argmax", "train_prior_sample"},
        )
        _choice(
            "evaluation.composition.catalog_noise_coupling",
            self.catalog_noise_coupling,
            {"independent", "trajectory_prefix"},
        )


@dataclass(frozen=True)
class EvaluationConfig:
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
        return to_yaml_string(self)

    def save_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_yaml(), encoding="utf-8")


def load_config(
    path: Path | str | None = None, overrides: Sequence[str] = ()
) -> ExperimentConfig:
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
    "CITY_ID",
    "PAD_ID",
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
    "PreprocessingArtifacts",
    "RealizerConfig",
    "RunLayout",
    "SamplingConfig",
    "TextEncoderConfig",
    "TokenizerArtifacts",
    "TokenizerConfig",
    "load_config",
]
