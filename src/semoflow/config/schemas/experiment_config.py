from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semoflow.config.overrides import apply_overrides
from semoflow.config.schemas.data_config import DataConfig
from semoflow.config.schemas.encoder_config import (
    EncoderPipelineConfig,
)
from semoflow.config.schemas.preprocessing_config import (
    PreprocessingConfig,
)
from semoflow.config.schemas.tokenizer_config import (
    TokenizerConfig,
)
from semoflow.utils.serialization import (
    to_serialisable_dict,
    to_yaml_string,
)

_SNAPSHOT_COMMON = ("experiment", "data", "runtime", "output")
_SNAPSHOT_SECTIONS: dict[str, tuple[str, ...]] = {
    "preprocessing": ("preprocessing",),
    "encoder": ("encoder_pipeline", "tokenizer_config", "paths"),
    "tokenizer": ("tokenizer_config",),
}


@dataclass
class ExperimentMeta:
    name: str = "unnamed"
    phase: str = "tokenizer"


@dataclass
class OutputConfig:
    output_dir: Path = field(
        default_factory=lambda: Path("outputs/experiment")
    )
    config_filename: str = "config_resolved.yaml"


@dataclass
class PathsConfig:
    semantic_id_lookup_path: str | None = None
    semantic_id_grounding_path: str | None = None
    preprocessed_processed_visits_path: str | None = None


@dataclass
class RuntimeConfig:
    device: str = "auto"
    seed: int = 42
    print_full_config: bool = False

    def validate(self) -> None:
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("runtime.device must be a non-empty string")


@dataclass
class ExperimentConfig:
    experiment: ExperimentMeta = field(default_factory=ExperimentMeta)
    preprocessing: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )
    encoder_pipeline: EncoderPipelineConfig = field(
        default_factory=EncoderPipelineConfig
    )
    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.data.validate()
        self.tokenizer_config.validate()
        self.runtime.validate()
        self.encoder_pipeline.text.validate()
        self.encoder_pipeline.graph.validate()

    def with_overrides(self, overrides: list[str]) -> ExperimentConfig:
        updated: ExperimentConfig = apply_overrides(self, overrides)
        updated.validate()
        return updated

    def to_dict(self) -> dict[str, Any]:
        return to_serialisable_dict(self)

    def to_yaml(self) -> str:
        return to_yaml_string(self)

    def save_yaml(self, path: str | Path, phase: str | None = None) -> None:
        phase = phase or self.experiment.phase
        config = self.to_dict()
        if phase in _SNAPSHOT_SECTIONS:
            keep = _SNAPSHOT_COMMON + _SNAPSHOT_SECTIONS[phase]
            config = {k: v for k, v in config.items() if k in keep}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(to_yaml_string(config), encoding="utf-8")


__all__ = [
    "ExperimentConfig",
    "ExperimentMeta",
    "OutputConfig",
    "PathsConfig",
    "RuntimeConfig",
]
