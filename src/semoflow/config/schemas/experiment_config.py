from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from semoflow.config.schemas.data_config import DataConfig
from semoflow.config.schemas.encoder_config import EncoderPipelineConfig
from semoflow.config.schemas.evaluation_config import EvaluationConfig
from semoflow.config.schemas.planner_config import PlannerTrainingConfig
from semoflow.config.schemas.preprocessing_config import PreprocessingConfig
from semoflow.config.schemas.realizer_config import (
    PrefixTimeSuffixLatentFlowRealizerConfig,
)
from semoflow.config.schemas.tokenizer_config import TokenizerConfig
from semoflow.utils.serialization import (
    normalize_value,
    to_yaml_string,
)

# Sections each phase reads; its resolved-config snapshot keeps only these.
_SNAPSHOT_SECTIONS: dict[str, tuple[str, ...]] = {
    "preprocessing": ("experiment", "data", "preprocessing", "output"),
    "encoder": (
        "experiment",
        "data",
        "encoder_pipeline",
        "tokenizer_config",
        "output",
    ),
    "tokenizer": ("experiment", "data", "tokenizer_config", "output"),
    "planner": ("experiment", "planner"),
    "realizer": ("experiment", "realizer"),
    "evaluation": ("experiment", "evaluation"),
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
    planner: PlannerTrainingConfig | None = None
    realizer: PrefixTimeSuffixLatentFlowRealizerConfig | None = None
    evaluation: EvaluationConfig | None = None
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        self.data.validate()
        self.tokenizer_config.validate()
        self.encoder_pipeline.text.validate()
        self.encoder_pipeline.graph.validate()

    def to_dict(self, phase: str | None = None) -> dict[str, Any]:
        """Every section, or only the sections ``phase`` reads."""
        config = cast(dict[str, Any], normalize_value(self))
        keep = _SNAPSHOT_SECTIONS.get(phase or "")
        if keep is None:
            return config
        return {k: v for k, v in config.items() if k in keep}

    def to_yaml(self, phase: str | None = None) -> str:
        return to_yaml_string(self.to_dict(phase))

    def save_yaml(self, path: str | Path, phase: str | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            self.to_yaml(phase or self.experiment.phase), encoding="utf-8"
        )


__all__ = [
    "ExperimentConfig",
    "ExperimentMeta",
    "OutputConfig",
]
