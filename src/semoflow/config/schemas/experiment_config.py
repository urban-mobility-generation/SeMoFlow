from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from semoflow.config.artifacts import RunLayout
from semoflow.config.schemas.data_config import DataConfig
from semoflow.config.schemas.encoder_config import EncoderConfig
from semoflow.config.schemas.evaluation_config import EvaluationConfig
from semoflow.config.schemas.planner_config import PlannerTrainingConfig
from semoflow.config.schemas.realizer_config import (
    PrefixTimeSuffixLatentFlowRealizerConfig,
)
from semoflow.config.schemas.tokenizer_config import TokenizerConfig
from semoflow.utils.serialization import normalize_value, to_yaml_string


@dataclass
class ExperimentConfig:
    """One experiment: every stage's settings, with all artifacts under
    ``run_dir`` in the fixed :class:`RunLayout`."""

    name: str = "semoflow"
    run_dir: Path = Path("outputs")
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    planner: PlannerTrainingConfig = field(
        default_factory=PlannerTrainingConfig
    )
    realizer: PrefixTimeSuffixLatentFlowRealizerConfig = field(
        default_factory=PrefixTimeSuffixLatentFlowRealizerConfig
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        self.data.validate()
        self.tokenizer.validate()
        self.encoder.text.validate()
        self.encoder.graph.validate()
        unknown = set(self.evaluation.cities or ()) - set(self.data.cities)
        if unknown:
            raise ValueError(
                f"evaluation.cities {sorted(unknown)} are not in data.cities"
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


__all__ = ["ExperimentConfig"]
