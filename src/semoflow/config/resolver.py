from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import yaml

from semoflow.config.overrides import from_dict
from semoflow.config.schemas.experiment_config import (
    ExperimentConfig,
)
from semoflow.logging.config_resolution import (
    log_config_resolution,
)

logger = logging.getLogger(__name__)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML at {path} did not parse to a mapping")
    sections = {f.name for f in dataclasses.fields(ExperimentConfig)}
    unknown = set(raw) - sections
    if unknown:
        raise ValueError(
            f"Unknown config sections in {path}: {sorted(unknown)}. "
            f"Valid sections: {sorted(sections)}"
        )
    return from_dict(ExperimentConfig, raw)


def resolve_experiment_config(
    config_path: Path | str | None = None,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    cfg = (
        ExperimentConfig()
        if config_path is None
        else load_experiment_config(config_path)
    )
    if overrides:
        cfg = cfg.with_overrides(overrides)
    log_config_resolution(
        logger,
        config_path=Path(config_path) if config_path is not None else None,
        overrides=overrides or [],
        experiment_name=cfg.experiment.name,
        phase=cfg.experiment.phase,
        output_dir=cfg.output.output_dir,
    )
    return cfg


__all__ = ["load_experiment_config", "resolve_experiment_config"]
