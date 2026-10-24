from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import yaml

from semoflow.config.overrides import apply_override, from_dict
from semoflow.config.schemas.experiment_config import ExperimentConfig
from semoflow.logging.config_resolution import log_config_resolution

logger = logging.getLogger(__name__)


def resolve_experiment_config(
    config_path: Path | str | None = None,
    overrides: Sequence[str] = (),
) -> ExperimentConfig:
    """Load a YAML config (or defaults), apply ``--set`` overrides, validate."""
    raw: object = {}
    if config_path is not None:
        text = Path(config_path).read_text(encoding="utf-8")
        raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML at {config_path} did not parse to a mapping")
    for override in overrides:
        apply_override(raw, override, ExperimentConfig)
    cfg = from_dict(ExperimentConfig, raw)
    log_config_resolution(
        logger,
        config_path=Path(config_path) if config_path is not None else None,
        overrides=overrides,
        experiment_name=cfg.experiment.name,
    )
    return cfg


__all__ = ["resolve_experiment_config"]
