from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from semoflow.logging.primitives import (
    log_kv_block,
    log_section,
    log_section_end,
)

if TYPE_CHECKING:
    from semoflow.config import ExperimentConfig


def log_config_resolution(
    logger: logging.Logger,
    *,
    config_path: Path | None,
    overrides: Sequence[str],
    experiment_name: str,
    phase: str,
    output_dir: Path,
) -> None:
    log_section(logger, "Config Resolution")
    log_kv_block(
        logger,
        None,
        [
            ("config_path", config_path or "defaults only"),
            ("experiment", experiment_name),
            ("phase", phase),
            ("output_dir", output_dir),
        ],
    )
    if overrides:
        logger.info("  Overrides (%d)", len(overrides))
        for override in overrides:
            logger.info("    %s", override)
    log_section_end(logger)


def log_config_summary(logger: logging.Logger, cfg: ExperimentConfig) -> None:
    log_section(logger, "Config Summary")
    log_kv_block(
        logger,
        "Experiment",
        [("name", cfg.experiment.name), ("phase", cfg.experiment.phase)],
    )
    log_kv_block(logger, "Data", [("data_root", cfg.data.data_root)])
    log_kv_block(
        logger,
        "Runtime",
        [("device", cfg.runtime.device), ("seed", cfg.runtime.seed)],
    )
    log_kv_block(logger, "Output", [("output_dir", cfg.output.output_dir)])
    log_section_end(logger)


def log_full_config_yaml(
    logger: logging.Logger, cfg: ExperimentConfig
) -> None:
    log_section(logger, "Full Resolved Config")
    for line in cfg.to_yaml().rstrip().splitlines():
        logger.info("  %s", line)
    log_section_end(logger)


__all__ = [
    "log_config_resolution",
    "log_config_summary",
    "log_full_config_yaml",
]
