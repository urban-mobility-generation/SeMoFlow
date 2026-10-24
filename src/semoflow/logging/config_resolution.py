from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from semoflow.logging.primitives import (
    log_kv_block,
    log_section,
    log_section_end,
)


def log_config_resolution(
    logger: logging.Logger,
    *,
    config_path: Path | None,
    overrides: Sequence[str],
    experiment_name: str,
) -> None:
    log_section(logger, "Config Resolution")
    log_kv_block(
        logger,
        None,
        [
            ("config_path", config_path or "defaults only"),
            ("experiment", experiment_name),
        ],
    )
    if overrides:
        logger.info("  Overrides (%d)", len(overrides))
        for override in overrides:
            logger.info("    %s", override)
    log_section_end(logger)


__all__ = ["log_config_resolution"]
