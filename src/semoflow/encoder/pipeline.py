from __future__ import annotations

import logging
import time
from typing import Any

import polars as pl

from semoflow.config import ExperimentConfig

logger = logging.getLogger(__name__)


def run_encoder_pipeline(cfg: ExperimentConfig) -> dict[str, Any]:
    """Embed POI text, then train the graph encoder on those features."""
    from semoflow.encoder.text.sentence_t5 import build_text_embeddings
    from semoflow.training.graph_encoder import train_heterosage

    started = time.perf_counter()
    logger.info("running text encoder  model=%s", cfg.encoder.text.model)
    build_text_embeddings(
        pl.read_parquet(cfg.layout.preprocessing.places_index),
        cfg.encoder.text,
        path=cfg.layout.text_embeddings,
    )
    logger.info(
        "text encoder done  path=%s  elapsed=%.1fs",
        cfg.layout.text_embeddings,
        time.perf_counter() - started,
    )
    started = time.perf_counter()
    result = train_heterosage(cfg)
    logger.info(
        "graph encoder done  elapsed=%.1fs", time.perf_counter() - started
    )
    return result


__all__ = ["run_encoder_pipeline"]
