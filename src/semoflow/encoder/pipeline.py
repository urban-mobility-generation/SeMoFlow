from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import polars as pl

from semoflow.config import ExperimentConfig

logger = logging.getLogger(__name__)


def run_text_encoder(cfg: ExperimentConfig) -> None:
    text_cfg = cfg.encoder_pipeline.text
    if not text_cfg.enabled:
        logger.info("text encoder skipped (enabled=false)")
        return
    from semoflow.encoder.text.sentence_t5 import build_text_embeddings

    places_index_path = cfg.tokenizer_config.places_index_output_path()
    if not places_index_path.exists():
        raise FileNotFoundError(
            f"Missing places index for text encoder: {places_index_path}"
        )
    logger.info("running text encoder  model=%s", text_cfg.model)
    started = time.perf_counter()
    build_text_embeddings(pl.read_parquet(places_index_path), text_cfg)
    text_out = Path(text_cfg.out_path)
    logger.info(
        "text encoder done  path=%s  elapsed=%.1fs",
        text_out,
        time.perf_counter() - started,
    )


def run_graph_encoder(cfg: ExperimentConfig) -> dict[str, Any] | None:
    graph_cfg = cfg.encoder_pipeline.graph
    if not graph_cfg.enabled:
        logger.info("graph encoder skipped (enabled=false)")
        return None
    from semoflow.training.graph_encoder import train_heterosage

    started = time.perf_counter()
    result = train_heterosage(cfg)
    logger.info(
        "graph encoder done  elapsed=%.1fs", time.perf_counter() - started
    )
    return result


def run_encoder_pipeline(cfg: ExperimentConfig) -> dict[str, Any] | None:
    encoders = cfg.encoder_pipeline
    if encoders.graph.enabled and not encoders.text.enabled:
        logger.warning(
            "GraphSAGE reads text embeddings as node features but "
            "encoder_pipeline.text.enabled=false; POI nodes will use "
            "random features"
        )
    run_text_encoder(cfg)
    return run_graph_encoder(cfg)


__all__ = [
    "run_encoder_pipeline",
    "run_graph_encoder",
    "run_text_encoder",
]
