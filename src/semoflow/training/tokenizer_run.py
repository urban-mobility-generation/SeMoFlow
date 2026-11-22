from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch.utils.data import DataLoader

from semoflow.config import ExperimentConfig
from semoflow.tokenization.features import (
    build_features,
    feature_columns,
    frame_to_numpy,
    scale_feature_blocks,
)
from semoflow.tokenization.rqvae import (
    RQVAE,
    build_tokenizer_model,
)
from semoflow.training.tokenizer import (
    TensorDictDataset,
    train_tokenizer,
)
from semoflow.training.tokenizer_publish import publish_tokenizer_artifacts

logger = logging.getLogger(__name__)


@dataclass
class PreparedFeatures:
    tensors: dict[str, torch.Tensor]
    feature_cols: list[str]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    feature_preprocessing: dict[str, Any]
    poi_features: pl.DataFrame
    places_index: pl.DataFrame
    stays: pl.DataFrame


def prepare_features(cfg: ExperimentConfig) -> PreparedFeatures:
    tcfg = cfg.tokenizer
    layout = cfg.layout
    places_index = pl.read_parquet(layout.preprocessing.places_index)
    stays = pl.read_parquet(layout.preprocessing.processed_visits)
    poi_features, pca_state = build_features(
        places_index=places_index,
        text_embeddings_path=layout.text_embeddings,
        graph_dir=layout.graph_dir,
        pca_text_dim=tcfg.pca_text_dim,
        pca_graph_dim=tcfg.pca_graph_dim,
        geo_relative_dim=tcfg.geo_relative_dim,
    )

    torch.manual_seed(tcfg.seed)
    feature_cols = feature_columns(poi_features)
    x_raw = torch.from_numpy(frame_to_numpy(poi_features, feature_cols))
    x_mean = x_raw.mean(dim=0, keepdim=True)
    x_std = x_raw.std(dim=0, keepdim=True).clamp(min=1e-8)
    x_norm = scale_feature_blocks(
        (x_raw - x_mean) / x_std,
        feature_cols,
        text_graph_scale=tcfg.text_graph_scale,
        geo_scale=tcfg.geo_scale,
    )
    return PreparedFeatures(
        tensors={"x": x_norm},
        feature_cols=feature_cols,
        feature_mean=x_mean,
        feature_std=x_std,
        feature_preprocessing={"pca": pca_state},
        poi_features=poi_features,
        places_index=places_index,
        stays=stays,
    )


def fit_tokenizer(
    cfg: ExperimentConfig, prepared: PreparedFeatures
) -> tuple[RQVAE, list[dict[str, float]]]:
    tcfg = cfg.tokenizer
    model = build_tokenizer_model(tcfg, input_dim=len(prepared.feature_cols))
    generator = torch.Generator()
    generator.manual_seed(tcfg.seed)
    device = torch.device(tcfg.device)
    num_workers = tcfg.num_workers
    if num_workers == 0 and device.type == "cuda":
        num_workers = min(8, os.cpu_count() or 1)
    dataloader = DataLoader(
        TensorDictDataset(prepared.tensors),
        batch_size=tcfg.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    history = train_tokenizer(
        model,
        dataloader,
        epochs=tcfg.epochs,
        lr=tcfg.lr,
        device=tcfg.device,
    )
    return model, history


def run_semantic_id_tokenizer(cfg: ExperimentConfig) -> Path | None:
    """Fit the RQ-VAE and publish its artifacts into the run directory."""
    arts = cfg.layout.tokenizer
    if arts.augmented_processed_visits.exists():
        logger.info("replacing existing tokenizer outputs: %s", arts.run_dir)
    logger.info("building semantic IDs  run_dir=%s", arts.run_dir)
    prepared = prepare_features(cfg)
    model, history = fit_tokenizer(cfg, prepared)
    return publish_tokenizer_artifacts(
        cfg, arts, model=model, prepared=prepared, history=history
    )


__all__ = [
    "PreparedFeatures",
    "fit_tokenizer",
    "prepare_features",
    "run_semantic_id_tokenizer",
]
