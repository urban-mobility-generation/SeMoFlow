from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from semoflow.config import ExperimentConfig
from semoflow.config.artifacts import TokenizerArtifacts
from semoflow.logging.primitives import configure_logging
from semoflow.preprocess.places_index import (
    load_places_index,
)
from semoflow.tokenization.features import (
    GRAPH_PREFIX,
    build_features,
    feature_columns,
    frame_to_numpy,
    graph_feature_frame,
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
from semoflow.training.tokenizer_publish import (
    publish_tokenizer_artifacts,
)
from semoflow.utils import (
    init_run_directory,
    write_run_manifest,
)

logger = logging.getLogger(__name__)


@dataclass
class PreparedFeatures:
    tensors: dict[str, torch.Tensor]
    feature_cols: list[str]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    feature_preprocessing: dict[str, Any]
    graph_dim: int | None
    poi_features: pl.DataFrame
    places_index: pl.DataFrame
    stays: pl.DataFrame


def prepare_features(cfg: ExperimentConfig) -> PreparedFeatures:
    tcfg = cfg.tokenizer_config
    places_index = load_places_index(tcfg.places_index_output_path())
    stays = pl.read_parquet(tcfg.processed_visits_output_path())
    graph_dir = (
        Path(tcfg.graphsage_artifact_dir)
        if tcfg.graphsage_artifact_dir
        else None
    )
    poi_features, pca_state = build_features(
        places_index=places_index,
        source=tcfg.input_embedding_source,
        text_embeddings_path=Path(tcfg.text_embeddings_path),
        graphsage_artifact_dir=graph_dir,
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
    tensors = {"x": x_norm}

    graph_dim = None
    if tcfg.lambda_graph > 0 or tcfg.lambda_prefix_graph > 0:
        if graph_dir is None:
            raise ValueError(
                "graph alignment requires tokenizer_config."
                "graphsage_artifact_dir"
            )
        graph_targets = _graph_targets(poi_features, places_index, graph_dir)
        tensors["graph_target"] = torch.from_numpy(graph_targets)
        graph_dim = int(graph_targets.shape[1])

    return PreparedFeatures(
        tensors=tensors,
        feature_cols=feature_cols,
        feature_mean=x_mean,
        feature_std=x_std,
        feature_preprocessing={"pca": pca_state},
        graph_dim=graph_dim,
        poi_features=poi_features,
        places_index=places_index,
        stays=stays,
    )


def _graph_targets(
    poi_features: pl.DataFrame,
    places_index: pl.DataFrame,
    artifact_dir: Path,
) -> np.ndarray:
    graph = graph_feature_frame(places_index, artifact_dir)
    graph_cols = [c for c in graph.columns if c.startswith(GRAPH_PREFIX)]
    aligned = poi_features.select("placekey_id").join(
        graph.select("placekey_id", *graph_cols),
        on="placekey_id",
        how="left",
        maintain_order="left",
    )
    return frame_to_numpy(aligned, graph_cols)


def fit_tokenizer(
    cfg: ExperimentConfig,
    prepared: PreparedFeatures,
    *,
    checkpoints_dir: Path,
) -> tuple[RQVAE, list[dict[str, float]]]:
    tcfg = cfg.tokenizer_config
    model = build_tokenizer_model(
        tcfg,
        input_dim=len(prepared.feature_cols),
        graph_dim=prepared.graph_dim,
    )
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
        mixed_precision=tcfg.mixed_precision,
        checkpoint_dir=checkpoints_dir,
    )
    return model, history


def run_semantic_id_tokenizer(
    cfg: ExperimentConfig,
    *,
    force: bool = False,
    log_level: str = "INFO",
    log_file: Path | None = None,
) -> Path | None:
    tcfg = cfg.tokenizer_config
    arts = TokenizerArtifacts.from_run_dir(tcfg.run_dir())
    paths = init_run_directory(arts.run_dir)
    if log_file is None:
        configure_logging(level=log_level, log_file=paths.logs_dir / "run.log")

    if arts.is_complete() and not force:
        logger.info(
            "semantic tokenizer run already complete: %s", arts.run_dir
        )
        return None
    if arts.augmented_processed_visits.exists() and not force:
        logger.info(
            "partial tokenizer run detected; republishing: %s", arts.run_dir
        )

    cfg.save_yaml(arts.config_yaml, phase="tokenizer")
    resolved_config_path = arts.run_dir / cfg.output.config_filename
    cfg.save_yaml(resolved_config_path, phase="tokenizer")
    write_run_manifest(
        arts.run_dir,
        config_path=resolved_config_path,
        output_dir=arts.run_dir,
        data_artifact_paths={
            "data_root": cfg.data.data_root,
            "preprocessed_processed_visits_path": cfg.paths.preprocessed_processed_visits_path,
            "semantic_id_lookup_path": cfg.paths.semantic_id_lookup_path,
            "semantic_id_grounding_path": cfg.paths.semantic_id_grounding_path,
            "tokenizer_preprocessing_dir": tcfg.preprocessing_dir,
            "tokenizer_graphsage_artifact_dir": tcfg.graphsage_artifact_dir,
        },
        metric_artifact_paths={
            "semantic_id_eval": arts.eval_dir / "semantic_id_eval.json"
        },
    )

    logger.info("building semantic IDs  run_dir=%s", arts.run_dir)
    prepared = prepare_features(cfg)
    model, history = fit_tokenizer(
        cfg, prepared, checkpoints_dir=paths.checkpoints_dir
    )
    return publish_tokenizer_artifacts(
        cfg, arts, model=model, prepared=prepared, history=history
    )


__all__ = [
    "PreparedFeatures",
    "fit_tokenizer",
    "prepare_features",
    "run_semantic_id_tokenizer",
]
