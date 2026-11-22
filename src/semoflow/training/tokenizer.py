"""Fit the RQ-VAE semantic-ID tokenizer and publish its run artifacts."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from semoflow.config import ExperimentConfig
from semoflow.tokenization.decoding.grounding_index import (
    SemanticIDGroundingIndex,
)
from semoflow.tokenization.features import (
    build_features,
    feature_columns,
    frame_to_numpy,
    scale_feature_blocks,
)
from semoflow.tokenization.rqvae import RQVAE, build_tokenizer_model
from semoflow.tokenization.token_space import sid_code_names

logger = logging.getLogger(__name__)

_LOOKUP_COLUMNS = ("placekey_id", "placekey", "city_id")
_GROUNDING_COLUMNS = (
    "placekey_id",
    "placekey",
    "top_category",
    "naics_code",
    "latitude",
    "longitude",
)
_ENCODE_BATCH = 2048


def _run_epoch(
    model: RQVAE,
    batches: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    totals: dict[str, torch.Tensor] = {}
    count = 0
    model.train(optimizer is not None)
    with torch.set_grad_enabled(optimizer is not None):
        for (x,) in batches:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                losses = model(x.to(device, non_blocking=True))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0) + value.detach()
            count += 1
    return {
        key: float(value.float().cpu().item() / max(count, 1))
        for key, value in totals.items()
    }


def _log(epoch: int, epochs: int, metrics: dict[str, float]) -> None:
    text = "  ".join(f"{key}={value:.6g}" for key, value in metrics.items())
    logger.info("[tokenizer] epoch %d/%d  %s", epoch, epochs, text)


def _fit(
    cfg: ExperimentConfig, x: torch.Tensor
) -> tuple[RQVAE, list[dict[str, float]]]:
    tcfg = cfg.tokenizer
    model = build_tokenizer_model(tcfg, input_dim=x.shape[1])
    device = torch.device(tcfg.device)
    num_workers = tcfg.num_workers
    if num_workers == 0 and device.type == "cuda":
        num_workers = min(8, os.cpu_count() or 1)
    batches = DataLoader(
        TensorDataset(x),
        batch_size=tcfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(tcfg.seed),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    _log(0, tcfg.epochs, _run_epoch(model, batches, device, None))
    history = []
    for epoch in range(1, tcfg.epochs + 1):
        metrics = _run_epoch(model, batches, device, optimizer)
        _log(epoch, tcfg.epochs, metrics)
        history.append({**metrics, "epoch": float(epoch)})
    return model, history


def _encode(model: RQVAE, x: torch.Tensor, device: str) -> torch.Tensor:
    model.to(torch.device(device))
    return torch.cat(
        [
            model.encode(x[start : start + _ENCODE_BATCH].to(device)).cpu()
            for start in range(0, x.shape[0], _ENCODE_BATCH)
        ]
    ).int()


def run_semantic_id_tokenizer(cfg: ExperimentConfig) -> Path:
    """Fit the RQ-VAE on POI features and publish lookup, grounding index,
    codebooks, model and SID-augmented visits into the run directory."""
    tcfg, layout = cfg.tokenizer, cfg.layout
    arts = layout.tokenizer
    if arts.augmented_processed_visits.exists():
        logger.info("replacing existing tokenizer outputs: %s", arts.run_dir)
    logger.info("building semantic IDs  run_dir=%s", arts.run_dir)
    places = pl.read_parquet(layout.preprocessing.places_index)
    features, pca_state = build_features(
        places_index=places,
        text_embeddings_path=layout.text_embeddings,
        graph_dir=layout.graph_dir,
        pca_text_dim=tcfg.pca_text_dim,
        pca_graph_dim=tcfg.pca_graph_dim,
        geo_relative_dim=tcfg.geo_relative_dim,
    )
    # Feature rows follow the places index, one per POI.
    if not features["placekey_id"].equals(places["placekey_id"]):
        raise ValueError("POI features do not follow the places index")

    torch.manual_seed(tcfg.seed)
    columns = feature_columns(features)
    x_raw = torch.from_numpy(frame_to_numpy(features, columns))
    x_mean = x_raw.mean(dim=0, keepdim=True)
    x_std = x_raw.std(dim=0, keepdim=True).clamp(min=1e-8)
    x = scale_feature_blocks(
        (x_raw - x_mean) / x_std,
        columns,
        text_graph_scale=tcfg.text_graph_scale,
        geo_scale=tcfg.geo_scale,
    )
    model, history = _fit(cfg, x)

    codes = _encode(model, x, tcfg.device).numpy()
    code_cols = list(sid_code_names(codes.shape[1]))
    code_frame = pl.DataFrame(
        {name: codes[:, level] for level, name in enumerate(code_cols)}
    )
    lookup = places.select(_LOOKUP_COLUMNS).hstack(code_frame)
    lookup.write_parquet(arts.placekey_lookup)
    SemanticIDGroundingIndex.from_frame(
        places.select(_GROUNDING_COLUMNS).hstack(code_frame),
        key_columns=code_cols,
        id_column="placekey_id",
        payload_columns=_GROUNDING_COLUMNS,
    ).save(arts.grounding_index)

    codebooks = model.get_codebooks().detach().cpu().float().contiguous()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": columns,
            "config": vars(tcfg),
            "history": history,
            "codebooks": codebooks,
            "feature_mean": x_mean,
            "feature_std": x_std,
            "feature_preprocessing": {"pca": pca_state},
        },
        arts.tokenizer_model,
    )
    torch.save({"codebooks": codebooks}, arts.tokenizer_codebooks)

    visits = pl.read_parquet(layout.preprocessing.processed_visits).join(
        lookup.select("placekey_id", *code_cols),
        on="placekey_id",
        how="left",
        maintain_order="left",
    )
    visits.with_columns(
        pl.col(code_cols).fill_null(0).cast(pl.Int32)
    ).write_parquet(arts.augmented_processed_visits)
    logger.info(
        "tokenizer training complete: %s  codebooks=%s",
        arts.augmented_processed_visits,
        arts.tokenizer_codebooks,
    )
    arts.mark_complete()
    return arts.augmented_processed_visits


__all__ = ["run_semantic_id_tokenizer"]
