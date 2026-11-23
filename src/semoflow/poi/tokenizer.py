from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from semoflow.config import ExperimentConfig, TokenizerConfig
from semoflow.poi.rqvae import RQVAE, build_tokenizer_model
from semoflow.poi.sids import SemanticIDGroundingIndex, sid_code_names
from semoflow.poi.text import load_poi_embeddings

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
_EARTH_KM_PER_DEG = 111.32


def _pca(values: np.ndarray, n_components: int) -> tuple[np.ndarray, dict]:
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(values).astype(np.float32)
    logger.info(
        "PCA %dd -> %dd  explained_var=%.4f",
        values.shape[1],
        n_components,
        float(pca.explained_variance_ratio_.sum()),
    )
    return reduced, {
        "mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
    }


def city_relative_position(
    places: pl.DataFrame, dim: int
) -> tuple[np.ndarray, dict[int, tuple[float, float]]]:
    fitted = (
        places.drop_nulls(["latitude", "longitude"])
        .group_by("city_id")
        .agg(pl.col("latitude").mean(), pl.col("longitude").mean())
    )
    centroids = {
        int(city): (float(lat), float(lon))
        for city, lat, lon in fitted.select(
            "city_id", "latitude", "longitude"
        ).iter_rows()
    }
    city = places["city_id"].to_numpy()
    lat0 = np.array([centroids[int(c)][0] for c in city])
    lon0 = np.array([centroids[int(c)][1] for c in city])
    lat = places["latitude"].to_numpy().astype(float)
    lon = places["longitude"].to_numpy().astype(float)
    finite = np.isfinite(lat) & np.isfinite(lon)
    dy = (np.where(finite, lat, lat0) - lat0) * _EARTH_KM_PER_DEG
    dx = (
        (np.where(finite, lon, lon0) - lon0)
        * _EARTH_KM_PER_DEG
        * np.cos(np.radians(lat0))
    )
    columns = [dy, dx]
    for wavelength in np.geomspace(40.0, 2.0, (dim - 2) // 4):
        for value in (dy, dx):
            angle = 2.0 * np.pi * value / wavelength
            columns += [np.sin(angle), np.cos(angle)]
    columns.append(finite)
    logger.info(
        "city-relative position  dim=%d  cities=%d  missing_coordinates=%d",
        dim,
        len(centroids),
        int((~finite).sum()),
    )
    return np.stack(columns, axis=1).astype(np.float32), centroids


def build_features(
    cfg: TokenizerConfig, places: pl.DataFrame, layout: Any
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    text, text_pca = _pca(
        load_poi_embeddings(layout.text_embeddings, places).numpy(),
        cfg.pca_text_dim,
    )
    graph, graph_pca = _pca(
        load_poi_embeddings(layout.graph_embeddings, places).numpy(),
        cfg.pca_graph_dim,
    )
    geo, centroids = city_relative_position(places, cfg.geo_relative_dim)
    columns = [
        *(f"pca_text_{i}" for i in range(cfg.pca_text_dim)),
        *(f"pca_graph_{i}" for i in range(cfg.pca_graph_dim)),
        *(f"geo_{i}" for i in range(cfg.geo_relative_dim)),
        "geoflag_valid",
    ]
    x = torch.from_numpy(np.concatenate([text, graph, geo], axis=1))
    state = {
        "text_pca": text_pca,
        "graph_pca": graph_pca,
        "geo_centroids": centroids,
    }
    return x, columns, state


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
    return {key: float(value) / count for key, value in totals.items()}


def _fit(
    tcfg: TokenizerConfig, x: torch.Tensor
) -> tuple[RQVAE, list[dict[str, float]]]:
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
    history = []
    for epoch in range(tcfg.epochs + 1):
        metrics = _run_epoch(
            model, batches, device, optimizer if epoch else None
        )
        logger.info(
            "[tokenizer] epoch %d/%d  %s",
            epoch,
            tcfg.epochs,
            "  ".join(f"{key}={value:.6g}" for key, value in metrics.items()),
        )
        if epoch:
            history.append({**metrics, "epoch": float(epoch)})
    return model, history


def run_semantic_id_tokenizer(cfg: ExperimentConfig) -> None:
    tcfg, layout = cfg.tokenizer, cfg.layout
    arts = layout.tokenizer
    logger.info("building semantic IDs  run_dir=%s", arts.run_dir)
    places = pl.read_parquet(layout.preprocessing.places_index)
    x_raw, columns, feature_state = build_features(tcfg, places, layout)

    torch.manual_seed(tcfg.seed)
    x_mean = x_raw.mean(dim=0, keepdim=True)
    x_std = x_raw.std(dim=0, keepdim=True).clamp(min=1e-8)
    x = (x_raw - x_mean) / x_std
    text_end = tcfg.pca_text_dim
    geo_start = text_end + tcfg.pca_graph_dim
    x[:, :text_end] *= tcfg.text_graph_scale
    x[:, geo_start : geo_start + tcfg.geo_relative_dim] *= tcfg.geo_scale
    model, history = _fit(tcfg, x)

    model.eval()
    with torch.no_grad():
        codes = torch.cat(
            [
                model.encode(batch.to(tcfg.device)).cpu()
                for batch in x.split(_ENCODE_BATCH)
            ]
        ).numpy()
    code_cols = list(sid_code_names(codes.shape[1]))
    code_frame = pl.DataFrame(
        {
            name: codes[:, level].astype(np.int32)
            for level, name in enumerate(code_cols)
        }
    )
    lookup = places.select(_LOOKUP_COLUMNS).hstack(code_frame)
    lookup.write_parquet(arts.placekey_lookup)
    SemanticIDGroundingIndex.from_frame(
        places.select(_GROUNDING_COLUMNS).hstack(code_frame),
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
            "feature_preprocessing": feature_state,
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
        "tokenizer complete  visits=%s  codebooks=%s",
        arts.augmented_processed_visits,
        arts.tokenizer_codebooks,
    )


__all__ = [
    "run_semantic_id_tokenizer",
]
