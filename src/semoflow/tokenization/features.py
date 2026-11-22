from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from semoflow.encoder.artifacts import load_poi_embedding_artifact

logger = logging.getLogger(__name__)

TEXT_PREFIX = "text_emb_"
GRAPH_PREFIX = "graph_emb_"
PCA_TEXT_PREFIX = "pca_text_"
PCA_GRAPH_PREFIX = "pca_graph_"
GEO_PREFIX = "geo_"
_ID_COLUMNS = ("placekey_id", "placekey")
_EARTH_KM_PER_DEG = 111.32


def feature_columns(frame: pl.DataFrame) -> list[str]:
    return [
        c
        for c, dtype in zip(frame.columns, frame.dtypes, strict=True)
        if c not in _ID_COLUMNS and dtype.is_numeric()
    ]


def frame_to_numpy(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame.select(columns).to_numpy().astype("float32", copy=False)
    return np.nan_to_num(values, copy=True).astype("float32", copy=False)


def _identity_frame(places_index: pl.DataFrame) -> pl.DataFrame:
    if "placekey_id" not in places_index.columns:
        raise ValueError("places index must contain placekey_id")
    return places_index.select(
        [
            pl.col(c).cast(pl.Utf8) if c == "placekey" else pl.col(c)
            for c in _ID_COLUMNS
            if c in places_index.columns
        ]
    )


def _with_block(
    base: pl.DataFrame, prefix: str, values: np.ndarray
) -> pl.DataFrame:
    return base.with_columns(
        [
            pl.Series(f"{prefix}{i}", values[:, i].astype("float32"))
            for i in range(values.shape[1])
        ]
    )


def text_feature_frame(
    places_index: pl.DataFrame, embeddings_path: Path
) -> pl.DataFrame:
    embeddings = (
        load_poi_embedding_artifact(embeddings_path, kind="text")
        .align_to(places_index)
        .numpy()
    )
    return _with_block(_identity_frame(places_index), TEXT_PREFIX, embeddings)


def graph_feature_frame(
    places_index: pl.DataFrame, artifact_dir: Path
) -> pl.DataFrame:
    emb_path = artifact_dir / "poi_graph_emb.pt"
    index_path = artifact_dir / "places_index.parquet"
    for path in (emb_path, index_path):
        if not path.exists():
            raise FileNotFoundError(f"GraphSAGE artifact not found: {path}")
    embeddings = (
        load_poi_embedding_artifact(
            emb_path, index_path=index_path, kind="graph"
        )
        .align_to(places_index)
        .numpy()
    )
    return _with_block(_identity_frame(places_index), GRAPH_PREFIX, embeddings)


def _pca(
    frame: pl.DataFrame, src_prefix: str, dst_prefix: str, n_components: int
) -> tuple[pl.DataFrame, dict[str, Any]]:
    from sklearn.decomposition import PCA

    src_cols = sorted(c for c in frame.columns if c.startswith(src_prefix))
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(frame.select(src_cols).to_numpy())
    logger.info(
        "PCA %s %dd -> %dd  explained_var=%.4f",
        src_prefix.rstrip("_"),
        len(src_cols),
        n_components,
        float(pca.explained_variance_ratio_.sum()),
    )
    parameters = {
        "input_columns": src_cols,
        "mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
    }
    return _with_block(frame.drop(src_cols), dst_prefix, reduced), parameters


def city_relative_position_frame(
    places_index: pl.DataFrame, dim: int
) -> tuple[pl.DataFrame, dict[int, tuple[float, float]]]:
    required = {"placekey_id", "latitude", "longitude", "city_id"}
    missing = required - set(places_index.columns)
    if missing:
        raise ValueError(
            "city-relative position requires "
            f"{sorted(required)} on the places index; missing {sorted(missing)}"
        )
    frame = places_index.select(sorted(required))
    fitted = (
        frame.drop_nulls(["latitude", "longitude"])
        .group_by("city_id")
        .agg(
            pl.col("latitude").mean().alias("lat0"),
            pl.col("longitude").mean().alias("lon0"),
        )
    )
    centroids = {
        int(c): (float(a), float(o))
        for c, a, o in zip(
            fitted["city_id"], fitted["lat0"], fitted["lon0"], strict=True
        )
    }
    city = frame["city_id"].to_numpy()
    lat0 = np.array([centroids[int(c)][0] for c in city])
    lon0 = np.array([centroids[int(c)][1] for c in city])
    lat = frame["latitude"].to_numpy().astype(float)
    lon = frame["longitude"].to_numpy().astype(float)
    finite = np.isfinite(lat) & np.isfinite(lon)
    lat = np.where(finite, lat, lat0)
    lon = np.where(finite, lon, lon0)
    dy = (lat - lat0) * _EARTH_KM_PER_DEG
    dx = (lon - lon0) * _EARTH_KM_PER_DEG * np.cos(np.radians(lat0))
    columns = {"geo_0": dy.astype("float32"), "geo_1": dx.astype("float32")}
    indicator = {"geoflag_valid": finite.astype("float32")}
    pairs = (dim - 2) // 4
    if pairs:
        index = 2
        for wavelength in np.geomspace(40.0, 2.0, pairs):
            for value in (dy, dx):
                angle = 2.0 * np.pi * value / wavelength
                columns[f"geo_{index}"] = np.sin(angle).astype("float32")
                columns[f"geo_{index + 1}"] = np.cos(angle).astype("float32")
                index += 2
    if len(columns) != dim:
        raise ValueError(
            f"geo_relative_dim {dim} produced {len(columns)} columns"
        )
    logger.info(
        "city-relative position: %d columns over %d cities, %d POIs without "
        "coordinates centred at their city centre and flagged",
        dim,
        len(centroids),
        int((~finite).sum()),
    )
    return (
        frame.select("placekey_id").with_columns(
            **{k: pl.Series(k, v) for k, v in (columns | indicator).items()}
        ),
        centroids,
    )


def build_features(
    *,
    places_index: pl.DataFrame,
    text_embeddings_path: Path,
    graph_dir: Path,
    pca_text_dim: int,
    pca_graph_dim: int,
    geo_relative_dim: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    frame, text_pca = _pca(
        text_feature_frame(places_index, text_embeddings_path),
        TEXT_PREFIX,
        PCA_TEXT_PREFIX,
        pca_text_dim,
    )
    graph, graph_pca = _pca(
        graph_feature_frame(places_index, graph_dir),
        GRAPH_PREFIX,
        PCA_GRAPH_PREFIX,
        pca_graph_dim,
    )
    frame = frame.with_columns(
        graph.select(
            c for c in graph.columns if c.startswith(PCA_GRAPH_PREFIX)
        ).get_columns()
    )
    geo, centroids = city_relative_position_frame(
        places_index, geo_relative_dim
    )
    rows = frame.height
    frame = frame.join(geo, on="placekey_id", maintain_order="left")
    if frame.height != rows:
        raise ValueError("city-relative position did not cover every POI")
    return frame, {
        **text_pca,
        "graph_pca": graph_pca,
        "geo_centroids": centroids,
    }


def scale_feature_blocks(
    x: torch.Tensor,
    columns: list[str],
    *,
    text_graph_scale: float,
    geo_scale: float,
) -> torch.Tensor:
    x = x.clone()
    for prefix, scale in (
        (PCA_TEXT_PREFIX, text_graph_scale),
        (GEO_PREFIX, geo_scale),
    ):
        index = [i for i, c in enumerate(columns) if c.startswith(prefix)]
        x[:, index] *= float(scale)
    return x


__all__ = [
    "build_features",
    "city_relative_position_frame",
    "feature_columns",
    "frame_to_numpy",
    "graph_feature_frame",
    "scale_feature_blocks",
    "text_feature_frame",
]
