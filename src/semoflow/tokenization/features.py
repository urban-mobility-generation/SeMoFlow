from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from semoflow.encoder.artifacts import (
    load_poi_embedding_artifact,
)

logger = logging.getLogger(__name__)

INPUT_SOURCES = ("pca_text", "pca_concat")
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
    if not columns:
        raise ValueError("feature columns must be non-empty")
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


def load_graph_embeddings(
    artifact_dir: Path,
) -> tuple[pl.DataFrame, torch.Tensor]:
    root = Path(artifact_dir).expanduser().resolve()
    emb_path = root / "poi_graph_emb.pt"
    index_path = root / "places_index.parquet"
    if not (emb_path.exists() and index_path.exists()):
        raise FileNotFoundError(
            f"graphsage_artifact_dir must contain {emb_path.name} and "
            f"{index_path.name}; got {root}"
        )
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)
    return pl.read_parquet(index_path), embeddings


def _pca_fit(
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


def _pca_apply(
    frame: pl.DataFrame,
    src_prefix: str,
    dst_prefix: str,
    n_components: int,
    parameters: dict[str, Any],
) -> pl.DataFrame:
    src_cols = sorted(c for c in frame.columns if c.startswith(src_prefix))
    if list(parameters.get("input_columns") or []) != src_cols:
        raise ValueError(
            "frozen PCA input columns do not match the tokenizer fit: "
            f"expected {parameters.get('input_columns')}, got {src_cols}"
        )
    mean = np.asarray(parameters.get("mean"), dtype=np.float32)
    components = np.asarray(parameters.get("components"), dtype=np.float32)
    if mean.shape != (len(src_cols),):
        raise ValueError(
            "frozen PCA mean has the wrong shape: "
            f"expected {(len(src_cols),)}, got {mean.shape}"
        )
    if components.shape != (n_components, len(src_cols)):
        raise ValueError(
            "frozen PCA components have the wrong shape: "
            f"expected {(n_components, len(src_cols))}, got "
            f"{components.shape}"
        )
    values = frame.select(src_cols).to_numpy().astype(np.float32, copy=False)
    reduced = (values - mean) @ components.T
    return _with_block(frame.drop(src_cols), dst_prefix, reduced)


def _pca(
    frame: pl.DataFrame,
    src_prefix: str,
    dst_prefix: str,
    n_components: int,
    parameters: dict[str, Any] | None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if parameters is None:
        return _pca_fit(frame, src_prefix, dst_prefix, n_components)
    return (
        _pca_apply(frame, src_prefix, dst_prefix, n_components, parameters),
        parameters,
    )


def city_relative_position_frame(
    places_index: pl.DataFrame,
    dim: int,
    centroids: dict[int, tuple[float, float]] | None = None,
) -> tuple[pl.DataFrame, dict[int, tuple[float, float]]]:
    required = {"placekey_id", "latitude", "longitude", "city_id"}
    missing = required - set(places_index.columns)
    if missing:
        raise ValueError(
            "city-relative position requires "
            f"{sorted(required)} on the places index; missing {sorted(missing)}"
        )
    frame = places_index.select(sorted(required))
    if centroids is None:
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
    unknown = sorted(set(int(c) for c in city) - set(centroids))
    if unknown:
        raise ValueError(
            "city-relative position has no fitted centroid for city ids "
            f"{unknown}; refit rather than centring on a foreign frame"
        )
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
    source: str,
    text_embeddings_path: Path,
    graphsage_artifact_dir: Path | None,
    pca_text_dim: int | None,
    pca_graph_dim: int | None = None,
    geo_relative_dim: int | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if source not in INPUT_SOURCES:
        raise ValueError(
            f"input_embedding_source must be one of {INPUT_SOURCES}; "
            f"got {source!r}"
        )
    if pca_text_dim is None:
        raise ValueError(f"pca_text_dim must be set for source {source!r}")
    fitting = state is None
    frame, state = _pca(
        text_feature_frame(places_index, text_embeddings_path),
        TEXT_PREFIX,
        PCA_TEXT_PREFIX,
        pca_text_dim,
        state,
    )
    if source == "pca_text":
        return frame, state

    if graphsage_artifact_dir is None:
        raise ValueError("pca_concat requires graphsage_artifact_dir")
    graph = graph_feature_frame(places_index, graphsage_artifact_dir)
    graph_prefix = GRAPH_PREFIX
    if pca_graph_dim is not None:
        if not fitting and "graph_pca" not in state:
            raise ValueError(
                "pca_graph_dim is set but the fitted preprocessing state "
                "carries no graph PCA basis; refit rather than encoding "
                "through a text-only basis"
            )
        graph, state["graph_pca"] = _pca(
            graph,
            GRAPH_PREFIX,
            PCA_GRAPH_PREFIX,
            pca_graph_dim,
            state.get("graph_pca"),
        )
        graph_prefix = PCA_GRAPH_PREFIX
    graph_cols = [c for c in graph.columns if c.startswith(graph_prefix)]
    frame = frame.with_columns(graph.select(graph_cols).get_columns())

    if geo_relative_dim is not None:
        if not fitting and "geo_centroids" not in state:
            raise ValueError(
                "geo_relative_dim is set but the fitted preprocessing state "
                "carries no city centroids; refit rather than centring this "
                "POI set on its own frame"
            )
        geo, state["geo_centroids"] = city_relative_position_frame(
            places_index, geo_relative_dim, state.get("geo_centroids")
        )
        before = frame.height
        frame = frame.join(
            geo, on="placekey_id", how="inner", maintain_order="left"
        )
        if frame.height != before:
            raise ValueError(
                "city-relative position did not cover every POI: "
                f"{before} rows became {frame.height}"
            )
    return frame, state


def _scale_block(
    x: torch.Tensor,
    columns: list[str],
    prefix: str,
    scale: float | None,
    field: str,
) -> torch.Tensor:
    if scale is None:
        return x
    index = [i for i, c in enumerate(columns) if c.startswith(prefix)]
    if not index:
        raise ValueError(
            f"{field} is set but no {prefix!r} columns are present"
        )
    scaled = x.clone()
    scaled[:, index] *= float(scale)
    logger.info(
        "%s %.4f applied to %d standardized %s columns (%d others at 1.0)",
        field,
        scale,
        len(index),
        prefix.rstrip("_"),
        x.shape[1] - len(index),
    )
    return scaled


def scale_feature_blocks(
    x: torch.Tensor,
    columns: list[str],
    *,
    text_graph_scale: float | None,
    geo_scale: float | None,
) -> torch.Tensor:
    x = _scale_block(
        x, columns, PCA_TEXT_PREFIX, text_graph_scale, "text_graph_scale"
    )
    return _scale_block(x, columns, GEO_PREFIX, geo_scale, "geo_scale")


__all__ = [
    "INPUT_SOURCES",
    "build_features",
    "city_relative_position_frame",
    "feature_columns",
    "frame_to_numpy",
    "graph_feature_frame",
    "load_graph_embeddings",
    "scale_feature_blocks",
    "text_feature_frame",
]
