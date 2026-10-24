from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from semoflow.config.schemas.tokenizer_config import TokenizerConfig
from semoflow.tokenization.features import (
    build_features,
    scale_feature_blocks,
)
from semoflow.tokenization.rqvae import build_tokenizer_model
from semoflow.training.tokenizer_publish import encode_feature_tensors

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrozenEncodeResult:
    output: str
    rows: int
    distinct_full_codes: int
    fitted_feature_columns: int
    zero_filled_columns: int
    dropped_columns: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _align_to_fitted_columns(
    frame: pl.DataFrame, fitted: list[str]
) -> tuple[np.ndarray, int, int]:
    present = set(frame.columns)
    missing = [c for c in fitted if c not in present]
    unseen = [
        c
        for c in frame.columns
        if c not in set(fitted) and c not in {"placekey_id", "placekey"}
    ]
    if missing:
        frame = frame.with_columns(
            [pl.lit(0.0, dtype=pl.Float32).alias(c) for c in missing]
        )
    logger.info(
        "column alignment: %d fitted, %d zero-filled (absent here), "
        "%d dropped (unseen by the fit)",
        len(fitted),
        len(missing),
        len(unseen),
    )
    if len(missing) == len(fitted):
        raise ValueError(
            "no fitted feature column is present in the new POI set; the "
            "checkpoint and the places index almost certainly disagree"
        )
    values = frame.select(fitted).to_numpy().astype("float32", copy=False)
    return values, len(missing), len(unseen)


def _rebuild_config(data: dict) -> TokenizerConfig:
    fields = {f.name for f in dataclasses.fields(TokenizerConfig)}
    unknown = sorted(set(data) - fields)
    if unknown:
        logger.info(
            "checkpoint config carries %d unknown keys: %s",
            len(unknown),
            unknown,
        )
    return TokenizerConfig(**{k: v for k, v in data.items() if k in fields})


@dataclass(frozen=True)
class FrozenFeatures:
    places: pl.DataFrame
    x: torch.Tensor
    payload: dict[str, Any]
    fitted_columns: list[str]
    zero_filled_columns: int
    dropped_columns: int


def load_frozen_features(
    *,
    checkpoint: Path,
    places_index: Path,
    city_ids: list[int] | None = None,
    text_embeddings: Path | None = None,
    graphsage_artifact_dir: Path | None = None,
) -> FrozenFeatures:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fitted_columns = list(payload["feature_columns"])
    config = _rebuild_config(payload["config"])
    source = config.input_embedding_source
    pca = payload.get("feature_preprocessing", {}).get("pca")
    if not isinstance(pca, dict):
        raise ValueError(
            "checkpoint has no fit-time PCA parameters; refusing to refit "
            "PCA on the target rows"
        )
    if payload.get("graph_dim") is None and (
        config.lambda_graph > 0 or config.lambda_prefix_graph > 0
    ):
        raise ValueError(
            "checkpoint is graph-aligned but does not record graph_dim; "
            "re-publish the tokenizer before frozen encoding"
        )
    logger.info(
        "frozen tokenizer: source=%s codebooks=%d size=%d dims=%d",
        source,
        payload["n_codebooks"],
        payload["codebook_size"],
        len(fitted_columns),
    )

    places = pl.read_parquet(places_index)
    if city_ids:
        if "city_id" not in places.columns:
            raise ValueError("places index has no city_id column")
        present = {int(c) for c in places["city_id"].unique().to_list()}
        absent = [c for c in city_ids if c not in present]
        if absent:
            raise ValueError(
                f"cities {absent} are not in {places_index} "
                f"(present: {sorted(present)})"
            )
        places = places.filter(pl.col("city_id").is_in(city_ids))
    logger.info("encoding %d POIs", places.height)

    if text_embeddings is None:
        text_embeddings = Path(config.text_embeddings_path)
    if graphsage_artifact_dir is None and config.graphsage_artifact_dir:
        graphsage_artifact_dir = Path(config.graphsage_artifact_dir)
    frame, _ = build_features(
        places_index=places,
        source=source,
        text_embeddings_path=text_embeddings,
        graphsage_artifact_dir=graphsage_artifact_dir,
        pca_text_dim=config.pca_text_dim,
        pca_graph_dim=config.pca_graph_dim,
        geo_relative_dim=config.geo_relative_dim,
        state=pca,
    )
    values, zero_filled, dropped = _align_to_fitted_columns(
        frame, fitted_columns
    )
    x = (torch.from_numpy(values) - payload["feature_mean"]) / payload[
        "feature_std"
    ]
    x = scale_feature_blocks(
        x,
        fitted_columns,
        text_graph_scale=config.text_graph_scale,
        geo_scale=config.geo_scale,
    )
    return FrozenFeatures(
        places=places,
        x=x,
        payload=payload,
        fitted_columns=fitted_columns,
        zero_filled_columns=zero_filled,
        dropped_columns=dropped,
    )


def encode_with_frozen_tokenizer(
    *,
    checkpoint: Path,
    places_index: Path,
    output: Path,
    city_ids: list[int] | None = None,
    text_embeddings: Path | None = None,
    graphsage_artifact_dir: Path | None = None,
    device: str = "cpu",
) -> FrozenEncodeResult:
    features = load_frozen_features(
        checkpoint=checkpoint,
        places_index=places_index,
        city_ids=city_ids,
        text_embeddings=text_embeddings,
        graphsage_artifact_dir=graphsage_artifact_dir,
    )
    payload = features.payload
    places = features.places
    fitted_columns = features.fitted_columns
    graph_dim = payload.get("graph_dim")
    model = build_tokenizer_model(
        _rebuild_config(payload["config"]),
        input_dim=len(fitted_columns),
        graph_dim=None if graph_dim is None else int(graph_dim),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    codes = encode_feature_tensors(model, {"x": features.x}, device=device)

    out = places.select(
        [c for c in ("placekey_id", "placekey") if c in places.columns]
    )
    if "city_id" in places.columns:
        out = out.with_columns(places["city_id"])
    for level in range(codes.shape[1]):
        out = out.with_columns(pl.Series(f"sem_code_{level}", codes[:, level]))

    output.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output)
    code_cols = [c for c in out.columns if c.startswith("sem_code_")]
    distinct = out.select(code_cols).n_unique()
    logger.info(
        "wrote %s  rows=%d  distinct full codes=%d",
        output,
        out.height,
        distinct,
    )
    return FrozenEncodeResult(
        output=str(output),
        rows=out.height,
        distinct_full_codes=distinct,
        fitted_feature_columns=len(fitted_columns),
        zero_filled_columns=features.zero_filled_columns,
        dropped_columns=features.dropped_columns,
    )


__all__ = ["FrozenEncodeResult", "encode_with_frozen_tokenizer"]
