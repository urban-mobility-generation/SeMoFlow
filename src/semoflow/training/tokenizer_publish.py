from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from semoflow.config import ExperimentConfig
from semoflow.config.artifacts import TokenizerArtifacts

logger = logging.getLogger(__name__)


def _attach_catalog_city_ids(
    lookup: pl.DataFrame,
    places_index: pl.DataFrame,
) -> pl.DataFrame:
    if "city_id" not in places_index.columns:
        return lookup
    city_lookup = places_index.select("placekey_id", "city_id").unique()
    if city_lookup["placekey_id"].n_unique() != city_lookup.height:
        raise ValueError(
            "tokenizer places index maps a placekey_id to multiple city_ids"
        )
    result = lookup.join(
        city_lookup,
        on="placekey_id",
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    if result["city_id"].null_count() > 0:
        raise ValueError(
            "tokenizer places index must provide city_id for every POI"
        )
    return result


def publish_tokenizer_artifacts(
    cfg: ExperimentConfig,
    artifacts: TokenizerArtifacts,
    *,
    model: Any,
    prepared: Any,
    history: list[Any],
) -> Path | None:
    import torch

    from semoflow.tokenization.decoding.codebooks import SIDCodebookBundle

    tcfg = cfg.tokenizer
    stays = prepared.stays
    aug_path = artifacts.augmented_processed_visits

    codes = encode_feature_tensors(model, prepared.tensors, device=tcfg.device)
    id_cols = [
        c
        for c in ("placekey_id", "placekey")
        if c in prepared.poi_features.columns
    ]
    if "placekey_id" not in id_cols:
        raise ValueError("tokenizer POI features must contain placekey_id")
    codes_df = _attach_catalog_city_ids(
        prepared.poi_features.select(id_cols), prepared.places_index
    )
    for i in range(codes.shape[1]):
        codes_df = codes_df.with_columns(
            pl.Series(f"sem_code_{i}", codes[:, i])
        )
    code_cols = [c for c in codes_df.columns if c.startswith("sem_code_")]
    bundle = SIDCodebookBundle.from_tokenizer(model, code_fields=code_cols)

    codes_df.write_parquet(artifacts.placekey_lookup)
    write_grounding_artifacts(
        codes_df,
        prepared.places_index,
        code_cols=code_cols,
        grounding_index_path=artifacts.grounding_index,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": prepared.feature_cols,
            "n_codebooks": model.n_codebooks,
            "codebook_size": model.codebook_size,
            "input_dim": len(prepared.feature_cols),
            "hidden_dim": tcfg.hidden_dim,
            "latent_dim": tcfg.latent_dim,
            "history": history,
            "config": vars(tcfg),
            "codebooks": bundle.codebooks,
            "code_fields": bundle.code_fields,
            "feature_mean": prepared.feature_mean,
            "feature_std": prepared.feature_std,
            "feature_preprocessing": prepared.feature_preprocessing,
        },
        artifacts.tokenizer_model,
    )
    torch.save(
        {
            "codebooks": bundle.codebooks,
            "code_fields": bundle.code_fields,
        },
        artifacts.tokenizer_codebooks,
    )

    stays_aug = stays.join(
        codes_df.select(["placekey_id", *code_cols]),
        on="placekey_id",
        how="left",
        maintain_order="left",
    )
    for c in code_cols:
        stays_aug = stays_aug.with_columns(
            pl.col(c).fill_null(0).cast(pl.Int32)
        )
    stays_aug.write_parquet(aug_path)
    logger.info(
        "tokenizer training complete: %s  codebooks=%s",
        aug_path,
        artifacts.tokenizer_codebooks,
    )
    artifacts.mark_complete()
    return aug_path


def encode_feature_tensors(
    model: Any,
    tensors: dict[str, Any],
    *,
    device: str,
) -> Any:
    import torch

    x = tensors["x"]
    parts: list[np.ndarray] = []
    model.to(torch.device(device))
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], 2048):
            batch = {
                k: v[start : start + 2048].to(device)
                for k, v in tensors.items()
            }
            parts.append(model.encode(batch).cpu().numpy())
    return np.concatenate(parts, axis=0).astype("int32")


def write_grounding_artifacts(
    codes_df: pl.DataFrame,
    metadata_frame: pl.DataFrame,
    *,
    code_cols: list[str],
    grounding_index_path: Path,
) -> Any:
    from semoflow.tokenization.decoding.grounding_index import (
        SemanticIDGroundingIndex,
    )

    payload_cols = [
        c
        for c in [
            "placekey_id",
            "placekey",
            "top_category",
            "naics_code",
            "latitude",
            "longitude",
        ]
        if c in metadata_frame.columns
    ]
    id_column = (
        "placekey_id" if "placekey_id" in codes_df.columns else "placekey"
    )
    grounding_frame = codes_df.join(
        metadata_frame.select(payload_cols).unique(id_column, keep="first"),
        on=id_column,
        how="left",
        maintain_order="left",
    )
    grounding_index = SemanticIDGroundingIndex.from_frame(
        grounding_frame,
        key_columns=code_cols,
        id_column=id_column,
        payload_columns=payload_cols,
    )
    grounding_index.save(grounding_index_path)
    return grounding_index
