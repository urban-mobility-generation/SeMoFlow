from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

from semoflow.encoder.artifacts import (
    load_poi_embedding_artifact,
    write_poi_embedding_artifact,
)

if TYPE_CHECKING:
    from semoflow.config import TextEncoderConfig

logger = logging.getLogger(__name__)


def _parse_devices(device: str) -> list[str]:
    match = re.fullmatch(r"\[(.+)\]", device.strip())
    if match is None:
        return [device.strip()]
    return [
        f"cuda:{item}" if item.isdigit() else item
        for item in (part.strip() for part in match.group(1).split(","))
    ]


def render_poi_texts(
    places: pl.DataFrame,
    *,
    include_address: bool = True,
    include_name: bool = True,
) -> list[str]:
    texts = []
    for row in places.iter_rows(named=True):
        parts = []
        if include_name:
            name = str(row.get("location_name") or "").strip()
            if name:
                parts.append(name)
            brand = str(row.get("brands") or "").strip()
            if brand and brand.lower() != name.lower():
                parts.append(f"({brand})")
        if include_address:
            addr = ", ".join(
                b
                for b in [
                    str(row.get(c) or "").strip()
                    for c in ("street_address", "city", "state", "zipcode")
                ]
                if b
            )
            if addr:
                parts.append(addr)
        cats = " > ".join(
            b
            for b in [
                str(row.get(c) or "").strip()
                for c in ("top_category", "sub_category")
            ]
            if b
        )
        naics = str(row.get("naics_code") or "").strip()
        if naics and naics not in ("", "000000"):
            cats += f" (NAICS {naics})"
        if cats:
            parts.append(cats)
        texts.append(" | ".join(parts) if parts else "unknown location")
    return texts


def encode_texts(
    texts: list[str],
    model_name: str,
    device: str,
    batch_size: int,
    precision: str = "float32",
) -> np.ndarray:
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if precision not in dtypes:
        raise ValueError(
            "text precision must be float32, float16, or bfloat16"
        )
    devices = _parse_devices(device)
    multi_gpu = len(devices) > 1

    if multi_gpu:
        logger.info(
            "multi-GPU text encoding  devices=%s  texts=%d",
            devices,
            len(texts),
        )
        encoder = SentenceTransformer(model_name, device="cpu").to(
            dtype=dtypes[precision]
        )
        pool = encoder.start_multi_process_pool(target_devices=devices)
        try:
            emb = encoder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                pool=pool,
            )
        finally:
            encoder.stop_multi_process_pool(pool)
        return np.asarray(emb, dtype=np.float32)

    encoder = SentenceTransformer(model_name, device=devices[0]).to(
        dtype=dtypes[precision]
    )
    return np.asarray(
        encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )


def build_text_embeddings(
    places: pl.DataFrame, config: TextEncoderConfig, *, path: Path
) -> np.ndarray:
    if path.exists():
        try:
            cached = load_poi_embedding_artifact(path, kind="text").align_to(
                places
            )
            return cached.numpy().astype(np.float32, copy=False)
        except (FileNotFoundError, ValueError):
            logger.info(
                "text embedding cache identity mismatch; rebuilding path=%s",
                path,
            )

    include_address = config.include_address_in_template
    include_name = config.include_name_in_template
    texts = render_poi_texts(
        places, include_address=include_address, include_name=include_name
    )
    embeddings = encode_texts(
        texts,
        model_name=config.model,
        device=config.device,
        batch_size=int(config.batch_size),
        precision=config.precision,
    )
    write_poi_embedding_artifact(
        path,
        torch.from_numpy(embeddings.astype(np.float32, copy=False)),
        places,
        kind="text",
        metadata={
            "model": config.model,
            "precision": config.precision,
            "include_address_in_template": include_address,
            "include_name_in_template": include_name,
        },
    )
    return embeddings
