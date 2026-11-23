from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import torch
from sentence_transformers import SentenceTransformer

from semoflow.config import TextEncoderConfig
from semoflow.utils import parse_device_spec, write_json

logger = logging.getLogger(__name__)
_PRECISIONS = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _index_path(path: Path) -> Path:
    return path.with_suffix(".index.parquet")


def write_poi_embeddings(
    path: Path,
    embeddings: torch.Tensor,
    placekey_ids: pl.Series,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings.detach().cpu().float(), path)
    pl.DataFrame({"placekey_id": placekey_ids}).with_row_index(
        "embedding_row"
    ).write_parquet(_index_path(path))
    if metadata is not None:
        write_json(_index_path(path).with_suffix(".json"), metadata)


def load_poi_embeddings(
    path: Path, places_index: pl.DataFrame
) -> torch.Tensor:
    matrix = torch.load(path, map_location="cpu")
    rows = places_index.select("placekey_id").join(
        pl.read_parquet(_index_path(path)),
        on="placekey_id",
        how="left",
        maintain_order="left",
    )["embedding_row"]
    if rows.null_count():
        raise ValueError(
            f"{path} is missing {rows.null_count()} places-index POIs"
        )
    return matrix.index_select(
        0, torch.from_numpy(rows.cast(pl.Int64).to_numpy().copy())
    )


def _join(values: list[str], separator: str) -> str:
    return separator.join(value for value in values if value)


def render_poi_texts(
    places: pl.DataFrame, *, include_address: bool, include_name: bool
) -> list[str]:
    texts = []
    for row in places.iter_rows(named=True):
        field = {key: str(value or "").strip() for key, value in row.items()}
        parts = []
        if include_name:
            name, brand = (
                field.get("location_name", ""),
                field.get("brands", ""),
            )
            parts.append(name)
            if brand and brand.lower() != name.lower():
                parts.append(f"({brand})")
        if include_address:
            parts.append(
                _join(
                    [
                        field.get(column, "")
                        for column in (
                            "street_address",
                            "city",
                            "state",
                            "zipcode",
                        )
                    ],
                    ", ",
                )
            )
        category = _join(
            [field.get("top_category", ""), field.get("sub_category", "")],
            " > ",
        )
        naics = field.get("naics_code", "")
        if naics and naics != "000000":
            category += f" (NAICS {naics})"
        parts.append(category)
        texts.append(_join(parts, " | ") or "unknown location")
    return texts


def build_text_embeddings(
    places: pl.DataFrame, config: TextEncoderConfig, *, path: Path
) -> None:
    if path.exists():
        try:
            load_poi_embeddings(path, places)
            return
        except (FileNotFoundError, ValueError):
            logger.info("text embeddings do not cover the places; rebuilding")
    texts = render_poi_texts(
        places,
        include_address=config.include_address_in_template,
        include_name=config.include_name_in_template,
    )
    _, device_ids = parse_device_spec(config.device)
    devices = [f"cuda:{index}" for index in device_ids]
    encoder = SentenceTransformer(
        config.model, device="cpu" if devices else config.device
    ).to(dtype=_PRECISIONS[config.precision])
    embeddings = encoder.encode(
        texts,
        batch_size=config.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=True,
        device=devices or None,
    )
    write_poi_embeddings(
        path,
        embeddings,
        places["placekey_id"],
        metadata={
            "model": config.model,
            "precision": config.precision,
            "include_address_in_template": config.include_address_in_template,
            "include_name_in_template": config.include_name_in_template,
        },
    )


__all__ = [
    "build_text_embeddings",
    "load_poi_embeddings",
    "write_poi_embeddings",
]
