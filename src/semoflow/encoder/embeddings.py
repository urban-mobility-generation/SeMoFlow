"""POI embedding matrices with a ``<name>.index.parquet`` sidecar that maps
each row to its ``placekey_id``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch

from semoflow.utils.serialization import write_json


def _index_path(path: Path) -> Path:
    return path.with_suffix(".index.parquet")


def write_poi_embeddings(
    path: Path,
    embeddings: torch.Tensor,
    placekey_ids: pl.Series,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    """Save ``embeddings`` (``.npy`` or a torch file) whose rows belong to
    ``placekey_ids``, plus its index sidecar and optional metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = embeddings.detach().cpu().float()
    if path.suffix == ".npy":
        np.save(path, embeddings.numpy())
    else:
        torch.save(embeddings, path)
    pl.DataFrame({"placekey_id": placekey_ids}).with_row_index(
        "embedding_row"
    ).write_parquet(_index_path(path))
    if metadata is not None:
        write_json(_index_path(path).with_suffix(".json"), metadata)


def load_poi_embeddings(
    path: Path, places_index: pl.DataFrame
) -> torch.Tensor:
    """The embedding rows of ``path`` in ``places_index`` row order."""
    path = Path(path)
    if path.suffix == ".npy":
        matrix = torch.from_numpy(np.load(path).astype(np.float32, copy=False))
    else:
        matrix = torch.load(path, map_location="cpu").float()
    index = pl.read_parquet(_index_path(path))
    rows = places_index.select("placekey_id").join(
        index.select("placekey_id", "embedding_row"),
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


__all__ = ["load_poi_embeddings", "write_poi_embeddings"]
