from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

from semoflow.utils.serialization import write_json

_SIDECAR_BY_ARTIFACT_NAME: dict[str, tuple[str, ...]] = {
    "poi_graph_emb.pt": ("places_index.parquet",),
    "cf_embeddings.pt": ("cf_index.parquet",),
    "text_embeddings.npy": ("places_index.parquet", "placekey_lookup.parquet"),
}


@dataclass(frozen=True)
class POIEmbeddingArtifact:
    embeddings: torch.Tensor
    index: pl.DataFrame
    kind: str = "unknown"
    metadata: dict[str, object] | None = None

    @property
    def identity_column(self) -> str:
        for column in ("placekey", "placekey_id"):
            if column in self.index.columns:
                return column
        raise ValueError(
            "embedding index must contain placekey_id or placekey"
        )

    def align_to(self, places_index: pl.DataFrame) -> torch.Tensor:
        key = self.identity_column
        if key not in places_index.columns:
            raise ValueError(
                f"target POI index has no identity column {key!r}"
            )
        target = places_index.select(pl.col(key)).with_row_index("target_row")
        if target[key].n_unique() != target.height:
            raise ValueError(f"target POI index has duplicate {key} values")
        rows = self.index.select([key, "embedding_row"])
        joined = target.join(rows, on=key, how="left")
        if joined["embedding_row"].null_count() > 0:
            missing = joined.filter(pl.col("embedding_row").is_null()).height
            raise ValueError(
                f"embedding artifact is missing {missing} target POI identities"
            )
        order = joined.sort("target_row")["embedding_row"].cast(pl.Int64)
        return self.embeddings.index_select(
            0, torch.from_numpy(order.to_numpy().copy())
        )


def _embedding_index_path(path: Path) -> Path:
    return path.with_suffix(".index.parquet")


def write_poi_embedding_artifact(
    path: str | Path,
    embeddings: torch.Tensor,
    places_index: pl.DataFrame,
    *,
    kind: str,
    metadata: dict[str, object] | None = None,
) -> POIEmbeddingArtifact:
    output = Path(path)
    keys = [
        key
        for key in ("placekey", "placekey_id")
        if key in places_index.columns
    ]
    index = places_index.select(keys).with_row_index("embedding_row")
    artifact = POIEmbeddingArtifact(
        embeddings.detach().cpu().float(), index, kind, metadata
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".npy":
        np.save(output, artifact.embeddings.numpy())
    else:
        torch.save(artifact.embeddings, output)
    index.write_parquet(_embedding_index_path(output))
    if metadata is not None:
        write_json(
            _embedding_index_path(output).with_suffix(".json"), metadata
        )
    return artifact


def _sidecar_path(output: Path, index_path: str | Path | None) -> Path:
    if index_path is not None:
        return Path(index_path)
    sidecar = _embedding_index_path(output)
    if sidecar.exists():
        return sidecar
    for name in _SIDECAR_BY_ARTIFACT_NAME.get(output.name, ()):
        if (output.parent / name).exists():
            return output.parent / name
    raise FileNotFoundError(
        f"identity sidecar missing for {output}; expected {sidecar}"
    )


def _load_matrix(output: Path) -> torch.Tensor:
    if output.suffix == ".npy":
        return torch.from_numpy(np.asarray(np.load(output), dtype=np.float32))
    payload = torch.load(output, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        payload = next(
            (
                payload[key]
                for key in ("embeddings", "poi_graph_emb", "weight")
                if key in payload
            ),
            None,
        )
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"cannot extract POI embeddings from {output}")
    return payload


def load_poi_embedding_artifact(
    path: str | Path,
    *,
    index_path: str | Path | None = None,
    kind: str = "unknown",
) -> POIEmbeddingArtifact:
    output = Path(path)
    embeddings = _load_matrix(output).detach().cpu().float()
    sidecar = _sidecar_path(output, index_path)
    index = pl.read_parquet(sidecar)
    if "embedding_row" not in index.columns:
        index = index.with_row_index("embedding_row")
    metadata_path = sidecar.with_suffix(".json")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else None
    )
    return POIEmbeddingArtifact(embeddings, index, kind, metadata)


__all__ = [
    "POIEmbeddingArtifact",
    "load_poi_embedding_artifact",
    "write_poi_embedding_artifact",
]
