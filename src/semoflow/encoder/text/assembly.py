from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch

from semoflow.encoder.artifacts import (
    load_poi_embedding_artifact,
    write_poi_embedding_artifact,
)


@dataclass(frozen=True)
class TextAssemblyResult:
    output_path: Path
    rows: int
    embedding_dim: int
    identity: str

    def to_dict(self) -> dict[str, object]:
        return {
            "output": str(self.output_path),
            "shape": [self.rows, self.embedding_dim],
            "identity": self.identity,
        }


def assemble_text_embeddings(
    places_index_path: Path,
    sources: Mapping[int, Path],
    output_path: Path,
) -> TextAssemblyResult:
    if not sources:
        raise ValueError("at least one city text source is required")
    places = pl.read_parquet(places_index_path).with_row_index("_target_row")
    required = {"placekey", "placekey_id", "city_id"}
    missing = required - set(places.columns)
    if missing:
        raise ValueError(f"places index is missing columns: {sorted(missing)}")
    if places["placekey"].n_unique() != places.height:
        raise ValueError(
            "joint places index contains duplicate placekey values"
        )

    observed_ids = set(int(value) for value in places["city_id"].unique())
    declared_ids = set(int(value) for value in sources)
    if declared_ids != observed_ids:
        raise ValueError(
            "source city_ids must exactly match the joint places index: "
            f"declared={sorted(declared_ids)} observed={sorted(observed_ids)}"
        )

    merged: torch.Tensor | None = None
    records: list[dict[str, object]] = []
    common_metadata: dict[str, object] | None = None
    covered = 0
    for city_id, path in sorted(sources.items()):
        artifact = load_poi_embedding_artifact(path, kind="text")
        if "placekey" not in artifact.index.columns:
            raise ValueError(
                f"city {city_id}: cross-catalog assembly requires a stable "
                "placekey sidecar; regenerate the source text artifact"
            )
        if artifact.metadata is None:
            raise ValueError(
                f"city {city_id}: text metadata sidecar is missing"
            )
        if artifact.metadata.get("include_address_in_template") is not False:
            raise ValueError(
                f"city {city_id}: source is not declared address-free"
            )
        source_metadata = {
            key: artifact.metadata.get(key)
            for key in (
                "model",
                "precision",
                "include_address_in_template",
                "include_name_in_template",
            )
        }
        if common_metadata is None:
            common_metadata = source_metadata
        elif source_metadata != common_metadata:
            raise ValueError(
                f"city {city_id}: text encoder metadata does not match the "
                f"other sources: {source_metadata} != {common_metadata}"
            )
        target = places.filter(pl.col("city_id") == city_id)
        aligned = artifact.align_to(target)
        if artifact.index.height != target.height:
            raise ValueError(
                f"city {city_id}: source rows={artifact.index.height} "
                f"target rows={target.height}"
            )
        if merged is None:
            merged = torch.empty(
                (places.height, aligned.shape[1]), dtype=torch.float32
            )
        elif aligned.shape[1] != merged.shape[1]:
            raise ValueError(
                f"city {city_id}: embedding width {aligned.shape[1]} does not "
                f"match {merged.shape[1]}"
            )
        rows = torch.from_numpy(
            target["_target_row"].cast(pl.Int64).to_numpy().copy()
        )
        merged.index_copy_(0, rows, aligned)
        covered += target.height
        records.append(
            {"city_id": city_id, "path": str(path), "rows": target.height}
        )

    if merged is None or common_metadata is None or covered != places.height:
        raise ValueError(f"coverage mismatch: {covered}/{places.height}")
    write_poi_embedding_artifact(
        output_path,
        merged,
        places.drop("_target_row"),
        kind="text",
        metadata={
            **common_metadata,
            "assembly_identity": "placekey",
            "sources": records,
        },
    )
    check = load_poi_embedding_artifact(output_path, kind="text")
    check.align_to(places.drop("_target_row"))
    return TextAssemblyResult(
        output_path=output_path,
        rows=int(check.embeddings.shape[0]),
        embedding_dim=int(check.embeddings.shape[1]),
        identity=check.identity_column,
    )


__all__ = ["TextAssemblyResult", "assemble_text_embeddings"]
