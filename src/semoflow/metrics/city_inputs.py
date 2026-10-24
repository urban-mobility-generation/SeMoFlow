"""One city's evaluation inputs, cut from joint multi-city artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from semoflow.preprocess.weekly_segments import SEGMENT_CITY_COLUMN
from semoflow.tokenization.decoding.grounding_index import (
    SemanticIDGroundingIndex,
)

SEGMENTS_NAME = "segments.parquet"
LOOKUP_NAME = "placekey_lookup.parquet"
GROUNDING_NAME = "grounding_index.pkl"


def extract_city_inputs(
    city_id: int,
    *,
    segments: Path,
    lookup: Path,
    grounding_index: Path,
    output_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Write ``city_id``'s segments, SID lookup, and grounding index.

    Each output keeps the joint artifact's rows, order, and identifiers, so
    the bank, candidate prior, and composition built from them are those of
    the joint run restricted to one city.
    """
    outputs = {
        name: output_dir / name
        for name in (SEGMENTS_NAME, LOOKUP_NAME, GROUNDING_NAME)
    }
    if not force:
        for path in outputs.values():
            if path.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing artifact: {path}"
                )

    city_lookup = pl.read_parquet(lookup).filter(pl.col("city_id") == city_id)
    city_segments = pl.read_parquet(segments).filter(
        pl.col(SEGMENT_CITY_COLUMN) == city_id
    )
    if city_lookup.is_empty() or city_segments.is_empty():
        raise ValueError(f"city {city_id} has no lookup rows or segments")
    joint = SemanticIDGroundingIndex.load(grounding_index)
    city_grounding = SemanticIDGroundingIndex.from_frame(
        joint.to_frame().filter(
            pl.col(joint.id_column).is_in(city_lookup["placekey_id"].implode())
        ),
        key_columns=joint.key_columns,
        id_column=joint.id_column,
        payload_columns=joint.payload_columns,
        order_agnostic=joint.order_agnostic,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    city_segments.write_parquet(outputs[SEGMENTS_NAME])
    city_lookup.write_parquet(outputs[LOOKUP_NAME])
    city_grounding.save(outputs[GROUNDING_NAME])
    return {
        "city_id": city_id,
        "segments": city_segments.height,
        "segments_by_split": dict(
            city_segments.group_by("_split").len().sort("_split").iter_rows()
        ),
        "lookup_rows": city_lookup.height,
        "grounding_keys": city_grounding.n_keys,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }


__all__ = ["extract_city_inputs"]
