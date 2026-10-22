from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import polars as pl

from semoflow.tokenization.decoding.grounding_index import (
    SemanticIDGroundingIndex,
)

_CODE_COLUMNS = tuple(f"sem_code_{level}" for level in range(6))
_LOOKUP_COLUMNS = ("placekey_id", "placekey", "city_id", *_CODE_COLUMNS)
_PLACES_COLUMNS = (
    "placekey_id",
    "placekey",
    "city_id",
    "latitude",
    "longitude",
    "geohash_5",
)
_GROUNDING_METADATA = ("placekey", "top_category")


def refuse_existing(*paths: Path, force: bool) -> None:
    if force:
        return
    for path in paths:
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing artifact: {path}"
            )


def write_sidecar_manifest(output: Path, manifest: dict[str, Any]) -> None:
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_union_target(union_manifest: Path) -> tuple[int, int]:
    manifest = json.loads(union_manifest.read_text(encoding="utf-8"))
    offset = int(manifest["target_placekey_id_offset"])
    target_city_ids = manifest["target_city_ids"]
    if len(target_city_ids) != 1:
        raise ValueError(
            f"expected exactly one target city id, got {target_city_ids}"
        )
    return offset, int(target_city_ids[0])


def _load_columns(
    path: Path, columns: Sequence[str], label: str
) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns {missing}")
    return frame.select(columns)


def build_union_lookup(
    *,
    source_lookup: Path,
    target_lookup: Path,
    output: Path,
    force: bool = False,
) -> dict[str, Any]:
    refuse_existing(output, force=force)
    source = _load_columns(source_lookup, _LOOKUP_COLUMNS, "source lookup")
    target = _load_columns(target_lookup, _LOOKUP_COLUMNS, "target lookup")

    source_cities = sorted(
        int(v) for v in source.get_column("city_id").unique()
    )
    target_cities = sorted(
        int(v) for v in target.get_column("city_id").unique()
    )
    overlap = sorted(set(source_cities) & set(target_cities))
    if overlap:
        raise ValueError(
            f"source and target share city ids {overlap}; the target must be "
            "renumbered to an unused id before the catalogs are unioned"
        )

    shared = source.join(target, on="placekey", how="inner")
    if shared.height:
        raise ValueError(
            f"{shared.height} placekeys appear in both catalogs; a POI cannot "
            "hold two catalog rows with different codes"
        )

    offset = int(cast(int, source.get_column("placekey_id").max()))
    shifted = target.with_columns(
        (pl.col("placekey_id") + offset).alias("placekey_id")
    )
    union = pl.concat([source, shifted], how="vertical").sort("placekey_id")
    if union.get_column("placekey_id").n_unique() != union.height:
        raise ValueError("union lookup has duplicate placekey_id values")

    output.parent.mkdir(parents=True, exist_ok=True)
    union.write_parquet(output)
    manifest = {
        "artifact_kind": "union_placekey_lookup",
        "source_lookup": str(source_lookup),
        "target_lookup": str(target_lookup),
        "source_rows": source.height,
        "target_rows": target.height,
        "union_rows": union.height,
        "source_city_ids": source_cities,
        "target_city_ids": target_cities,
        "catalog_city_ids": sorted(source_cities + target_cities),
        "target_placekey_id_offset": offset,
        "output": str(output),
    }
    write_sidecar_manifest(output, manifest)
    return manifest


def build_union_places_index(
    *,
    source_places: Path,
    target_places: Path,
    union_lookup: Path,
    union_manifest: Path,
    output: Path,
    force: bool = False,
) -> dict[str, Any]:
    refuse_existing(output, force=force)
    offset, target_city_id = read_union_target(union_manifest)

    source = _load_columns(
        source_places, _PLACES_COLUMNS, "source places index"
    )
    target = _load_columns(
        target_places, _PLACES_COLUMNS, "target places index"
    ).with_columns(
        (pl.col("placekey_id") + offset).alias("placekey_id"),
        pl.lit(target_city_id, dtype=pl.Int32).alias("city_id"),
    )
    union = pl.concat([source, target], how="vertical").sort("placekey_id")
    if union.get_column("placekey_id").n_unique() != union.height:
        raise ValueError("union places index has duplicate placekey_id values")

    lookup = pl.read_parquet(
        union_lookup, columns=["placekey_id", "placekey", "city_id"]
    )
    joined = lookup.join(union, on="placekey_id", how="left", suffix="_places")
    orphans = joined.filter(pl.col("placekey_places").is_null())
    if orphans.height:
        raise ValueError(
            f"{orphans.height} lookup rows have no places entry, e.g. "
            f"{orphans.get_column('placekey_id').head(3).to_list()}"
        )
    disagreeing = joined.filter(
        (pl.col("placekey") != pl.col("placekey_places"))
        | (pl.col("city_id") != pl.col("city_id_places"))
    )
    if disagreeing.height:
        raise ValueError(
            f"{disagreeing.height} rows disagree between the union lookup and "
            "the union places index on placekey or city_id"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    union.write_parquet(output)
    manifest = {
        "artifact_kind": "union_places_index",
        "source_places": str(source_places),
        "target_places": str(target_places),
        "union_lookup": str(union_lookup),
        "placekey_id_offset": offset,
        "target_city_id": target_city_id,
        "source_rows": source.height,
        "target_rows": target.height,
        "union_rows": union.height,
        "rows_with_coordinates": union.drop_nulls(
            ["latitude", "longitude"]
        ).height,
        "city_ids": sorted(
            int(v) for v in union.get_column("city_id").unique()
        ),
        "output": str(output),
    }
    write_sidecar_manifest(output, manifest)
    return manifest


def build_union_grounding_index(
    *,
    union_lookup: Path,
    places: Sequence[Path],
    output: Path,
    force: bool = False,
) -> dict[str, Any]:
    refuse_existing(output, force=force)
    if not places:
        raise ValueError("at least one places table is required")
    lookup = pl.read_parquet(union_lookup)
    if "placekey" not in lookup.columns:
        raise ValueError("union lookup must carry placekey")

    metadata = pl.concat(
        [
            _load_columns(path, _GROUNDING_METADATA, str(path))
            for path in places
        ],
        how="vertical",
    ).unique(subset=["placekey"])
    frame = lookup.join(metadata, on="placekey", how="left")
    unresolved = int(frame.get_column("top_category").is_null().sum())

    index = SemanticIDGroundingIndex.from_frame(frame)
    if index.n_keys == 0:
        raise ValueError("grounding index is empty")

    output.parent.mkdir(parents=True, exist_ok=True)
    index.save(output)
    manifest = {
        "artifact_kind": "union_grounding_index",
        "union_lookup": str(union_lookup),
        "places": [str(path) for path in places],
        "catalog_rows": frame.height,
        "distinct_keys": index.n_keys,
        "rows_without_top_category": unresolved,
        "output": str(output),
    }
    write_sidecar_manifest(output, manifest)
    return manifest


__all__ = [
    "build_union_grounding_index",
    "build_union_lookup",
    "build_union_places_index",
    "read_union_target",
    "refuse_existing",
    "write_sidecar_manifest",
]
