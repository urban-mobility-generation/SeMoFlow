from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

__all__ = ["load_places_index", "write_places_index"]


def _with_city_id(
    places: pl.DataFrame,
    *,
    city_id: int,
) -> pl.DataFrame:
    if "city_id" in places.columns:
        return places.with_columns(pl.col("city_id").cast(pl.Int32))
    return places.with_columns(
        pl.lit(city_id, dtype=pl.Int32).alias("city_id")
    )


def write_places_index(
    places_by_city: list[tuple[str, int, pl.DataFrame]],
    normalized_events: pl.DataFrame,
    *,
    path: Path,
) -> pl.DataFrame:
    place_parts = [
        _with_city_id(places, city_id=city_id)
        for _city, city_id, places in places_by_city
    ]
    places_all = pl.concat(place_parts, how="diagonal_relaxed")

    poi_columns = [
        column
        for column in ("placekey", "city_id")
        if column in normalized_events.columns
    ]
    poi_ids = normalized_events.select(poi_columns).unique(
        subset=["placekey"], keep="first", maintain_order=True
    )

    places_index = places_all.join(
        poi_ids,
        on="placekey",
        how="left",
        suffix="_event",
    ).unique(subset=["placekey"], keep="first", maintain_order=True)
    if "city_id_event" in places_index.columns:
        places_index = places_index.with_columns(
            pl.coalesce("city_id_event", "city_id")
            .cast(pl.Int32)
            .alias("city_id")
        ).drop("city_id_event")

    places_index = places_index.with_columns(
        pl.int_range(1, pl.len() + 1).cast(pl.Int32).alias("placekey_id")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    places_index.write_parquet(path)
    logger.info(
        "Places index written  rows=%s  path=%s", places_index.height, path
    )
    return places_index


def load_places_index(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Places index not found: {path}")
    return pl.read_parquet(path)
