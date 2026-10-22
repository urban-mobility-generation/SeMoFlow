from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from semoflow.tokenization.control_assignment import (
    frequency_codes,
    suffix_codes,
)
from semoflow.utils.geohash import geohash_center

logger = logging.getLogger(__name__)

KM_PER_DEGREE_LATITUDE = 110.574
KM_PER_DEGREE_LONGITUDE = 111.320


@dataclass(frozen=True)
class CategoryRelativeGeoResult:
    source_dir: str
    output_dir: str
    seed: int
    n_pois: int
    n_levels: int
    n_full_sids: int
    n_prefix_buckets: int
    pooled_category_share: float
    branching: int
    n_cities: int
    n_geo_cells: int
    n_geo_codes: int
    geo_code_ceiling: int
    prefix_bucket_ceiling: int
    median_offset_km: float
    prefix_bucket_median: float
    prefix_bucket_max: int
    prefix_singleton_fraction: float
    full_sid_collisions: int
    source_lookup_fingerprint: str
    lookup_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, cumulative[-1] / 2.0))
    return float(values[order[min(index, len(order) - 1)]])


def _nested_cell_codes(
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    branching: int,
) -> np.ndarray:
    codes = np.zeros(len(dx), dtype=np.int32)
    tiebreak = np.arange(len(dx))
    order = np.lexsort((tiebreak, dy, dx))
    if len(dx) <= branching * branching:
        codes[order] = np.arange(len(dx), dtype=np.int32)
        return codes
    strip_order = np.lexsort((tiebreak, dx))
    for strip, members in enumerate(np.array_split(strip_order, branching)):
        if members.size == 0:
            continue
        inner = members[np.lexsort((tiebreak[members], dy[members]))]
        for index, bins in enumerate(np.array_split(inner, branching)):
            codes[bins] = strip * branching + index
    return codes


def _cell_offsets(
    places: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    cells = places["geohash_5"].to_list()
    labels = sorted(set(cells))
    index = {label: position for position, label in enumerate(labels)}
    cell_of_poi = np.asarray([index[value] for value in cells], dtype=np.int64)
    centers = np.asarray([geohash_center(label) for label in labels])

    if "city_id" in places.columns:
        city_of_poi = places["city_id"].cast(pl.Int64).to_numpy()
    else:
        city_of_poi = np.zeros(places.height, dtype=np.int64)
    city_of_cell = np.zeros(len(labels), dtype=np.int64)
    city_of_cell[cell_of_poi] = city_of_poi
    if not np.array_equal(city_of_cell[cell_of_poi], city_of_poi):
        raise ValueError(
            "a geohash_5 cell spans more than one city_id; relative offsets "
            "need each cell to belong to exactly one city"
        )

    weights = np.bincount(cell_of_poi, minlength=len(labels)).astype(
        np.float64
    )
    dx = np.zeros(len(labels))
    dy = np.zeros(len(labels))
    for city in np.unique(city_of_cell):
        members = np.flatnonzero(city_of_cell == city)
        latitude0 = _weighted_median(centers[members, 0], weights[members])
        longitude0 = _weighted_median(centers[members, 1], weights[members])
        dy[members] = (
            centers[members, 0] - latitude0
        ) * KM_PER_DEGREE_LATITUDE
        dx[members] = (
            (centers[members, 1] - longitude0)
            * KM_PER_DEGREE_LONGITUDE
            * math.cos(math.radians(latitude0))
        )
    return cell_of_poi, city_of_cell, dx, dy, len(labels)


def build_category_relative_geo_lookup(
    places: pl.DataFrame,
    features: np.ndarray,
    *,
    codebook_size: int,
    n_levels: int,
    seed: int,
    branching: int | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    required = {"placekey_id", "top_category", "geohash_5"}
    missing = sorted(required - set(places.columns))
    if missing:
        raise ValueError(f"places index is missing columns: {missing}")
    if len(features) != places.height:
        raise ValueError("features and places rows disagree")
    if n_levels < 4:
        raise ValueError(
            "need at least one suffix level below a depth-3 prefix"
        )
    if places["geohash_5"].null_count() or (places["geohash_5"] == "").any():
        raise ValueError("every POI needs a geohash_5 cell for this baseline")
    if places["geohash_5"].str.len_chars().min() != 5:
        raise ValueError("geohash_5 values must all be 5 characters")

    if branching is None:
        branching = math.isqrt(codebook_size)
    if branching < 2:
        raise ValueError("branching must be at least 2")
    if branching * branching > codebook_size:
        raise ValueError(
            f"a {branching}x{branching} grid needs {branching * branching} "
            f"codes but the codebook holds {codebook_size}"
        )

    category_codes, pooled_category = frequency_codes(
        places["top_category"].fill_null("__missing__").to_list(),
        codebook_size=codebook_size,
    )
    cell_of_poi, city_of_cell, dx, dy, n_cells = _cell_offsets(places)

    coarse = np.zeros(n_cells, dtype=np.int32)
    for city in np.unique(city_of_cell):
        members = np.flatnonzero(city_of_cell == city)
        coarse[members] = _nested_cell_codes(
            dx[members], dy[members], branching=branching
        )
    fine = np.zeros(n_cells, dtype=np.int32)
    for city in np.unique(city_of_cell):
        for code in range(branching * branching):
            members = np.flatnonzero((city_of_cell == city) & (coarse == code))
            if members.size == 0:
                continue
            fine[members] = _nested_cell_codes(
                dx[members], dy[members], branching=branching
            )

    prefix = np.stack(
        [category_codes, coarse[cell_of_poi], fine[cell_of_poi]], axis=1
    )
    _, groups, counts = np.unique(
        prefix, axis=0, return_inverse=True, return_counts=True
    )
    groups = groups.reshape(-1)
    suffix, collisions = suffix_codes(
        features,
        groups,
        codebook_size=codebook_size,
        suffix_levels=n_levels - 3,
        seed=seed,
    )
    codes = np.concatenate([prefix, suffix], axis=1)
    lookup = places.select(
        [c for c in ("placekey_id", "placekey") if c in places.columns]
    ).with_columns(
        [
            pl.Series(f"sem_code_{level}", codes[:, level])
            for level in range(n_levels)
        ]
    )

    n_geo_codes = int(
        np.unique(np.stack([coarse, fine], axis=1), axis=0).shape[0]
    )
    geo_code_ceiling = min(branching**4, n_cells)
    diagnostics = {
        "pooled_category_share": pooled_category,
        "branching": branching,
        "n_cities": len(np.unique(city_of_cell)),
        "n_geo_cells": n_cells,
        "n_geo_codes": n_geo_codes,
        "geo_code_ceiling": geo_code_ceiling,
        "prefix_bucket_ceiling": codebook_size * geo_code_ceiling,
        "median_offset_km": float(
            np.median(np.hypot(dx[cell_of_poi], dy[cell_of_poi]))
        ),
        "n_prefix_buckets": len(counts),
        "prefix_bucket_median": float(np.median(counts)),
        "prefix_bucket_max": int(counts.max()),
        "prefix_singleton_fraction": float((counts == 1).mean()),
        "full_sid_collisions": int(collisions),
    }
    logger.info(
        "category+relative-geo prefix: %d buckets (ceiling %d), %d geo codes "
        "over %d cells, median %.1f, max %d, singletons %.3f",
        diagnostics["n_prefix_buckets"],
        diagnostics["prefix_bucket_ceiling"],
        n_geo_codes,
        n_cells,
        diagnostics["prefix_bucket_median"],
        diagnostics["prefix_bucket_max"],
        diagnostics["prefix_singleton_fraction"],
    )
    return lookup, diagnostics


def write_category_relative_geo_tokenizer_dir(
    *,
    source_dir: Path,
    output_dir: Path,
    places_index: Path,
    checkpoint: Path,
    text_embeddings: Path | None = None,
    graphsage_artifact_dir: Path | None = None,
    city_ids: list[int] | None = None,
    seed: int,
    branching: int | None = None,
) -> CategoryRelativeGeoResult:
    from semoflow.tokenization.control_dir import (
        write_control_tokenizer_dir,
    )
    from semoflow.tokenization.frozen_encode import (
        load_frozen_features,
    )
    from semoflow.tokenization.token_space import (
        sid_code_columns,
    )

    source_lookup = pl.read_parquet(source_dir / "placekey_lookup.parquet")
    n_levels = len(sid_code_columns(source_lookup.columns))
    frozen = load_frozen_features(
        checkpoint=checkpoint,
        places_index=places_index,
        city_ids=city_ids,
        text_embeddings=text_embeddings,
        graphsage_artifact_dir=graphsage_artifact_dir,
    )
    codebook_size = int(frozen.payload["codebook_size"])
    places = frozen.places
    features = frozen.x.detach().cpu().numpy()

    keep = set(source_lookup["placekey_id"].to_list())
    mask = np.asarray(
        [int(value) in keep for value in places["placekey_id"].to_list()],
        dtype=np.bool_,
    )
    if not mask.all():
        logger.info(
            "restricting to the source lookup's %d POIs (%d dropped)",
            int(mask.sum()),
            int((~mask).sum()),
        )
        places = places.filter(pl.Series(mask))
        features = features[mask]

    lookup, diagnostics = build_category_relative_geo_lookup(
        places,
        features,
        codebook_size=codebook_size,
        n_levels=n_levels,
        seed=seed,
        branching=branching,
    )
    manifest = write_control_tokenizer_dir(
        source_dir=source_dir,
        output_dir=output_dir,
        lookup=lookup,
        control={
            "control": "category_relative_geo_assignment",
            "seed": seed,
            "prefix": (
                "top_category / city-relative equal-count cell / "
                "equal-count sub-cell"
            ),
            "suffix": "fit-space k-means inside each prefix bucket",
            "checkpoint": str(checkpoint),
        },
        manifest_extra=diagnostics,
    )
    return CategoryRelativeGeoResult(
        source_dir=manifest["source_dir"],
        output_dir=manifest["output_dir"],
        seed=seed,
        n_pois=manifest["n_pois"],
        n_levels=manifest["n_levels"],
        n_full_sids=manifest["n_full_sids"],
        source_lookup_fingerprint=manifest["source_lookup_fingerprint"],
        lookup_fingerprint=manifest["lookup_fingerprint"],
        **diagnostics,
    )


__all__ = [
    "CategoryRelativeGeoResult",
    "build_category_relative_geo_lookup",
    "write_category_relative_geo_tokenizer_dir",
]
