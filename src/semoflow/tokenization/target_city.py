from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import polars as pl
import torch

from semoflow.utils.serialization import write_json


def derive_target_city_tokenizer(
    *,
    tokenizer: Path,
    places_index: Path,
    target_city_id: int,
    output_tokenizer: Path,
    output_places_index: Path,
    force: bool = False,
) -> dict[str, Any]:
    if not force:
        for path in (output_tokenizer, output_places_index):
            if path.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing artifact: {path}"
                )

    places = pl.read_parquet(places_index)
    for column in ("city_id", "latitude", "longitude"):
        if column not in places.columns:
            raise ValueError(f"places index is missing {column!r}")
    source_city_ids = sorted(
        int(value) for value in places.get_column("city_id").unique()
    )
    if len(source_city_ids) != 1:
        raise ValueError(
            "this derivation is for a single-city target catalog; got "
            f"city ids {source_city_ids}"
        )

    payload = torch.load(tokenizer, map_location="cpu", weights_only=False)
    preprocessing = payload.get("feature_preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("tokenizer has no feature_preprocessing mapping")
    pca = preprocessing.get("pca")
    if not isinstance(pca, dict) or not isinstance(
        pca.get("geo_centroids"), dict
    ):
        raise ValueError(
            "tokenizer has no fitted pca.geo_centroids; a city-relative geo "
            "block is what this derivation exists to fix"
        )
    centroids = pca["geo_centroids"]
    top_level = preprocessing.get("geo_centroids")
    if isinstance(top_level, dict) and top_level != centroids:
        raise ValueError(
            "tokenizer's two geo_centroids copies disagree; refusing to guess "
            "which one the encoder will read"
        )
    if target_city_id in centroids:
        raise ValueError(
            f"city id {target_city_id} already has a fitted centroid "
            f"{centroids[target_city_id]}; pick an unused ID rather than "
            "overwriting a source city's frame"
        )

    coordinates = places.drop_nulls(["latitude", "longitude"])
    if coordinates.is_empty():
        raise ValueError("target catalog has no usable coordinates")
    centroid = (
        cast(float, coordinates.get_column("latitude").mean()),
        cast(float, coordinates.get_column("longitude").mean()),
    )

    updated = dict(centroids) | {target_city_id: centroid}
    preprocessing = dict(preprocessing)
    preprocessing["pca"] = dict(pca) | {"geo_centroids": updated}
    if isinstance(top_level, dict):
        preprocessing["geo_centroids"] = updated
    payload["feature_preprocessing"] = preprocessing

    output_tokenizer.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_tokenizer)
    output_places_index.parent.mkdir(parents=True, exist_ok=True)
    places.with_columns(
        pl.lit(target_city_id, dtype=places.schema["city_id"]).alias("city_id")
    ).write_parquet(output_places_index)

    manifest = {
        "artifact_kind": "target_city_tokenizer_derivation",
        "source_tokenizer": str(tokenizer),
        "source_places_index": str(places_index),
        "renumbered_from_city_id": source_city_ids[0],
        "target_city_id": target_city_id,
        "target_centroid": list(centroid),
        "catalog_rows": places.height,
        "rows_with_coordinates": coordinates.height,
        "source_city_centroids": {
            str(key): list(value) for key, value in sorted(centroids.items())
        },
        "output_tokenizer": str(output_tokenizer),
        "output_places_index": str(output_places_index),
    }
    write_json(output_tokenizer.with_suffix(".manifest.json"), manifest)
    return manifest


__all__ = ["derive_target_city_tokenizer"]
