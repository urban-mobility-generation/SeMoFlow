from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import torch

from semoflow.config import CITY_ID
from semoflow.geodesy import unit_vectors
from semoflow.utils import write_json

if TYPE_CHECKING:
    from semoflow.config import ExperimentConfig

logger = logging.getLogger(__name__)
GLOBAL_SPLIT_SEED = 20260712
SPLIT_COLUMN = "split"


def stable_key(value: str) -> str:
    return hashlib.blake2b(value.encode(), digest_size=12).hexdigest()


def hash_fraction(value: str, seed: int) -> float:
    digest = hashlib.blake2b(
        f"{seed}:{value}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") / 2**64


def assign_split(key: str) -> str:
    fraction = hash_fraction(key, GLOBAL_SPLIT_SEED)
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "val"
    return "test"


SOURCE_URL = "https://sites.google.com/site/yangdingqi/home/foursquare-dataset"
CITY_NAMES = {
    "los_angeles": "Los Angeles",
    "washington_dc": "Washington D.C.",
    "atlanta": "Atlanta",
    "san_francisco": "San Francisco",
    "milwaukee": "Milwaukee",
}
CHECKINS_FILE = "dataset_TIST2015_Checkins.txt"
POIS_FILE = "dataset_TIST2015_POIs.txt"
CITIES_FILE = "dataset_TIST2015_Cities.txt"
_CHECKIN_COLUMNS = (
    "user_id",
    "venue_id",
    "timestamp_utc_raw",
    "timezone_offset_min",
)
_POI_COLUMNS = (
    "venue_id",
    "latitude",
    "longitude",
    "category_name",
    "country_code",
)
_CITY_COLUMNS = (
    "source_city_name",
    "latitude",
    "longitude",
    "country_code",
    "country_name",
    "city_type",
)
_PLACE_COLUMNS = (
    "placekey",
    "top_category",
    "census_block_group",
    "geohash_5",
    "latitude",
    "longitude",
    "source_venue_id",
    "city_id",
)
_TIMESTAMP_FORMAT = "%a %b %d %H:%M:%S %z %Y"
_GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"
_GEOHASH_PRECISION = 5
_CHUNK = 100_000


def _scan(
    source: Path, filename: str, columns: tuple[str, ...]
) -> pl.LazyFrame:
    path = source / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Foursquare file not found: {path}. Download and extract the "
            f"global release from {SOURCE_URL}"
        )
    return pl.scan_csv(
        path,
        has_header=False,
        separator="\t",
        new_columns=list(columns),
        schema_overrides={column: pl.Utf8 for column in columns},
        infer_schema_length=0,
        null_values=[""],
        quote_char=None,
    )


def _us_rows(frame: pl.LazyFrame) -> pl.DataFrame:
    rows = (
        frame.with_columns(
            pl.col("latitude").cast(pl.Float64, strict=False),
            pl.col("longitude").cast(pl.Float64, strict=False),
            pl.col("country_code").str.to_uppercase(),
        )
        .filter(pl.col("country_code") == "US")
        .collect()
    )
    return cast(pl.DataFrame, rows)


def _blank(column: str) -> pl.Expr:
    return pl.col(column).is_null() | (
        pl.col(column).str.strip_chars().str.len_chars() == 0
    )


def _bad_coordinates() -> pl.Expr:
    return (
        pl.col("latitude").is_null()
        | pl.col("longitude").is_null()
        | ~pl.col("latitude").is_between(-90.0, 90.0)
        | ~pl.col("longitude").is_between(-180.0, 180.0)
    )


def _unit_vectors(frame: pl.DataFrame) -> np.ndarray:
    return unit_vectors(
        torch.from_numpy(frame.select("latitude", "longitude").to_numpy())
    ).numpy()


def encode_geohash(latitude: float, longitude: float) -> str:
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    bits: list[int] = []
    even = True
    while len(bits) < _GEOHASH_PRECISION * 5:
        interval = lon_interval if even else lat_interval
        value = longitude if even else latitude
        midpoint = (interval[0] + interval[1]) / 2.0
        if value >= midpoint:
            bits.append(1)
            interval[0] = midpoint
        else:
            bits.append(0)
            interval[1] = midpoint
        even = not even
    return "".join(
        _GEOHASH_ALPHABET[
            sum(
                bit << (4 - offset)
                for offset, bit in enumerate(bits[start : start + 5])
            )
        ]
        for start in range(0, _GEOHASH_PRECISION * 5, 5)
    )


def read_catalog(source: Path, cities: list[str]) -> pl.DataFrame:
    centers = _us_rows(_scan(source, CITIES_FILE, _CITY_COLUMNS))
    if centers.is_empty() or centers.filter(_bad_coordinates()).height:
        raise ValueError("Foursquare city table has no valid US centers")
    names = [CITY_NAMES[city] for city in cities]
    missing = set(names) - set(centers["source_city_name"].to_list())
    if missing:
        raise ValueError(f"Foursquare city table lacks {sorted(missing)}")

    pois = _us_rows(_scan(source, POIS_FILE, _POI_COLUMNS))
    if pois.filter(_blank("venue_id") | _bad_coordinates()).height:
        raise ValueError(
            "Foursquare POI table has US rows without a venue or coordinates"
        )
    points = _unit_vectors(pois)
    center_points = _unit_vectors(centers)
    nearest = np.concatenate(
        [
            np.argmax(points[start : start + _CHUNK] @ center_points.T, axis=1)
            for start in range(0, len(points), _CHUNK)
        ]
        or [np.empty(0, dtype=np.int64)]
    )
    city_map = pl.DataFrame(
        {
            "source_city_name": names,
            "city": cities,
            "city_id": pl.Series(
                [CITY_ID[city] for city in cities], dtype=pl.Int32
            ),
        }
    )
    catalog = (
        pois.with_columns(
            pl.Series(
                "source_city_name",
                centers["source_city_name"].gather(nearest),
            )
        )
        .join(city_map, on="source_city_name", maintain_order="left")
        .drop("source_city_name")
    )
    if catalog.is_empty():
        raise ValueError("Foursquare POI table has no POIs in the cities")
    if (
        catalog.group_by("venue_id")
        .agg(pl.col("city").n_unique())
        .filter(pl.col("city") > 1)
        .height
    ):
        raise ValueError("Foursquare assigns a venue to several cities")
    return catalog.unique("venue_id", keep="first", maintain_order=True)


def read_checkins(source: Path, catalog: pl.DataFrame) -> pl.DataFrame:
    events = (
        _scan(source, CHECKINS_FILE, _CHECKIN_COLUMNS)
        .join(catalog.lazy(), on="venue_id", maintain_order="left")
        .with_columns(
            pl.col("timezone_offset_min").cast(pl.Int32, strict=False),
            pl.col("timestamp_utc_raw")
            .str.strptime(pl.Datetime, format=_TIMESTAMP_FORMAT, strict=False)
            .alias("timestamp_utc"),
        )
        .collect()
    )
    events = cast(pl.DataFrame, events)
    invalid = events.filter(
        _blank("user_id")
        | _blank("category_name")
        | pl.col("timestamp_utc").is_null()
        | ~pl.col("timezone_offset_min").is_between(-840, 840)
        | pl.col("timezone_offset_min").is_null()
    )
    if invalid.height:
        raise ValueError(
            f"Foursquare has {invalid.height} check-ins without a user, "
            "category, timestamp, or timezone offset"
        )
    events = events.with_columns(
        (pl.col("timestamp_utc") + pl.duration(minutes="timezone_offset_min"))
        .dt.replace_time_zone(None)
        .alias("timestamp"),
        pl.format("foursquare:{}:{}", "city", "venue_id").alias("placekey"),
        pl.format("foursquare:{}", "user_id").alias("caid"),
        pl.col("category_name").alias("top_category"),
    )
    users = (
        events.select("city", "caid")
        .unique()
        .with_columns(
            pl.struct("city", "caid")
            .map_elements(
                lambda row: stable_key(f"{row['city']}|{row['caid']}"),
                return_dtype=pl.Utf8,
            )
            .alias("device_key"),
            pl.col("caid")
            .map_elements(
                assign_split,
                return_dtype=pl.Utf8,
            )
            .alias(SPLIT_COLUMN),
        )
    )
    cells = (
        events.select("latitude", "longitude")
        .unique()
        .with_columns(
            pl.struct("latitude", "longitude")
            .map_elements(
                lambda row: encode_geohash(row["latitude"], row["longitude"]),
                return_dtype=pl.Utf8,
            )
            .alias("geohash_5")
        )
    )
    timestamp = pl.col("timestamp")
    return (
        events.join(users, on=["city", "caid"], maintain_order="left")
        .join(cells, on=["latitude", "longitude"], maintain_order="left")
        .with_columns(
            timestamp.dt.date().alias("local_day"),
            timestamp.dt.hour().cast(pl.Int32).alias("hour"),
            timestamp.dt.weekday().cast(pl.Int32).alias("weekday"),
            (timestamp.dt.weekday() >= 6).alias("is_weekend"),
            timestamp.dt.month().cast(pl.Int32).alias("month"),
        )
        .sort(["user_id", "timestamp_utc", "venue_id"], maintain_order=True)
    )


def place_index(events: pl.DataFrame) -> pl.DataFrame:
    return (
        events.unique("placekey", keep="first", maintain_order=True)
        .rename({"venue_id": "source_venue_id"})
        .with_columns(pl.lit(None, dtype=pl.Utf8).alias("census_block_group"))
        .select(_PLACE_COLUMNS)
        .with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("placekey_id")
        )
    )


def run_foursquare_pipeline(cfg: ExperimentConfig) -> None:
    cities = list(cfg.data.cities)
    unknown = sorted(set(cities) - set(CITY_NAMES))
    if unknown:
        raise ValueError(
            f"Foursquare has no cities {unknown}; expected {sorted(CITY_NAMES)}"
        )
    source = cfg.data.source_dir.resolve()
    artifacts = cfg.layout.preprocessing
    artifacts.run_dir.mkdir(parents=True, exist_ok=True)

    catalog = read_catalog(source, cities)
    checkins = read_checkins(source, catalog)
    events = pl.concat(
        [checkins.filter(pl.col("city") == city) for city in cities]
    )
    counts = dict(events["city"].value_counts().iter_rows())
    empty = [city for city in cities if city not in counts]
    if empty:
        raise ValueError(f"Foursquare has no check-ins for {empty}")

    places = place_index(events)
    places.write_parquet(artifacts.places_index)
    stays = events.join(
        places.select("placekey", "placekey_id"),
        on="placekey",
        maintain_order="left",
    ).drop("placekey")
    stays.write_parquet(artifacts.processed_visits)
    logger.info(
        "preprocessed  visits=%d  places=%d  path=%s",
        stays.height,
        places.height,
        artifacts.run_dir,
    )

    write_json(
        artifacts.run_dir / "foursquare_source_manifest.json",
        {
            "source_url": SOURCE_URL,
            "source_path": str(source),
            "source_files": [CHECKINS_FILE, POIS_FILE, CITIES_FILE],
            "city_assignment": "nearest US city center in the city table",
            "timestamps": "UTC plus each check-in's timezone offset",
            "cities": cities,
            "city_id": {city: CITY_ID[city] for city in cities},
            "checkins": {city: counts[city] for city in cities},
            "pois": {
                city: catalog.filter(pl.col("city") == city).height
                for city in cities
            },
            "split_seed": GLOBAL_SPLIT_SEED,
        },
    )


__all__ = [
    "SPLIT_COLUMN",
    "run_foursquare_pipeline",
]
