from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl

from semoflow.config import CITY_ID
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


def assign_split(key: str, *, seed: int = GLOBAL_SPLIT_SEED) -> str:
    fraction = hash_fraction(key, seed)
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
_EMPTY_PLACE_COLUMNS = (
    "location_name",
    "sub_category",
    "naics_code",
    "street_address",
    "city",
    "state",
    "zipcode",
    "census_block_group",
    "brands",
)
_PLACE_COLUMNS = (
    "placekey",
    "location_name",
    "top_category",
    "sub_category",
    "naics_code",
    "street_address",
    "city",
    "state",
    "zipcode",
    "census_block_group",
    "brands",
    "geohash_5",
    "latitude",
    "longitude",
    "source_venue_id",
    "city_id",
)
_TIMESTAMP_FORMAT = "%a %b %d %H:%M:%S %z %Y"
_GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"
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
    lat = np.deg2rad(frame["latitude"].to_numpy())
    lon = np.deg2rad(frame["longitude"].to_numpy())
    return np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def encode_geohash(
    latitude: float, longitude: float, *, precision: int = 5
) -> str:
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    bits: list[int] = []
    even = True
    while len(bits) < precision * 5:
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
        for start in range(0, precision * 5, 5)
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
                lambda caid: assign_split(caid, seed=GLOBAL_SPLIT_SEED),
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
        .with_columns(
            pl.lit(None, dtype=pl.Utf8).alias(column)
            for column in _EMPTY_PLACE_COLUMNS
        )
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


SEGMENT_CITY_COLUMN = "_city_id"


def segment_events(
    segments: pl.DataFrame, lookup_path: Path, columns: list[str]
) -> pl.DataFrame:
    timestamp = pl.col("_timestamps")
    return (
        segments.select(
            pl.int_range(pl.len()).alias("_row"), "placekey_id", "_timestamps"
        )
        .explode("placekey_id", "_timestamps")
        .with_columns(
            pl.int_range(pl.len()).over("_row").alias("_pos"),
            (
                (
                    timestamp - timestamp.dt.truncate("1w")
                ).dt.total_microseconds()
                / 1_000_000
                / 60.0
            ).alias("_minutes"),
        )
        .join(
            pl.read_parquet(lookup_path, columns=["placekey_id", *columns]),
            on="placekey_id",
            maintain_order="left",
        )
    )


def pad_events(
    events: pl.DataFrame,
    values: np.ndarray,
    *,
    rows: int,
    width: int,
    fill: float | int,
) -> np.ndarray:
    padded = np.full(
        (rows, width, *values.shape[1:]), fill, dtype=values.dtype
    )
    padded[events["_row"].to_numpy(), events["_pos"].to_numpy()] = values
    return padded


SEGMENT_MIN_LENGTH = 5
SEGMENT_MAX_LENGTH = 128


def build_weekly_segment_cache(
    *,
    stays_path: Path,
    output: Path,
    min_active_days: int,
    force: bool = False,
) -> dict[str, Any]:
    if min_active_days < 1:
        raise ValueError("min_active_days must be >= 1")
    if output.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing artifact: {output}"
        )
    window = ["device_key", "_week"]
    stays = (
        pl.read_parquet(
            stays_path,
            columns=[
                "device_key",
                "timestamp",
                "placekey_id",
                SPLIT_COLUMN,
                "city_id",
            ],
        )
        .sort(["device_key", "timestamp"])
        .with_columns(pl.col("timestamp").dt.truncate("1w").alias("_week"))
        .with_columns(
            pl.int_range(0, pl.len()).over(window).alias("_pos"),
            pl.len().over(window).alias("_length"),
            pl.col("timestamp")
            .dt.date()
            .n_unique()
            .over(window)
            .alias("_active_days"),
        )
    )
    n_before = stays.select(pl.struct(window).n_unique()).item()
    stays = stays.filter(
        (pl.col("_length") >= SEGMENT_MIN_LENGTH)
        & (pl.col("_active_days") >= min_active_days)
    )
    logger.info(
        "weekly quality gate: %s -> %s windows (min_seq_len=%d, min_active_days=%d)",
        f"{n_before:,}",
        f"{stays.select(pl.struct(window).n_unique()).item():,}",
        SEGMENT_MIN_LENGTH,
        min_active_days,
    )
    grouped = (
        stays.with_columns(
            (pl.struct(window).rank("dense") - 1)
            .cast(pl.UInt32)
            .alias("_seg_id")
        )
        .filter(pl.col("_pos") < SEGMENT_MAX_LENGTH)
        .sort(["_seg_id", "_pos"])
        .group_by("_seg_id", maintain_order=True)
        .agg(
            pl.col("placekey_id"),
            pl.col("device_key").first(),
            pl.col("timestamp").alias("_timestamps"),
            pl.col(SPLIT_COLUMN).first().alias("_split"),
            pl.col("city_id").first().alias(SEGMENT_CITY_COLUMN),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    grouped.write_parquet(temporary)
    temporary.replace(output)
    return {
        "output": str(output),
        "segments": grouped.height,
        "segments_by_split": dict(
            grouped["_split"].value_counts().iter_rows()
        ),
        "min_seq_len": SEGMENT_MIN_LENGTH,
        "max_seq_len": SEGMENT_MAX_LENGTH,
        "min_active_days": min_active_days,
    }


__all__ = [
    "GLOBAL_SPLIT_SEED",
    "SEGMENT_CITY_COLUMN",
    "build_weekly_segment_cache",
    "run_foursquare_pipeline",
]
