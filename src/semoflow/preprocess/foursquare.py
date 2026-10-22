from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from semoflow.config import constants
from semoflow.config.schemas.data_config import (
    FoursquareAdapterConfig,
)
from semoflow.preprocess.pipeline import (
    PLACE_INDEX_COLUMNS,
    PreprocessWorkspace,
    log_pipeline_done,
)
from semoflow.preprocess.places_index import (
    write_places_index,
)
from semoflow.preprocess.quality_gate import (
    QualityGateConfig,
    filter_quality,
)
from semoflow.utils import write_run_manifest
from semoflow.utils.data import stable_device_key
from semoflow.utils.provenance import (
    closure_from_stays,
    write_closure,
)
from semoflow.utils.splits import (
    GLOBAL_SPLIT_SEED,
    SPLIT_COLUMN,
    assign_split,
)

if TYPE_CHECKING:
    from semoflow.config import ExperimentConfig

logger = logging.getLogger(__name__)

FOURSQUARE_SOURCE_URL = (
    "https://sites.google.com/site/yangdingqi/home/foursquare-dataset"
)
FOURSQUARE_GLOBAL_CITY_NAMES = {
    "los_angeles": "Los Angeles",
    "washington_dc": "Washington D.C.",
    "atlanta": "Atlanta",
    "san_francisco": "San Francisco",
    "milwaukee": "Milwaukee",
}
FOURSQUARE_GLOBAL_CITY_IDS = {
    city: constants.CITY_ID[city] for city in FOURSQUARE_GLOBAL_CITY_NAMES
}
FOURSQUARE_GLOBAL_CHECKIN_COLUMNS = (
    "user_id",
    "venue_id",
    "timestamp_utc_raw",
    "timezone_offset_min",
)
FOURSQUARE_GLOBAL_POI_COLUMNS = (
    "venue_id",
    "latitude",
    "longitude",
    "category_name",
    "country_code",
)
FOURSQUARE_GLOBAL_CITY_COLUMNS = (
    "source_city_name",
    "latitude",
    "longitude",
    "country_code",
    "country_name",
    "city_type",
)
_TIMESTAMP_FORMAT = "%a %b %d %H:%M:%S %z %Y"
_GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"


def _global_source_file(source: Path, filename: str) -> Path:
    if not source.is_dir():
        raise ValueError(
            "Foursquare global_2015 input must be an extracted directory; "
            f"got {source}"
        )
    path = source / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Foursquare global_2015 file not found: {path}"
        )
    return path


def _scan_global_tsv(
    path: Path,
    *,
    columns: tuple[str, ...],
    batch_size: int,
) -> pl.LazyFrame:
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


def _unit_sphere_coordinates(
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    lat = np.deg2rad(latitude)
    lon = np.deg2rad(longitude)
    cos_lat = np.cos(lat)
    return np.column_stack(
        [cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)]
    )


def _nearest_center_indices(
    points: np.ndarray,
    centers: np.ndarray,
    *,
    chunk_size: int = 100_000,
) -> np.ndarray:
    nearest = np.empty(points.shape[0], dtype=np.int32)
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        nearest[start:stop] = np.argmax(
            points[start:stop] @ centers.T, axis=1
        ).astype(np.int32)
    return nearest


def read_foursquare_global_catalog(
    source: str | Path,
    *,
    cities: list[str],
    config: FoursquareAdapterConfig,
) -> pl.DataFrame:
    source_path = Path(source).expanduser()
    cities_path = _global_source_file(
        source_path, config.global_cities_filename
    )
    pois_path = _global_source_file(source_path, config.global_pois_filename)

    center_batches = list(
        pl.scan_csv(
            cities_path,
            has_header=False,
            separator="\t",
            new_columns=list(FOURSQUARE_GLOBAL_CITY_COLUMNS),
            schema_overrides={
                column: pl.Utf8 for column in FOURSQUARE_GLOBAL_CITY_COLUMNS
            },
            infer_schema_length=0,
            null_values=[""],
            quote_char=None,
        ).collect_batches(chunk_size=config.global_batch_size)
    )
    if not center_batches:
        raise ValueError("global Foursquare city table is empty")
    centers = pl.concat(center_batches, how="diagonal_relaxed")
    centers = centers.with_columns(
        [
            pl.col("latitude").cast(pl.Float64, strict=False),
            pl.col("longitude").cast(pl.Float64, strict=False),
            pl.col("country_code").str.to_uppercase(),
        ]
    ).filter(pl.col("country_code") == "US")
    if centers.is_empty():
        raise ValueError("global Foursquare city table contains no US centers")
    invalid_centers = centers.filter(
        pl.col("latitude").is_null()
        | pl.col("longitude").is_null()
        | (pl.col("latitude") < -90.0)
        | (pl.col("latitude") > 90.0)
        | (pl.col("longitude") < -180.0)
        | (pl.col("longitude") > 180.0)
    )
    if invalid_centers.height:
        raise ValueError(
            "global Foursquare city table contains invalid US center coordinates"
        )

    requested_names = []
    for city in cities:
        if city not in FOURSQUARE_GLOBAL_CITY_NAMES:
            raise ValueError(
                f"unsupported global Foursquare city {city!r}; expected "
                f"{sorted(FOURSQUARE_GLOBAL_CITY_NAMES)}"
            )
        requested_names.append(FOURSQUARE_GLOBAL_CITY_NAMES[city])
    available_names = set(centers["source_city_name"].to_list())
    missing_names = set(requested_names) - available_names
    if missing_names:
        raise ValueError(
            "global Foursquare city table is missing requested centers: "
            f"{sorted(missing_names)}"
        )

    center_names = centers["source_city_name"].to_list()
    center_points = _unit_sphere_coordinates(
        centers["latitude"].to_numpy(), centers["longitude"].to_numpy()
    )
    city_map = pl.DataFrame(
        {
            "_assigned_source_city": requested_names,
            "city": cities,
            "city_id": pl.Series(
                "city_id",
                [FOURSQUARE_GLOBAL_CITY_IDS[city] for city in cities],
                dtype=pl.Int32,
            ),
        }
    )

    selected_batches: list[pl.DataFrame] = []
    for batch in _scan_global_tsv(
        pois_path,
        columns=FOURSQUARE_GLOBAL_POI_COLUMNS,
        batch_size=config.global_batch_size,
    ).collect_batches(chunk_size=config.global_batch_size):
        batch = batch.with_columns(
            [
                pl.col("latitude").cast(pl.Float64, strict=False),
                pl.col("longitude").cast(pl.Float64, strict=False),
                pl.col("country_code").str.to_uppercase(),
            ]
        ).filter(pl.col("country_code") == "US")
        if batch.is_empty():
            continue
        invalid = batch.filter(
            pl.col("venue_id").is_null()
            | (pl.col("venue_id").str.strip_chars().str.len_chars() == 0)
            | pl.col("latitude").is_null()
            | pl.col("longitude").is_null()
            | (pl.col("latitude") < -90.0)
            | (pl.col("latitude") > 90.0)
            | (pl.col("longitude") < -180.0)
            | (pl.col("longitude") > 180.0)
        )
        if invalid.height:
            raise ValueError(
                "global Foursquare POI table contains invalid US rows; "
                "refusing to impute venue identities or coordinates"
            )
        points = _unit_sphere_coordinates(
            batch["latitude"].to_numpy(), batch["longitude"].to_numpy()
        )
        nearest = _nearest_center_indices(points, center_points)
        batch = batch.with_columns(
            pl.Series(
                "_assigned_source_city",
                [center_names[int(index)] for index in nearest],
            )
        )
        selected = (
            batch.filter(
                pl.col("_assigned_source_city").is_in(requested_names)
            )
            .join(city_map, on="_assigned_source_city", how="inner")
            .drop("_assigned_source_city")
        )
        if not selected.is_empty():
            selected_batches.append(selected)

    if not selected_batches:
        raise ValueError(
            "global Foursquare POI table contains no POIs for requested cities"
        )
    catalog = pl.concat(selected_batches, how="diagonal_relaxed")
    duplicate_venues = (
        catalog.group_by("venue_id")
        .agg(pl.col("city").n_unique().alias("city_count"))
        .filter(pl.col("city_count") > 1)
    )
    if not duplicate_venues.is_empty():
        raise ValueError(
            "global Foursquare POI table assigns a venue to multiple requested "
            "cities"
        )
    return catalog.unique("venue_id", keep="first", maintain_order=True)


def normalize_foursquare_global_events(
    raw: pl.DataFrame,
    *,
    poi_catalog: pl.DataFrame,
) -> pl.DataFrame:
    missing = set(FOURSQUARE_GLOBAL_CHECKIN_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(
            f"global Foursquare frame is missing columns: {sorted(missing)}"
        )
    frame = raw.join(poi_catalog, on="venue_id", how="inner").with_columns(
        [
            pl.col("latitude").cast(pl.Float64, strict=False),
            pl.col("longitude").cast(pl.Float64, strict=False),
            pl.col("timezone_offset_min").cast(pl.Int32, strict=False),
            pl.col("timestamp_utc_raw")
            .str.strptime(
                pl.Datetime,
                format=_TIMESTAMP_FORMAT,
                strict=False,
            )
            .alias("timestamp_utc"),
        ]
    )
    invalid_text = pl.any_horizontal(
        [
            pl.col(column).is_null()
            | (pl.col(column).str.strip_chars().str.len_chars() == 0)
            for column in ("user_id", "venue_id", "category_name", "city")
        ]
    )
    invalid_numeric = (
        pl.col("latitude").is_null()
        | pl.col("longitude").is_null()
        | (pl.col("latitude") < -90.0)
        | (pl.col("latitude") > 90.0)
        | (pl.col("longitude") < -180.0)
        | (pl.col("longitude") > 180.0)
        | pl.col("timezone_offset_min").is_null()
        | (pl.col("timezone_offset_min") < -840)
        | (pl.col("timezone_offset_min") > 840)
    )
    invalid = frame.filter(
        invalid_text | invalid_numeric | pl.col("timestamp_utc").is_null()
    )
    if invalid.height:
        raise ValueError(
            f"global Foursquare contains {invalid.height} invalid selected rows; "
            "refusing to impute identities, coordinates, offsets, or time"
        )
    return (
        frame.with_columns(
            [
                (
                    pl.col("timestamp_utc")
                    + pl.duration(minutes=pl.col("timezone_offset_min"))
                )
                .dt.replace_time_zone(None)
                .alias("timestamp"),
                pl.format(
                    "foursquare:{}:{}", pl.col("city"), pl.col("venue_id")
                ).alias("placekey"),
                pl.format("foursquare:{}", pl.col("user_id")).alias("caid"),
                pl.col("category_name").alias("top_category"),
                pl.lit("utf-8").alias("source_text_encoding"),
            ]
        )
        .with_columns(
            [
                pl.struct(["city", "caid"])
                .map_elements(
                    lambda value: stable_device_key(
                        str(value["city"]), str(value["caid"])
                    ),
                    return_dtype=pl.Utf8,
                )
                .alias("device_key"),
                pl.col("caid")
                .map_elements(
                    lambda value: assign_split(
                        str(value), seed=GLOBAL_SPLIT_SEED
                    ),
                    return_dtype=pl.Utf8,
                )
                .alias(SPLIT_COLUMN),
                pl.struct(["latitude", "longitude"])
                .map_elements(
                    lambda value: encode_geohash(
                        value["latitude"], value["longitude"]
                    ),
                    return_dtype=pl.Utf8,
                )
                .alias("geohash_5"),
                pl.col("timestamp").dt.date().alias("local_day"),
                pl.col("timestamp").dt.hour().cast(pl.Int32).alias("hour"),
                pl.col("timestamp")
                .dt.weekday()
                .cast(pl.Int32)
                .alias("weekday"),
                (pl.col("timestamp").dt.weekday() >= 6).alias("is_weekend"),
                pl.col("timestamp").dt.month().cast(pl.Int32).alias("month"),
            ]
        )
        .sort(["user_id", "timestamp_utc", "venue_id"])
    )


def encode_geohash(
    latitude: float,
    longitude: float,
    *,
    precision: int = 5,
) -> str:
    if precision < 1:
        raise ValueError("geohash precision must be positive")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"latitude out of range: {latitude}")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError(f"longitude out of range: {longitude}")
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


_MISSING_PLACE_COLUMNS = (
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


def foursquare_place_index(events: pl.DataFrame) -> pl.DataFrame:
    return (
        events.unique("placekey", keep="first", maintain_order=True)
        .rename({"venue_id": "source_venue_id"})
        .with_columns(
            pl.col("category_name").alias("top_category"),
            *(
                pl.lit(None, dtype=pl.Utf8).alias(column)
                for column in _MISSING_PLACE_COLUMNS
            ),
        )
        .select([*PLACE_INDEX_COLUMNS, "source_venue_id"])
    )


def _source_path(data_root: Path, config: FoursquareAdapterConfig) -> Path:
    if config.archive_path:
        candidate = Path(config.archive_path).expanduser()
        return candidate if candidate.is_absolute() else data_root / candidate
    return data_root / config.raw_dir


def _source_readme(source: Path) -> str | None:
    for name in ("dataset_TIST2015_readme.txt", "dataset_TIST2015_README.txt"):
        if (source / name).is_file():
            return str(source / name)
    return None


def run_foursquare_pipeline(cfg: ExperimentConfig) -> None:
    if cfg.data.dataset_name != "foursquare":
        raise ValueError(
            "run_foursquare_pipeline requires data.dataset_name='foursquare'"
        )
    adapter = cfg.data.foursquare
    adapter.validate()
    cities = list(cfg.data.cities)
    if not cities or any(
        city not in FOURSQUARE_GLOBAL_CITY_NAMES for city in cities
    ):
        raise ValueError(
            "Foursquare data.cities must contain only "
            f"{sorted(FOURSQUARE_GLOBAL_CITY_NAMES)}"
        )

    source = _source_path(
        Path(cfg.data.data_root).expanduser().resolve(), adapter
    )
    if not source.exists():
        raise FileNotFoundError(
            f"Foursquare source not found: {source}. Download and extract the "
            f"global release from {FOURSQUARE_SOURCE_URL}"
        )
    t0 = time.perf_counter()
    workspace = PreprocessWorkspace.from_config(cfg)
    workspace.prepare_output()

    catalog = read_foursquare_global_catalog(
        source,
        cities=cities,
        config=adapter,
    )
    checkins_path = _global_source_file(
        source, adapter.global_checkins_filename
    )
    selected_venue_ids = catalog["venue_id"].to_list()
    event_batches: list[pl.DataFrame] = []
    for batch in _scan_global_tsv(
        checkins_path,
        columns=FOURSQUARE_GLOBAL_CHECKIN_COLUMNS,
        batch_size=adapter.global_batch_size,
    ).collect_batches(chunk_size=adapter.global_batch_size):
        selected = batch.filter(pl.col("venue_id").is_in(selected_venue_ids))
        if selected.is_empty():
            continue
        event_batches.append(
            normalize_foursquare_global_events(
                selected,
                poi_catalog=catalog,
            )
        )
    if not event_batches:
        raise ValueError(
            "global Foursquare check-in table contains no events for requested "
            "cities"
        )
    raw_events = pl.concat(event_batches, how="diagonal_relaxed")

    all_events: list[pl.DataFrame] = []
    all_scores: list[pl.DataFrame] = []
    place_inputs: list[tuple[str, int, pl.DataFrame]] = []
    raw_counts: dict[str, int] = {}
    source_encodings: dict[str, list[str]] = {}
    for city in cities:
        city_id = FOURSQUARE_GLOBAL_CITY_IDS[city]
        city_events = raw_events.filter(pl.col("city") == city).sort(
            ["user_id", "timestamp_utc", "venue_id"]
        )
        if city_events.is_empty():
            raise ValueError(
                f"global Foursquare check-in table contains no events for {city}"
            )
        raw_counts[city] = city_events.height
        source_encodings[city] = ["utf-8"]
        scores, filtered = filter_quality(
            city_events,
            config=QualityGateConfig(
                min_quality_score=adapter.min_quality_score,
                drop_missing_coords=adapter.drop_missing_coordinates,
            ),
        )
        all_events.append(filtered)
        all_scores.append(scores)
        place_inputs.append(
            (city, city_id, foursquare_place_index(city_events))
        )

    events = pl.concat(all_events, how="diagonal_relaxed")
    places = write_places_index(
        place_inputs,
        events.select(["placekey", "city_id"]),
        path=workspace.artifacts.places_index,
    )
    stays = events.join(
        places.select(["placekey", "placekey_id"]),
        on="placekey",
        how="left",
    ).drop("placekey")
    pl.concat(all_scores, how="diagonal_relaxed").write_parquet(
        workspace.artifacts.quality_scores
    )
    stays.write_parquet(workspace.artifacts.processed_visits)
    closure = closure_from_stays(
        stays,
        artifact=str(workspace.artifacts.processed_visits),
        source=str(source),
        split_seed=GLOBAL_SPLIT_SEED,
    )
    write_closure(closure, workspace.artifacts.run_dir)
    readme = _source_readme(source)
    source_files = {
        "checkins": str(checkins_path),
        "pois": str(_global_source_file(source, adapter.global_pois_filename)),
        "cities": str(
            _global_source_file(source, adapter.global_cities_filename)
        ),
    }
    source_manifest = {
        "schema_version": 2,
        "dataset_name": "foursquare",
        "source_format": "global_2015",
        "source_url": FOURSQUARE_SOURCE_URL,
        "source_path": str(source),
        "source_files": source_files,
        "source_readme_path": readme,
        "event_semantics": "checkin",
        "timestamp_semantics": "UTC timestamp plus per-event timezone offset minutes",
        "dwell_policy": "not available; no dwell inferred or imputed",
        "city_assignment": "nearest US city center from global city table",
        "cities": cities,
        "raw_row_counts": raw_counts,
        "source_text_encodings": source_encodings,
        "city_id": {city: FOURSQUARE_GLOBAL_CITY_IDS[city] for city in cities},
        "selected_poi_counts": {
            city: catalog.filter(pl.col("city") == city).height
            for city in cities
        },
        "split_seed": GLOBAL_SPLIT_SEED,
    }
    (
        workspace.artifacts.run_dir / "foursquare_source_manifest.json"
    ).write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_run_manifest(
        workspace.artifacts.run_dir,
        config_path=workspace.artifacts.run_dir / cfg.output.config_filename,
        output_dir=workspace.artifacts.run_dir,
        data_artifact_paths={
            "foursquare_source": source,
            "foursquare_checkins": checkins_path,
        },
        metric_artifact_paths={
            "processed_visits": workspace.artifacts.processed_visits,
            "quality_scores": workspace.artifacts.quality_scores,
            "places_index": workspace.artifacts.places_index,
            "foursquare_source_manifest": workspace.artifacts.run_dir
            / "foursquare_source_manifest.json",
        },
    )
    log_pipeline_done(t0, workspace.output_dir)


__all__ = [
    "FOURSQUARE_GLOBAL_CITY_NAMES",
    "FOURSQUARE_SOURCE_URL",
    "encode_geohash",
    "foursquare_place_index",
    "normalize_foursquare_global_events",
    "read_foursquare_global_catalog",
    "run_foursquare_pipeline",
]
