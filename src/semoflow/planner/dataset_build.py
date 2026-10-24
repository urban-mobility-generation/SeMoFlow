from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import torch

from semoflow.config.artifacts import TokenizerArtifacts
from semoflow.config.schemas.planner_config import (
    OccurrencePlannerDataOptions,
)
from semoflow.planner.artifacts import (
    DATASET_ARTIFACT_TYPE,
    save_torch_atomic,
)
from semoflow.planner.conditioning import (
    CONTEXT_MODE_CITY_CALENDAR_ORIGIN,
    SUPPORTED_CONTEXT_MODES,
    city_calendar_origin_features,
    city_conditioning_for_mode,
    context_schema_for_mode,
)
from semoflow.planner.gap_codec import (
    PLANNER_GAP_BUCKET_EDGES_MINUTES,
    PLANNER_GAP_CODEC_VERSION,
    PLANNER_GAP_REPRESENTATIVES_MINUTES,
    gap_tokens_from_event_time_minutes,
)
from semoflow.preprocess.weekly_segments import build_weekly_segment_cache
from semoflow.tokenization.decoding.codebooks import extract_codebook_bundle
from semoflow.tokenization.token_space import sid_code_columns

OBSERVATION_WINDOW_MINUTES = 7.0 * 24.0 * 60.0


@dataclass(frozen=True, slots=True)
class OccurrencePlannerDatasetBuildResult:
    output_path: Path
    train_rows: int
    validation_rows: int
    legal_prefixes: int
    overflow_policy: str
    source_rows: int
    eligible_rows: int
    overflow_rows: int
    truncated_rows: int
    dropped_overflow_rows: int
    overflow_fraction: float
    raw_mean_length: float
    effective_mean_length: float
    tokenizer_latent_dim: int
    centroid_coverage: float

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _require_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("_timestamps values must be datetime objects")
    return value


def _week_start(timestamp: object) -> datetime:
    value = _require_datetime(timestamp)
    midnight = value.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight - timedelta(days=value.weekday())


def _event_minutes(timestamps: list[object]) -> torch.Tensor:
    if not timestamps:
        raise ValueError("weekly segment must contain at least one timestamp")
    week_start = _week_start(timestamps[0])
    values: list[float] = []
    for timestamp in timestamps:
        value = _require_datetime(timestamp)
        minutes = (value - week_start).total_seconds() / 60.0
        if not 0.0 <= minutes <= OBSERVATION_WINDOW_MINUTES:
            raise ValueError(
                "weekly segment timestamp lies outside its Monday-start window"
            )
        values.append(float(minutes))
    result = torch.tensor(values, dtype=torch.float32)
    if result.numel() > 1 and bool((result[1:] < result[:-1]).any()):
        raise ValueError("weekly segment timestamps must be non-decreasing")
    return result


def _load_prefix_codebooks(
    path: Path | None,
    *,
    prefix_depth: int,
    codebook_size: int,
) -> torch.Tensor | None:
    if path is None:
        return None
    raw = torch.load(path, map_location="cpu", weights_only=True)
    value = raw.get("codebooks") if isinstance(raw, dict) else raw
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"tokenizer codebook artifact {path} does not carry a tensor"
        )
    codebooks = value.detach().to("cpu", torch.float32)
    if codebooks.ndim != 3:
        raise ValueError("tokenizer codebooks must have shape [levels, K, D]")
    if codebooks.shape[0] < prefix_depth:
        raise ValueError("tokenizer codebooks do not cover prefix_depth")
    if codebooks.shape[1] != codebook_size:
        raise ValueError(
            "tokenizer codebook size disagrees with --codebook-size"
        )
    if codebooks.shape[2] < 1 or not bool(torch.isfinite(codebooks).all()):
        raise ValueError("tokenizer codebooks must be finite and non-empty")
    return codebooks[:prefix_depth].contiguous().clone()


def _load_prefix_centroids(
    lookup_path: Path,
    *,
    places_paths: Sequence[Path] | None,
    legal_prefixes: torch.Tensor,
    prefix_depth: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not places_paths:
        return None, None
    schema = pl.scan_parquet(lookup_path).collect_schema()
    names = schema.names()
    required = {"placekey", "latitude", "longitude"}
    frames: list[pl.DataFrame] = [
        pl.read_parquet(places_path, columns=sorted(required))
        for places_path in places_paths
        if required.issubset(pl.read_parquet_schema(places_path))
    ]
    if not frames:
        return None, None
    if "placekey" not in names:
        raise ValueError(
            "joining --places coordinates requires placekey in the lookup"
        )
    selected = list(sid_code_columns(names))[:prefix_depth]
    places = pl.concat(frames, how="vertical").unique(
        subset=["placekey"], keep="first"
    )
    grouped = (
        pl.read_parquet(lookup_path, columns=["placekey", *selected])
        .join(places, on="placekey", how="inner", maintain_order="left")
        .drop_nulls(["latitude", "longitude"])
        .group_by(selected)
        .agg(
            pl.col("latitude").mean().alias("latitude"),
            pl.col("longitude").mean().alias("longitude"),
        )
    )
    vocabulary = int(legal_prefixes.shape[0])
    centroids = torch.zeros((vocabulary, 2), dtype=torch.float32)
    available = torch.zeros(vocabulary, dtype=torch.bool)
    index_of = {
        tuple(row.tolist()): position
        for position, row in enumerate(legal_prefixes)
    }
    for row in grouped.iter_rows():
        codes = tuple(int(value) for value in row[:prefix_depth])
        position = index_of.get(codes)
        if position is None:
            continue
        latitude = float(row[prefix_depth])
        longitude = float(row[prefix_depth + 1])
        if not math.isfinite(latitude) or not math.isfinite(longitude):
            continue
        if abs(latitude) > 90.0 or abs(longitude) > 180.0:
            raise ValueError("places coordinates are outside WGS84")
        centroids[position] = torch.tensor(
            [latitude, longitude], dtype=torch.float32
        )
        available[position] = True
    if not bool(available.any()):
        raise ValueError(
            "no legal prefix received a centroid from the supplied --places"
        )
    return centroids, available


def _load_prefix_lookup(
    lookup_path: Path,
    *,
    places_paths: Sequence[Path] | None,
    prefix_depth: int,
    codebook_size: int,
    context_mode: str,
) -> tuple[
    dict[int, tuple[torch.Tensor, int, str | None]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    schema = pl.scan_parquet(lookup_path).collect_schema()
    names = schema.names()
    code_columns = list(sid_code_columns(names))
    if len(code_columns) < prefix_depth:
        raise ValueError(
            "lookup does not carry enough semantic-ID levels for prefix_depth"
        )
    selected = code_columns[:prefix_depth]
    has_city = "city_id" in names
    columns = ["placekey_id", *selected]
    if has_city:
        columns.insert(1, "city_id")
    geohash_by_placekey: dict[str, str] | None = None
    if (
        context_mode == CONTEXT_MODE_CITY_CALENDAR_ORIGIN
        and "geohash_5" in names
    ):
        columns.append("geohash_5")
    elif context_mode == CONTEXT_MODE_CITY_CALENDAR_ORIGIN:
        if not places_paths:
            raise ValueError(
                "city_calendar_origin context requires geohash_5 in the "
                "lookup or --places"
            )
        if "placekey" not in names:
            raise ValueError(
                "joining --places requires placekey in the planner lookup"
            )
        required_places = {"placekey", "geohash_5"}
        place_frames: list[pl.DataFrame] = []
        for places_path in places_paths:
            places_schema = pl.read_parquet_schema(places_path)
            missing_places = sorted(required_places.difference(places_schema))
            if missing_places:
                raise ValueError(
                    f"places {places_path} are missing required columns: "
                    + ", ".join(missing_places)
                )
            place_frames.append(
                pl.read_parquet(places_path, columns=["placekey", "geohash_5"])
            )
        places = pl.concat(
            place_frames,
            how="vertical_relaxed",
        )
        if places["placekey"].n_unique() != places.height:
            raise ValueError("places contains duplicate placekey rows")
        geohash_by_placekey = {
            str(placekey): str(geohash)
            for placekey, geohash in places.iter_rows()
            if geohash is not None
        }
        columns.append("placekey")
    frame = pl.read_parquet(lookup_path).select(columns)

    raw_rows: list[tuple[int, int, tuple[int, ...], str | None]] = []
    seen_placekeys: set[int] = set()
    global_tuples: set[tuple[int, ...]] = set()
    city_tuples: dict[int, set[tuple[int, ...]]] = {}
    for row in frame.iter_rows():
        placekey_id = int(row[0])
        cursor = 1
        city_id = int(row[cursor]) if has_city else 0
        if has_city:
            cursor += 1
        code_stop = cursor + len(selected)
        codes = tuple(int(value) for value in row[cursor:code_stop])
        origin_geohash: str | None = None
        if context_mode == CONTEXT_MODE_CITY_CALENDAR_ORIGIN:
            raw_geo = row[code_stop]
            origin_geohash = (
                str(raw_geo)
                if geohash_by_placekey is None and raw_geo is not None
                else geohash_by_placekey.get(str(raw_geo))
                if geohash_by_placekey is not None
                else None
            )
        if (
            context_mode == CONTEXT_MODE_CITY_CALENDAR_ORIGIN
            and not origin_geohash
        ):
            raise ValueError(
                "city_calendar_origin context requires non-null geohash_5"
            )
        if any(code < 0 or code >= codebook_size for code in codes):
            raise ValueError("lookup SID code is outside --codebook-size")
        if placekey_id in seen_placekeys:
            raise ValueError("lookup contains duplicate placekey_id")
        seen_placekeys.add(placekey_id)
        raw_rows.append((placekey_id, city_id, codes, origin_geohash))
        global_tuples.add(codes)
        city_tuples.setdefault(city_id, set()).add(codes)
    if not raw_rows:
        raise ValueError("lookup is empty")

    ordered_tuples = sorted(global_tuples)
    legal_prefixes = torch.tensor(ordered_tuples, dtype=torch.long)
    tuple_to_id = {codes: index for index, codes in enumerate(ordered_tuples)}
    catalog_city_ids = torch.tensor(sorted(city_tuples), dtype=torch.long)
    city_legal_prefix_mask = torch.zeros(
        (catalog_city_ids.numel(), legal_prefixes.shape[0]),
        dtype=torch.bool,
    )
    city_to_row = {
        int(city_id): row
        for row, city_id in enumerate(catalog_city_ids.tolist())
    }
    for city_id, tuples in city_tuples.items():
        row = city_to_row[city_id]
        for codes in tuples:
            city_legal_prefix_mask[row, tuple_to_id[codes]] = True

    mapping: dict[int, tuple[torch.Tensor, int, str | None]] = {}
    for placekey_id, city_id, codes, origin_geohash in raw_rows:
        mapping[placekey_id] = (
            torch.tensor(codes, dtype=torch.long),
            city_id,
            origin_geohash,
        )
    return (
        mapping,
        legal_prefixes,
        catalog_city_ids,
        city_legal_prefix_mask,
    )


def _materialize_split(
    frame: pl.DataFrame,
    *,
    lookup: dict[int, tuple[torch.Tensor, int, str | None]],
    prefix_depth: int,
    max_occurrences: int,
    context_mode: str,
) -> dict[str, object]:
    rows = frame.height
    if rows < 1:
        raise ValueError("planner train/validation split must be non-empty")
    prefixes = torch.zeros(
        (rows, max_occurrences, prefix_depth), dtype=torch.long
    )
    gaps = torch.zeros((rows, max_occurrences), dtype=torch.long)
    lengths = torch.empty(rows, dtype=torch.long)
    event_times = torch.zeros((rows, max_occurrences), dtype=torch.float32)
    city_ids = torch.empty(rows, dtype=torch.long)
    context_schema = context_schema_for_mode(context_mode)
    context_features = (
        torch.empty((rows, len(context_schema)), dtype=torch.float32)
        if context_schema
        else None
    )
    for row_index, row in enumerate(frame.iter_rows(named=True)):
        placekey_ids = [int(value) for value in row["placekey_id"]]
        timestamps = list(row["_timestamps"])
        if len(placekey_ids) != len(timestamps):
            raise ValueError("placekey_id and _timestamps lengths differ")
        length = len(placekey_ids)
        missing = [value for value in placekey_ids if value not in lookup]
        if missing:
            raise ValueError(
                "segment contains placekey_id absent from lookup: "
                f"{missing[:5]}"
            )
        lookup_rows = [lookup[value] for value in placekey_ids]
        row_city_ids = {city_id for _, city_id, _ in lookup_rows}
        if len(row_city_ids) != 1:
            raise ValueError(
                "one planner sequence cannot contain POIs from multiple cities"
            )
        inferred_city = next(iter(row_city_ids))
        if "city_id" in row and int(row["city_id"]) != inferred_city:
            raise ValueError(
                "segment city_id disagrees with lookup place city identity"
            )
        city_ids[row_index] = inferred_city
        row_prefixes = torch.stack([prefix for prefix, _, _ in lookup_rows])
        row_times = _event_minutes(timestamps)
        row_gaps = gap_tokens_from_event_time_minutes(
            row_times,
            bucket_edges_minutes=PLANNER_GAP_BUCKET_EDGES_MINUTES,
        )
        lengths[row_index] = length
        prefixes[row_index, :length] = row_prefixes
        gaps[row_index, :length] = row_gaps
        event_times[row_index, :length] = row_times
        if context_features is not None:
            origin_geohash = lookup_rows[0][2]
            assert origin_geohash is not None
            context_features[row_index] = city_calendar_origin_features(
                week_start=_week_start(timestamps[0]),
                origin_geohash=origin_geohash,
            )
    return {
        "prefix_codes": prefixes,
        "gap_tokens": gaps,
        "lengths": lengths,
        "context_features": context_features,
        "event_time_minutes": event_times,
        "city_ids": city_ids,
    }


def _load_segments(segments_path: Path) -> pl.DataFrame:
    schema = pl.read_parquet_schema(segments_path)
    required = {"_split", "placekey_id", "_timestamps"}
    missing = sorted(required.difference(schema.keys()))
    if missing:
        raise ValueError(
            "segments are missing required columns: " + ", ".join(missing)
        )
    segment_columns = ["_split", "placekey_id", "_timestamps"]
    if "city_id" in schema:
        segment_columns.append("city_id")
    frame = (
        pl.scan_parquet(segments_path)
        .filter(pl.col("_split").is_in(["train", "val"]))
        .select(segment_columns)
        .collect()
    )
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(
            "collecting planner segments did not return a DataFrame"
        )
    return frame


def _bound_occurrences(
    frame: pl.DataFrame,
    *,
    min_occurrences: int,
    max_occurrences: int,
    overflow_policy: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Apply occurrence bounds; return the frame and its length stats."""
    place_lengths = pl.col("placekey_id").list.len()
    timestamp_lengths = pl.col("_timestamps").list.len()
    source_rows = frame.height
    if frame.filter(place_lengths != timestamp_lengths).height:
        raise ValueError("placekey_id and _timestamps lengths differ")
    frame = frame.filter(place_lengths >= min_occurrences)
    eligible_rows = frame.height
    if eligible_rows < 1:
        raise ValueError("no segments satisfy min_occurrences")
    raw_mean_length = float(frame.select(place_lengths.mean()).item())
    overflow_rows = frame.filter(place_lengths > max_occurrences).height
    truncated_rows = 0
    dropped_overflow_rows = 0
    if overflow_rows:
        if overflow_policy == "error":
            raise ValueError(
                f"{overflow_rows} segments exceed max_occurrences="
                f"{max_occurrences}"
            )
        if overflow_policy == "drop":
            frame = frame.filter(place_lengths <= max_occurrences)
            dropped_overflow_rows = overflow_rows
        else:
            frame = frame.with_columns(
                pl.col("placekey_id").list.slice(0, max_occurrences),
                pl.col("_timestamps").list.slice(0, max_occurrences),
            )
            truncated_rows = overflow_rows
    if frame.height < 1:
        raise ValueError("overflow policy removed all eligible segments")
    return frame, {
        "source_rows": source_rows,
        "eligible_rows": eligible_rows,
        "overflow_rows": overflow_rows,
        "truncated_rows": truncated_rows,
        "dropped_overflow_rows": dropped_overflow_rows,
        "overflow_fraction": overflow_rows / eligible_rows,
        "raw_mean_length": raw_mean_length,
        "effective_mean_length": float(
            frame.select(place_lengths.mean()).item()
        ),
    }


def _data_config(
    *,
    codebook_size: int,
    prefix_depth: int,
    max_occurrences: int,
    context_mode: str,
) -> dict[str, object]:
    context_schema = context_schema_for_mode(context_mode)
    return {
        "codebook_size": codebook_size,
        "gap_vocab": len(PLANNER_GAP_REPRESENTATIVES_MINUTES),
        "prefix_depth": prefix_depth,
        "max_occurrences": max_occurrences,
        "context_dim": len(context_schema),
        "context_schema": context_schema,
        "city_conditioning": city_conditioning_for_mode(context_mode),
        "gap_codec_version": PLANNER_GAP_CODEC_VERSION,
        "gap_bucket_edges_minutes": PLANNER_GAP_BUCKET_EDGES_MINUTES,
        "gap_duration_minutes": PLANNER_GAP_REPRESENTATIVES_MINUTES,
        "observation_window_minutes": OBSERVATION_WINDOW_MINUTES,
    }


def build_occurrence_planner_dataset(
    *,
    segments_path: Path,
    lookup_path: Path,
    places_paths: Sequence[Path] | None = None,
    output_path: Path,
    codebook_size: int,
    prefix_depth: int = 3,
    min_occurrences: int = 1,
    max_occurrences: int = 30,
    overflow_policy: str = "truncate",
    context_mode: str = "none",
    codebooks_path: Path | None = None,
    force: bool = False,
) -> OccurrencePlannerDatasetBuildResult:
    if codebook_size < 1 or prefix_depth < 1:
        raise ValueError("codebook_size and prefix_depth must be positive")
    if min_occurrences < 1 or max_occurrences < min_occurrences:
        raise ValueError("occurrence bounds are invalid")
    if overflow_policy not in {"truncate", "drop", "error"}:
        raise ValueError(
            "overflow_policy must be 'truncate', 'drop', or 'error'"
        )
    if context_mode not in SUPPORTED_CONTEXT_MODES:
        raise ValueError(
            "context_mode must be 'none', 'city', or 'city_calendar_origin'"
        )
    segments = _load_segments(segments_path)
    (
        lookup,
        legal_prefixes,
        catalog_city_ids,
        city_legal_prefix_mask,
    ) = _load_prefix_lookup(
        lookup_path,
        places_paths=places_paths,
        prefix_depth=prefix_depth,
        codebook_size=codebook_size,
        context_mode=context_mode,
    )
    prefix_codebooks = _load_prefix_codebooks(
        codebooks_path,
        prefix_depth=prefix_depth,
        codebook_size=codebook_size,
    )
    legal_prefix_centroids, legal_prefix_centroid_available = (
        _load_prefix_centroids(
            lookup_path,
            places_paths=places_paths,
            legal_prefixes=legal_prefixes,
            prefix_depth=prefix_depth,
        )
    )
    frame, stats = _bound_occurrences(
        segments,
        min_occurrences=min_occurrences,
        max_occurrences=max_occurrences,
        overflow_policy=overflow_policy,
    )
    splits = {
        name: frame.filter(pl.col("_split") == split).drop("_split")
        for name, split in (("train", "train"), ("validation", "val"))
    }
    if any(split.height < 1 for split in splits.values()):
        raise ValueError(
            "overflow and occurrence filters must retain both train and "
            "validation rows"
        )
    payload = {
        "artifact_type": DATASET_ARTIFACT_TYPE,
        "data_config": _data_config(
            codebook_size=codebook_size,
            prefix_depth=prefix_depth,
            max_occurrences=max_occurrences,
            context_mode=context_mode,
        ),
        "legal_prefixes": legal_prefixes,
        "prefix_codebooks": prefix_codebooks,
        "legal_prefix_centroids": legal_prefix_centroids,
        "legal_prefix_centroid_available": legal_prefix_centroid_available,
        "catalog_city_ids": catalog_city_ids,
        "city_legal_prefix_mask": city_legal_prefix_mask,
        **{
            name: _materialize_split(
                split,
                lookup=lookup,
                prefix_depth=prefix_depth,
                max_occurrences=max_occurrences,
                context_mode=context_mode,
            )
            for name, split in splits.items()
        },
    }
    save_torch_atomic(output_path, payload, force=force)
    return OccurrencePlannerDatasetBuildResult(
        output_path=Path(output_path),
        train_rows=splits["train"].height,
        validation_rows=splits["validation"].height,
        legal_prefixes=int(legal_prefixes.shape[0]),
        overflow_policy=overflow_policy,
        **stats,
        tokenizer_latent_dim=(
            0 if prefix_codebooks is None else int(prefix_codebooks.shape[-1])
        ),
        centroid_coverage=(
            0.0
            if legal_prefix_centroid_available is None
            else float(legal_prefix_centroid_available.float().mean())
        ),
    )


def build_planner_data(
    data: OccurrencePlannerDataOptions,
    *,
    dataset_path: Path,
    force: bool = False,
) -> dict[str, object]:
    """Build the weekly segments, then the planner dataset from them."""
    tokenizer = TokenizerArtifacts.from_run_dir(data.tokenization_dir)
    segments = build_weekly_segment_cache(
        stays_path=tokenizer.augmented_processed_visits,
        output=data.segments_path,
        min_active_days=data.min_active_days,
        force=force,
    )
    dataset = build_occurrence_planner_dataset(
        segments_path=data.segments_path,
        lookup_path=tokenizer.placekey_lookup,
        places_paths=(
            None if data.places_path is None else [data.places_path]
        ),
        output_path=dataset_path,
        codebook_size=extract_codebook_bundle(
            tokenizer.tokenizer_codebooks
        ).codebook_size,
        max_occurrences=data.max_occurrences,
        context_mode=data.context_mode,
        force=force,
    )
    return {"segments": segments, "dataset": dataset.to_dict()}


__all__ = [
    "OBSERVATION_WINDOW_MINUTES",
    "OccurrencePlannerDatasetBuildResult",
    "build_occurrence_planner_dataset",
    "build_planner_data",
]
