from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import torch

from semoflow.config import (
    PlannerConfig,
    RealizerConfig,
    RunLayout,
)
from semoflow.poi.sids import load_codebooks, sid_code_columns, sid_code_names
from semoflow.preprocess import SPLIT_COLUMN
from semoflow.utils import save_torch_atomic

logger = logging.getLogger(__name__)

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
    force: bool,
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


N_PART_OF_DAY_BINS = 6


N_TIME_TOKENS = 42


PART_OF_DAY_START_MINUTES: tuple[int, ...] = (0, 360, 600, 840, 1080, 1320)


def time_tokens_from_week_minutes(minutes: torch.Tensor) -> torch.Tensor:
    part_of_day = torch.bucketize(
        minutes % 1440.0,
        torch.tensor(PART_OF_DAY_START_MINUTES[1:], dtype=minutes.dtype),
        right=True,
    )
    return (minutes / 1440.0).long() * N_PART_OF_DAY_BINS + part_of_day


PLANNER_GAP_BUCKET_EDGES_MINUTES = (
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    45.0,
    90.0,
    180.0,
    360.0,
    720.0,
    1080.0,
    1440.0,
    2160.0,
    2880.0,
    4320.0,
)


PLANNER_GAP_REPRESENTATIVES_MINUTES = (
    0.5,
    1.5,
    3.5,
    7.5,
    15.0,
    32.5,
    67.5,
    135.0,
    270.0,
    540.0,
    900.0,
    1260.0,
    1800.0,
    2520.0,
    3600.0,
    5040.0,
)


def reconstruct_event_time_minutes(
    gap_tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    representative_minutes: tuple[float, ...],
    observation_window_minutes: float,
) -> torch.Tensor:
    representatives = torch.tensor(
        representative_minutes, dtype=torch.float32, device=gap_tokens.device
    )
    durations = torch.where(valid_mask, representatives[gap_tokens], 0.0)
    return torch.cumsum(durations, dim=-1).clamp(
        max=float(observation_window_minutes)
    )


def gap_tokens_from_event_time_minutes(
    event_time_minutes: torch.Tensor,
) -> torch.Tensor:
    durations = torch.diff(
        event_time_minutes,
        prepend=torch.zeros_like(event_time_minutes[..., :1]),
    )
    edges = torch.tensor(
        PLANNER_GAP_BUCKET_EDGES_MINUTES, dtype=durations.dtype
    )
    return torch.bucketize(durations, edges, right=False).to(torch.long)


DATASET_ARTIFACT_TYPE = "occurrence_planner_dataset"


OBSERVATION_WINDOW_MINUTES = 7.0 * 24.0 * 60.0


PREFIX_DEPTH = 3


_PREFIX_COLUMNS = sid_code_names(PREFIX_DEPTH)


def _legal_prefixes(
    lookup_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frame = pl.read_parquet(lookup_path, columns=["city_id", *_PREFIX_COLUMNS])
    city_tuples: dict[int, set[tuple[int, ...]]] = {}
    for city_id, *prefix in frame.iter_rows():
        city_tuples.setdefault(city_id, set()).add(tuple(prefix))

    ordered_tuples = sorted(set().union(*city_tuples.values()))
    legal_prefixes = torch.tensor(ordered_tuples, dtype=torch.long)
    tuple_to_id = {codes: index for index, codes in enumerate(ordered_tuples)}
    catalog_city_ids = torch.tensor(sorted(city_tuples), dtype=torch.long)
    city_legal_prefix_mask = torch.zeros(
        (catalog_city_ids.numel(), legal_prefixes.shape[0]),
        dtype=torch.bool,
    )
    for row, city_id in enumerate(catalog_city_ids.tolist()):
        for codes in city_tuples[city_id]:
            city_legal_prefix_mask[row, tuple_to_id[codes]] = True
    return legal_prefixes, catalog_city_ids, city_legal_prefix_mask


def _load_prefix_centroids(
    lookup_path: Path,
    places_path: Path,
    *,
    legal_prefixes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = list(_PREFIX_COLUMNS)
    places = pl.read_parquet(
        places_path, columns=["placekey", "latitude", "longitude"]
    ).unique(subset=["placekey"], keep="first")
    grouped = (
        pl.read_parquet(lookup_path, columns=["placekey", *selected])
        .join(places, on="placekey", how="inner", maintain_order="left")
        .drop_nulls(["latitude", "longitude"])
        .group_by(selected)
        .agg(pl.col("latitude").mean(), pl.col("longitude").mean())
    )
    vocabulary = int(legal_prefixes.shape[0])
    centroids = torch.zeros((vocabulary, 2), dtype=torch.float32)
    available = torch.zeros(vocabulary, dtype=torch.bool)
    index_of = {
        tuple(row.tolist()): position
        for position, row in enumerate(legal_prefixes)
    }
    for row in grouped.iter_rows():
        position = index_of[tuple(int(value) for value in row[:PREFIX_DEPTH])]
        centroids[position] = torch.tensor(row[PREFIX_DEPTH:])
        available[position] = True
    return centroids, available


def _materialize_split(
    frame: pl.DataFrame, lookup_path: Path, *, max_occurrences: int
) -> dict[str, torch.Tensor]:
    events = segment_events(frame, lookup_path, [*_PREFIX_COLUMNS, "city_id"])
    pad = partial(
        pad_events, events, rows=frame.height, width=max_occurrences, fill=0
    )
    lengths = torch.from_numpy(
        frame["placekey_id"].list.len().to_numpy().astype("int64")
    )
    minutes = torch.from_numpy(
        pad(events["_minutes"].to_numpy().astype("float32"))
    )
    valid = torch.arange(max_occurrences)[None, :] < lengths[:, None]
    gaps = gap_tokens_from_event_time_minutes(minutes).masked_fill(~valid, 0)
    return {
        "prefix_codes": torch.from_numpy(
            pad(events.select(_PREFIX_COLUMNS).to_numpy().astype("int64"))
        ),
        "gap_tokens": gaps,
        "lengths": lengths,
        "city_ids": torch.from_numpy(
            events.filter(pl.col("_pos") == 0)["city_id"]
            .to_numpy()
            .astype("int64")
        ),
    }


def build_planner_data(
    data: PlannerConfig,
    layout: RunLayout,
    *,
    force: bool,
) -> dict[str, object]:
    tokenizer = layout.tokenizer
    dataset_path = layout.planner_dataset
    segments = build_weekly_segment_cache(
        stays_path=tokenizer.augmented_processed_visits,
        output=layout.segments,
        min_active_days=data.min_active_days,
        force=force,
    )
    lookup_path = tokenizer.placekey_lookup
    codebook_size = int(load_codebooks(tokenizer.tokenizer_codebooks).shape[1])
    legal_prefixes, catalog_city_ids, city_legal_prefix_mask = _legal_prefixes(
        lookup_path
    )
    centroids = _load_prefix_centroids(
        lookup_path,
        layout.preprocessing.places_index,
        legal_prefixes=legal_prefixes,
    )

    length = pl.col("placekey_id").list.len()
    frame = pl.read_parquet(
        layout.segments, columns=["_split", "placekey_id", "_timestamps"]
    ).filter(pl.col("_split").is_in(["train", "val"]))
    truncated_rows = frame.filter(length > data.max_occurrences).height
    frame = frame.with_columns(
        pl.col("placekey_id").list.slice(0, data.max_occurrences),
        pl.col("_timestamps").list.slice(0, data.max_occurrences),
    )
    splits = {
        name: frame.filter(pl.col("_split") == split)
        for name, split in (("train", "train"), ("validation", "val"))
    }
    payload = {
        "artifact_type": DATASET_ARTIFACT_TYPE,
        "data_config": {
            "codebook_size": codebook_size,
            "gap_vocab": len(PLANNER_GAP_REPRESENTATIVES_MINUTES),
            "prefix_depth": PREFIX_DEPTH,
            "max_occurrences": data.max_occurrences,
            "city_conditioning": data.city_conditioning,
            "gap_bucket_edges_minutes": PLANNER_GAP_BUCKET_EDGES_MINUTES,
            "gap_duration_minutes": PLANNER_GAP_REPRESENTATIVES_MINUTES,
            "observation_window_minutes": OBSERVATION_WINDOW_MINUTES,
        },
        "legal_prefixes": legal_prefixes,
        "legal_prefix_centroids": centroids[0],
        "legal_prefix_centroid_available": centroids[1],
        "catalog_city_ids": catalog_city_ids,
        "city_legal_prefix_mask": city_legal_prefix_mask,
        **{
            name: _materialize_split(
                split, lookup_path, max_occurrences=data.max_occurrences
            )
            for name, split in splits.items()
        },
    }
    save_torch_atomic(dataset_path, payload, force=force)
    return {
        "segments": segments,
        "dataset": {
            "output_path": str(dataset_path),
            "train_rows": splits["train"].height,
            "validation_rows": splits["validation"].height,
            "legal_prefixes": int(legal_prefixes.shape[0]),
            "truncated_rows": truncated_rows,
            "mean_length": float(frame.select(length.mean()).item()),
            "centroid_coverage": float(centroids[1].float().mean()),
        },
    }


BANK_SPLITS = ("train", "val", "test")


def _split_segments(
    segments_path: Path | str, split: str, *, min_length: int, max_events: int
) -> pl.LazyFrame:
    length = pl.col("placekey_id").list.len()
    return (
        pl.scan_parquet(segments_path)
        .filter(pl.col("_split") == split)
        .filter(length.is_between(min_length, max_events))
    )


def _build_split(
    segments_path: Path,
    lookup_path: Path,
    *,
    split: str,
    max_events: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = cast(
        pl.DataFrame,
        _split_segments(
            segments_path,
            split,
            min_length=SEGMENT_MIN_LENGTH,
            max_events=max_events,
        ).collect(),
    )
    columns = list(sid_code_columns(pl.read_parquet_schema(lookup_path)))
    events = segment_events(frame, lookup_path, columns)
    pad = partial(pad_events, events, rows=frame.height, width=max_events)
    tokens = time_tokens_from_week_minutes(
        torch.from_numpy(events["_minutes"].to_numpy())
    )
    return (
        pad(events.select(columns).to_numpy().astype(np.int16), fill=0),
        pad(tokens.numpy().astype(np.int16), fill=-1),
        pad(np.ones(events.height, dtype=np.bool_), fill=False),
    )


def build_bank(
    *,
    lookup_path: Path,
    segments_path: Path,
    output_path: Path,
    max_events: int,
) -> dict[str, object]:
    parts = [
        _build_split(
            segments_path, lookup_path, split=split, max_events=max_events
        )
        for split in BANK_SPLITS
    ]
    row_counts = {
        split: len(part[0])
        for split, part in zip(BANK_SPLITS, parts, strict=True)
    }
    payload = {
        "sid_codes": torch.from_numpy(np.concatenate([p[0] for p in parts])),
        "time_tokens": torch.from_numpy(np.concatenate([p[1] for p in parts])),
        "valid_mask": torch.from_numpy(np.concatenate([p[2] for p in parts])),
        "row_counts": row_counts,
        "source": {
            "lookup_path": str(lookup_path),
            "segments_path": str(segments_path),
            "max_events": max_events,
            "min_length": SEGMENT_MIN_LENGTH,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "output": str(output_path),
        "rows": sum(row_counts.values()),
        "width": max_events,
        "codebook_depth": int(payload["sid_codes"].shape[-1]),
        "row_counts": row_counts,
    }


def load_bank_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def bank_split_window(payload: dict[str, Any], split: str) -> tuple[int, int]:
    counts = payload["row_counts"]
    offset = sum(
        int(counts[name]) for name in BANK_SPLITS[: BANK_SPLITS.index(split)]
    )
    return offset, int(counts[split])


def bank_segments(
    payload: dict[str, Any],
    split: str,
    columns: list[str],
    *,
    rows: int | None = None,
) -> pl.DataFrame:
    offset, count = bank_split_window(payload, split)
    rows = count if rows is None else min(rows, count)
    source = payload["source"]
    frame = cast(
        pl.DataFrame,
        _split_segments(
            source["segments_path"],
            split,
            min_length=int(source["min_length"]),
            max_events=int(source["max_events"]),
        )
        .head(rows)
        .select(pl.col("placekey_id").list.len().alias("_length"), *columns)
        .collect(),
    )
    expected = payload["valid_mask"][offset : offset + rows].sum(dim=1).long()
    lengths = torch.from_numpy(frame["_length"].to_numpy().astype(np.int64))
    if frame.height != rows or not torch.equal(lengths, expected):
        raise ValueError(f"segments do not align with the bank's {split} rows")
    return frame.drop("_length")


def bank_split_city_ids(
    payload: dict[str, Any], split: str, *, rows: int | None = None
) -> torch.Tensor:
    frame = bank_segments(payload, split, [SEGMENT_CITY_COLUMN], rows=rows)
    return torch.from_numpy(frame[SEGMENT_CITY_COLUMN].to_numpy()).long()


def build_realizer_data(
    settings: RealizerConfig, layout: RunLayout
) -> dict[str, object]:
    return build_bank(
        lookup_path=layout.tokenizer.placekey_lookup,
        segments_path=layout.segments,
        output_path=layout.bank,
        max_events=settings.max_events,
    )


__all__ = [
    "DATASET_ARTIFACT_TYPE",
    "N_PART_OF_DAY_BINS",
    "N_TIME_TOKENS",
    "SEGMENT_CITY_COLUMN",
    "bank_segments",
    "bank_split_city_ids",
    "bank_split_window",
    "build_bank",
    "build_planner_data",
    "build_realizer_data",
    "load_bank_payload",
    "reconstruct_event_time_minutes",
    "time_tokens_from_week_minutes",
]
