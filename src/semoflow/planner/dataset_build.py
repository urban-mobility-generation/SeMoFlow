from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import torch

from semoflow.config.artifacts import RunLayout
from semoflow.config.schema import (
    PlannerDataConfig,
)
from semoflow.planner.artifacts import (
    DATASET_ARTIFACT_TYPE,
    save_torch_atomic,
)
from semoflow.planner.gap_codec import (
    PLANNER_GAP_BUCKET_EDGES_MINUTES,
    PLANNER_GAP_CODEC_VERSION,
    PLANNER_GAP_REPRESENTATIVES_MINUTES,
    gap_tokens_from_event_time_minutes,
)
from semoflow.preprocess.weekly_segments import build_weekly_segment_cache
from semoflow.tokenization.decoding.codebooks import load_codebooks
from semoflow.tokenization.token_space import sid_code_columns

OBSERVATION_WINDOW_MINUTES = 7.0 * 24.0 * 60.0
PREFIX_DEPTH = 3


def _event_minutes(timestamps: list[datetime]) -> torch.Tensor:
    """Minutes since the Monday midnight that starts the first visit's week."""
    first = timestamps[0]
    week_start = first.replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=first.weekday())
    return torch.tensor(
        [(value - week_start).total_seconds() / 60.0 for value in timestamps],
        dtype=torch.float32,
    )


def _load_prefix_lookup(
    lookup_path: Path, *, prefix_depth: int
) -> tuple[
    dict[int, tuple[torch.Tensor, int]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Map placekey ids to (prefix, city) and derive prefix legality."""
    names = pl.read_parquet_schema(lookup_path)
    selected = list(sid_code_columns(names))[:prefix_depth]
    has_city = "city_id" in names
    frame = pl.read_parquet(
        lookup_path,
        columns=["placekey_id", *(["city_id"] if has_city else []), *selected],
    )
    mapping: dict[int, tuple[torch.Tensor, int]] = {}
    city_tuples: dict[int, set[tuple[int, ...]]] = {}
    for row in frame.iter_rows():
        city_id = int(row[1]) if has_city else 0
        codes = tuple(int(value) for value in row[1 + has_city :])
        mapping[int(row[0])] = (torch.tensor(codes, dtype=torch.long), city_id)
        city_tuples.setdefault(city_id, set()).add(codes)

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
    return mapping, legal_prefixes, catalog_city_ids, city_legal_prefix_mask


def _load_prefix_centroids(
    lookup_path: Path,
    places_path: Path,
    *,
    legal_prefixes: torch.Tensor,
    prefix_depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean POI coordinate of every legal prefix that has located POIs."""
    selected = list(sid_code_columns(pl.read_parquet_schema(lookup_path)))[
        :prefix_depth
    ]
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
        position = index_of[tuple(int(value) for value in row[:prefix_depth])]
        centroids[position] = torch.tensor(row[prefix_depth:])
        available[position] = True
    return centroids, available


def _materialize_split(
    frame: pl.DataFrame,
    *,
    lookup: dict[int, tuple[torch.Tensor, int]],
    prefix_depth: int,
    max_occurrences: int,
) -> dict[str, torch.Tensor]:
    rows = frame.height
    prefixes = torch.zeros(
        (rows, max_occurrences, prefix_depth), dtype=torch.long
    )
    gaps = torch.zeros((rows, max_occurrences), dtype=torch.long)
    lengths = torch.empty(rows, dtype=torch.long)
    city_ids = torch.empty(rows, dtype=torch.long)
    for row_index, row in enumerate(frame.iter_rows(named=True)):
        lookup_rows = [lookup[int(value)] for value in row["placekey_id"]]
        length = len(lookup_rows)
        lengths[row_index] = length
        city_ids[row_index] = lookup_rows[0][1]
        prefixes[row_index, :length] = torch.stack(
            [prefix for prefix, _ in lookup_rows]
        )
        gaps[row_index, :length] = gap_tokens_from_event_time_minutes(
            _event_minutes(list(row["_timestamps"])),
            bucket_edges_minutes=PLANNER_GAP_BUCKET_EDGES_MINUTES,
        )
    return {
        "prefix_codes": prefixes,
        "gap_tokens": gaps,
        "lengths": lengths,
        "city_ids": city_ids,
    }


def build_planner_data(
    data: PlannerDataConfig,
    layout: RunLayout,
    *,
    force: bool = False,
) -> dict[str, object]:
    """Build the weekly segments, then the planner dataset from them.

    Weeks longer than ``max_occurrences`` keep their first visits.
    """
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
    lookup, legal_prefixes, catalog_city_ids, city_legal_prefix_mask = (
        _load_prefix_lookup(lookup_path, prefix_depth=PREFIX_DEPTH)
    )
    centroids = _load_prefix_centroids(
        lookup_path,
        layout.preprocessing.places_index,
        legal_prefixes=legal_prefixes,
        prefix_depth=PREFIX_DEPTH,
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
            "gap_codec_version": PLANNER_GAP_CODEC_VERSION,
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
                split,
                lookup=lookup,
                prefix_depth=PREFIX_DEPTH,
                max_occurrences=data.max_occurrences,
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


__all__ = ["OBSERVATION_WINDOW_MINUTES", "PREFIX_DEPTH", "build_planner_data"]
