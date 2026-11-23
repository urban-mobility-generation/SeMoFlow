from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Sampler,
    SequentialSampler,
    TensorDataset,
)

from semoflow.config import PlannerDataConfig, RunLayout
from semoflow.planner.layers import LegalPrefixIndex
from semoflow.planner.model import OccurrenceBatch, OccurrencePlannerConfig
from semoflow.poi.sids import load_codebooks, sid_code_columns
from semoflow.preprocess import build_weekly_segment_cache
from semoflow.utils import load_artifact, save_torch_atomic

PLANNER_GAP_CODEC_VERSION = "positive_duration_v1"
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
    durations = torch.diff(event_time_minutes, prepend=torch.zeros(1))
    edges = torch.tensor(
        PLANNER_GAP_BUCKET_EDGES_MINUTES, dtype=durations.dtype
    )
    return torch.bucketize(durations, edges, right=False).to(torch.long)


DATASET_ARTIFACT_TYPE = "occurrence_planner_dataset"


@dataclass(frozen=True, slots=True)
class OccurrencePlannerDataConfig:
    codebook_size: int
    gap_vocab: int
    prefix_depth: int
    max_occurrences: int
    city_conditioning: bool
    gap_codec_version: str
    gap_bucket_edges_minutes: tuple[float, ...]
    gap_duration_minutes: tuple[float, ...]
    observation_window_minutes: float


@dataclass(frozen=True, slots=True)
class OccurrencePlannerSplit:
    prefix_codes: torch.Tensor
    gap_tokens: torch.Tensor
    lengths: torch.Tensor
    city_ids: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.lengths.shape[0])


@dataclass(frozen=True, slots=True)
class OccurrencePlannerDataset:
    config: OccurrencePlannerDataConfig
    legal_prefixes: torch.Tensor
    catalog_city_ids: torch.Tensor
    city_legal_prefix_mask: torch.Tensor
    train: OccurrencePlannerSplit
    validation: OccurrencePlannerSplit
    source_path: Path
    legal_prefix_centroids: torch.Tensor | None = None
    legal_prefix_centroid_available: torch.Tensor | None = None


def load_occurrence_planner_dataset(path: Path) -> OccurrencePlannerDataset:
    payload = load_artifact(path, DATASET_ARTIFACT_TYPE)
    return OccurrencePlannerDataset(
        config=OccurrencePlannerDataConfig(**payload["data_config"]),
        legal_prefixes=payload["legal_prefixes"],
        legal_prefix_centroids=payload["legal_prefix_centroids"],
        legal_prefix_centroid_available=payload[
            "legal_prefix_centroid_available"
        ],
        catalog_city_ids=payload["catalog_city_ids"],
        city_legal_prefix_mask=payload["city_legal_prefix_mask"],
        train=OccurrencePlannerSplit(**payload["train"]),
        validation=OccurrencePlannerSplit(**payload["validation"]),
        source_path=Path(path).resolve(),
    )


def _shift_right(
    values: torch.Tensor, valid_mask: torch.Tensor, fill: float
) -> torch.Tensor:
    shifted = torch.full_like(values, fill)
    previous_valid = valid_mask[:, 1:]
    if values.ndim == 3:
        previous_valid = previous_valid[..., None]
    shifted[:, 1:] = torch.where(previous_valid, values[:, :-1], fill)
    return shifted


def collate_occurrence_tensors(
    tensors: tuple[torch.Tensor, ...],
    *,
    config: OccurrencePlannerConfig,
    legality: LegalPrefixIndex,
) -> OccurrenceBatch:
    prefix_codes, gap_tokens, lengths, city_ids = tensors
    batch_size = prefix_codes.shape[0]
    device = prefix_codes.device
    max_len = int(lengths.max().item())

    sub_prefix = prefix_codes[:, :max_len]
    sub_gaps = gap_tokens[:, :max_len]
    valid_mask = (
        torch.arange(max_len, device=device)[None, :] < lengths[:, None]
    )
    default_prefix = (
        legality.codes[0]
        .to(device)
        .view(1, 1, -1)
        .expand(batch_size, max_len, -1)
    )
    target_prefixes = torch.where(
        valid_mask[..., None], sub_prefix, default_prefix
    )
    target_gaps = torch.where(valid_mask, sub_gaps, 0)
    event_times = reconstruct_event_time_minutes(
        target_gaps,
        valid_mask,
        representative_minutes=config.gap_duration_minutes,
        observation_window_minutes=config.observation_window_minutes,
    )
    return OccurrenceBatch(
        previous_prefix_codes=_shift_right(sub_prefix, valid_mask, -1),
        previous_gap_tokens=_shift_right(sub_gaps, valid_mask, -1),
        previous_event_time_minutes=_shift_right(event_times, valid_mask, 0.0),
        target_prefix_codes=target_prefixes,
        target_gap_tokens=target_gaps,
        valid_mask=valid_mask,
        lengths=lengths,
        city_ids=city_ids,
    )


class SeededShuffleSampler(Sampler[int]):
    def __init__(self, size: int, *, seed: int) -> None:
        self._size = int(size)
        self._random = random.Random(seed)

    def __iter__(self) -> Iterator[int]:
        order = list(range(self._size))
        self._random.shuffle(order)
        return iter(order)

    def __len__(self) -> int:
        return self._size


def build_occurrence_tensor_dataloader(
    split: OccurrencePlannerSplit,
    *,
    config: OccurrencePlannerConfig,
    legality: LegalPrefixIndex,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    dataset = TensorDataset(
        split.prefix_codes, split.gap_tokens, split.lengths, split.city_ids
    )
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=BatchSampler(
            SequentialSampler(dataset) if sampler is None else sampler,
            batch_size=batch_size,
            drop_last=False,
        ),
        collate_fn=partial(
            collate_occurrence_tensors, config=config, legality=legality
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


OBSERVATION_WINDOW_MINUTES = 7.0 * 24.0 * 60.0
PREFIX_DEPTH = 3


def _event_minutes(timestamps: list[datetime]) -> torch.Tensor:
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
    selected = list(sid_code_columns(pl.read_parquet_schema(lookup_path)))
    frame = pl.read_parquet(
        lookup_path,
        columns=["placekey_id", "city_id", *selected[:prefix_depth]],
    )
    mapping: dict[int, tuple[torch.Tensor, int]] = {}
    city_tuples: dict[int, set[tuple[int, ...]]] = {}
    for placekey_id, city_id, *prefix in frame.iter_rows():
        codes = tuple(int(value) for value in prefix)
        mapping[int(placekey_id)] = (
            torch.tensor(codes, dtype=torch.long),
            int(city_id),
        )
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
            _event_minutes(list(row["_timestamps"]))
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


__all__ = [
    "OccurrencePlannerDataset",
    "SeededShuffleSampler",
    "build_occurrence_tensor_dataloader",
    "build_planner_data",
    "load_occurrence_planner_dataset",
]
