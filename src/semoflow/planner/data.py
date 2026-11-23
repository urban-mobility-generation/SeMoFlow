from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
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
from semoflow.poi.sids import load_codebooks, sid_code_names
from semoflow.preprocess import (
    build_weekly_segment_cache,
    pad_events,
    segment_events,
)
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
    durations = torch.diff(
        event_time_minutes,
        prepend=torch.zeros_like(event_time_minutes[..., :1]),
    )
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


__all__ = [
    "OccurrencePlannerDataset",
    "SeededShuffleSampler",
    "build_occurrence_tensor_dataloader",
    "build_planner_data",
    "load_occurrence_planner_dataset",
]
