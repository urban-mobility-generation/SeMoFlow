from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import torch

from semoflow.config import (
    RealizerConfig,
    RunLayout,
)
from semoflow.preprocess import SEGMENT_CITY_COLUMN
from semoflow.realizer.time_tokens import time_tokens_from_timestamps
from semoflow.tokenization.sids import sid_code_columns

BANK_SPLITS = ("train", "val", "test")
MIN_LENGTH = 5


def _split_segments(
    segments_path: Path | str, split: str, *, min_length: int, max_events: int
) -> pl.LazyFrame:
    length = pl.col("placekey_id").list.len()
    return (
        pl.scan_parquet(segments_path)
        .filter(pl.col("_split") == split)
        .filter(length.is_between(min_length, max_events))
    )


def _load_lookup(path: Path) -> dict[int, np.ndarray]:
    columns = list(
        sid_code_columns(pl.scan_parquet(path).collect_schema().names())
    )
    frame = pl.read_parquet(path).select(["placekey_id", *columns])
    return {
        int(row[0]): np.asarray(row[1:], dtype=np.int16)
        for row in frame.iter_rows()
    }


def _build_split(
    segments_path: Path,
    lookup: dict[int, np.ndarray],
    *,
    split: str,
    max_events: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = cast(
        pl.DataFrame,
        _split_segments(
            segments_path, split, min_length=MIN_LENGTH, max_events=max_events
        ).collect(),
    )
    if frame.is_empty():
        raise ValueError(f"no usable {split} segments found")
    rows = frame.iter_rows(named=True)
    selected: list[tuple[np.ndarray, np.ndarray]] = []
    for row in rows:
        codes = np.stack([lookup[int(pid)] for pid in row["placekey_id"]])
        selected.append(
            (codes, time_tokens_from_timestamps(row["_timestamps"]))
        )
    depth = int(selected[0][0].shape[1])
    sid_codes = np.zeros((len(selected), max_events, depth), dtype=np.int16)
    time_tokens = np.full((len(selected), max_events), -1, dtype=np.int16)
    valid_mask = np.zeros((len(selected), max_events), dtype=np.bool_)
    for index, (codes, times) in enumerate(selected):
        length = len(codes)
        sid_codes[index, :length] = codes
        time_tokens[index, :length] = times
        valid_mask[index, :length] = True
    return sid_codes, time_tokens, valid_mask


def build_bank(
    *,
    lookup_path: Path,
    segments_path: Path,
    output_path: Path,
    max_events: int,
) -> dict[str, object]:
    lookup = _load_lookup(lookup_path)
    parts = [
        _build_split(segments_path, lookup, split=split, max_events=max_events)
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
            "min_length": MIN_LENGTH,
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
    "BANK_SPLITS",
    "bank_segments",
    "bank_split_city_ids",
    "bank_split_window",
    "build_bank",
    "build_realizer_data",
    "load_bank_payload",
]
