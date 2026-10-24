from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from semoflow.config.schemas.realizer_config import (
    PrefixTimeSuffixLatentFlowRealizerConfig,
)
from semoflow.preprocess.weekly_segments import SEGMENT_CITY_COLUMN
from semoflow.realizer.time_tokens import (
    N_TIME_TOKENS,
    time_tokens_from_timestamps,
)
from semoflow.tokenization.token_space import sid_code_columns


def _load_lookup(path: Path) -> dict[int, np.ndarray]:
    columns = list(
        sid_code_columns(pl.scan_parquet(path).collect_schema().names())
    )
    if not columns:
        raise ValueError(f"lookup carries no sem_code_* columns: {path}")
    frame = pl.read_parquet(path).select(["placekey_id", *columns])
    lookup: dict[int, np.ndarray] = {}
    for row in frame.iter_rows():
        lookup[int(row[0])] = np.asarray(row[1:], dtype=np.int16)
    if not lookup:
        raise ValueError(f"SID lookup is empty: {path}")
    return lookup


def _build_split(
    segments_path: Path,
    lookup: dict[int, np.ndarray],
    *,
    split: str,
    max_events: int,
    min_length: int,
    max_segments: int,
    allow_empty: bool = False,
    width: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    query = (
        pl.scan_parquet(segments_path)
        .filter(pl.col("_split") == split)
        .filter(
            pl.col("placekey_id").list.len().is_between(min_length, max_events)
        )
    )
    if max_segments > 0:
        query = query.head(max_segments)
    frame = query.collect()
    rows = frame.iter_rows(named=True)
    selected: list[tuple[np.ndarray, np.ndarray]] = []
    skipped = 0
    for row in rows:
        placekey_ids = [int(value) for value in row["placekey_id"]]
        timestamps = list(row["_timestamps"])
        if len(placekey_ids) != len(timestamps):
            raise ValueError(
                f"segment {row['_seg_id']} has misaligned IDs/timestamps"
            )
        sid_rows = [lookup.get(placekey_id) for placekey_id in placekey_ids]
        if any(codes is None for codes in sid_rows):
            skipped += 1
            continue
        codes = np.stack(
            [codes for codes in sid_rows if codes is not None], axis=0
        )
        times = time_tokens_from_timestamps(timestamps)
        if len(times) != len(codes):
            raise ValueError(
                f"segment {row['_seg_id']} has invalid time width"
            )
        selected.append((codes, times))
    if not selected:
        if allow_empty:
            sid_width = width or 6
            return (
                np.zeros((0, max_events, sid_width), dtype=np.int16),
                np.zeros((0, max_events), dtype=np.int16),
                np.zeros((0, max_events), dtype=bool),
                skipped,
            )
        raise ValueError(f"no usable {split} segments found")
    sid_width = int(selected[0][0].shape[1])
    sid_codes = np.zeros(
        (len(selected), max_events, sid_width), dtype=np.int16
    )
    time_tokens = np.full((len(selected), max_events), -1, dtype=np.int16)
    valid_mask = np.zeros((len(selected), max_events), dtype=np.bool_)
    for index, (codes, times) in enumerate(selected):
        length = len(codes)
        sid_codes[index, :length] = codes
        time_tokens[index, :length] = times
        valid_mask[index, :length] = True
    return sid_codes, time_tokens, valid_mask, skipped


def build_bank(
    *,
    lookup_path: Path,
    segments_path: Path,
    output_path: Path,
    max_events: int,
    min_length: int = 5,
    max_train_segments: int = 0,
    max_val_segments: int = 0,
    max_test_segments: int = 0,
    allow_empty_splits: bool = False,
) -> dict[str, object]:
    if max_events < 1 or min_length < 1 or min_length > max_events:
        raise ValueError("event length bounds are invalid")
    lookup = _load_lookup(lookup_path)
    by_split: dict[str, tuple] = {}
    skipped: dict[str, int] = {}
    width: int | None = None
    for split, cap in (
        ("val", max_val_segments),
        ("train", max_train_segments),
        ("test", max_test_segments),
    ):
        part = _build_split(
            segments_path,
            lookup,
            split=split,
            max_events=max_events,
            min_length=min_length,
            max_segments=cap,
            allow_empty=allow_empty_splits and split != "val",
            width=width,
        )
        if width is None and len(part[0]):
            width = int(part[0].shape[2])
        by_split[split] = part[:3]
        skipped[split] = part[3]
    parts = [by_split["train"], by_split["val"], by_split["test"]]
    sid_codes = np.concatenate([part[0] for part in parts], axis=0)
    time_tokens = np.concatenate([part[1] for part in parts], axis=0)
    valid_mask = np.concatenate([part[2] for part in parts], axis=0)
    train_count = len(parts[0][0])
    val_count = len(parts[1][0])
    test_count = len(parts[2][0])
    payload = {
        "schema_version": 1,
        "sid_codes": torch.from_numpy(sid_codes),
        "time_tokens": torch.from_numpy(time_tokens),
        "valid_mask": torch.from_numpy(valid_mask),
        "row_counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        },
        "codebook_size": 64,
        "time_vocab": N_TIME_TOKENS,
        "source": {
            "lookup_path": str(lookup_path),
            "segments_path": str(segments_path),
            "max_events": max_events,
            "min_length": min_length,
            "skipped_missing_sid": skipped,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "output": str(output_path),
        "rows": len(sid_codes),
        "width": int(max_events),
        "codebook_depth": int(sid_codes.shape[-1]),
        "row_counts": payload["row_counts"],
        "skipped_missing_sid": skipped,
    }


BANK_SPLITS = ("train", "val", "test")


def load_bank_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"realizer bank {path} must contain a mapping")
    required = ("sid_codes", "time_tokens", "valid_mask", "row_counts")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"realizer bank {path} is missing {missing}")
    for key in ("sid_codes", "time_tokens", "valid_mask"):
        if not isinstance(payload[key], torch.Tensor):
            raise TypeError(f"realizer bank {path} {key} must be a tensor")
    if payload["sid_codes"].ndim != 3:
        raise ValueError("realizer bank sid_codes must have shape [N, L, M]")
    if payload["time_tokens"].shape != payload["valid_mask"].shape:
        raise ValueError("realizer bank time_tokens and valid_mask must align")
    if payload["valid_mask"].shape != payload["sid_codes"].shape[:2]:
        raise ValueError("realizer bank valid_mask must align with sid_codes")
    return payload


def resolve_bank_split_window(
    payload: dict[str, Any],
    split: str,
    *,
    path: Path,
) -> tuple[int, int]:
    if split not in BANK_SPLITS:
        raise ValueError(
            f"unknown bank split {split!r}; expected one of {BANK_SPLITS}"
        )
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, dict):
        raise ValueError(
            f"bank {path} carries no row_counts metadata; split {split!r} "
            "cannot be resolved"
        )
    if split not in row_counts:
        raise ValueError(
            f"bank {path} row_counts missing partition {split!r}; "
            f"available: {sorted(row_counts)}"
        )
    offset = 0
    for name in BANK_SPLITS:
        if name == split:
            break
        offset += int(row_counts.get(name, 0))
    return offset, int(row_counts[split])


def bank_split_city_ids(
    payload: dict[str, Any],
    bank_path: Path,
    *,
    split: str,
    valid_mask: torch.Tensor,
    time_tokens: torch.Tensor | None = None,
) -> torch.LongTensor:
    if valid_mask.ndim != 2 or valid_mask.shape[0] == 0:
        raise ValueError("valid_mask must be a non-empty [N, L] mask")
    if time_tokens is not None and time_tokens.shape != valid_mask.shape:
        raise ValueError("time_tokens must align with valid_mask")
    count = int(valid_mask.shape[0])
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("realizer bank is missing source metadata")
    segments_value = source.get("segments_path")
    if not isinstance(segments_value, str):
        raise ValueError(
            "realizer bank source metadata is missing segments_path"
        )
    skipped = source.get("skipped_missing_sid", {})
    if isinstance(skipped, dict) and int(skipped.get(split, 0)):
        raise ValueError(
            f"cannot align {split} rows: the bank skipped segments with "
            "missing SIDs"
        )
    _, declared_count = resolve_bank_split_window(
        payload, split, path=bank_path
    )
    if count > declared_count:
        raise ValueError(
            f"requested {count} rows but {split} has {declared_count}"
        )
    max_events = int(source.get("max_events", payload["sid_codes"].size(1)))
    min_length = int(source.get("min_length", 1))
    lazy = pl.scan_parquet(segments_value)
    if SEGMENT_CITY_COLUMN not in lazy.collect_schema().names():
        raise ValueError(
            "city-conditioned bank source carries no `_city_id` column"
        )
    columns = [
        pl.col(SEGMENT_CITY_COLUMN),
        pl.col("placekey_id").list.len().alias("_length"),
    ]
    if time_tokens is not None:
        columns.append(pl.col("_timestamps"))
    frame = (
        lazy.filter(pl.col("_split") == split)
        .filter(
            pl.col("placekey_id").list.len().is_between(min_length, max_events)
        )
        .head(count)
        .select(columns)
        .collect()
    )
    if frame.height < count:
        raise ValueError(
            f"segment source has {frame.height} {split} rows; expected {count}"
        )
    expected_lengths = valid_mask.sum(dim=1).long().cpu()
    lengths = torch.tensor(frame["_length"].to_numpy(), dtype=torch.long)
    if not torch.equal(lengths, expected_lengths):
        mismatch = torch.where(lengths.ne(expected_lengths))[0][0]
        raise ValueError(
            f"segment and bank lengths disagree at {split} row {int(mismatch)}"
        )
    if time_tokens is not None:
        for row_index, timestamps in enumerate(frame["_timestamps"]):
            length = int(lengths[row_index])
            observed = [
                int(v) for v in time_tokens_from_timestamps(timestamps)
            ]
            expected = [int(v) for v in time_tokens[row_index, :length]]
            if observed != expected:
                raise ValueError(
                    "bank time tokens disagree with source timestamps at "
                    f"{split} row {row_index}"
                )
    return torch.tensor(
        frame[SEGMENT_CITY_COLUMN].to_numpy(), dtype=torch.long
    )


def build_realizer_data(
    settings: PrefixTimeSuffixLatentFlowRealizerConfig,
) -> dict[str, object]:
    """Build the bank ``settings`` trains on, keyed by its catalog lookup."""
    data, lookup = settings.data, settings.catalog_path
    if data is None or lookup is None:
        raise ValueError(
            "building the realizer bank needs realizer.data and "
            "realizer.catalog_path"
        )
    return build_bank(
        lookup_path=lookup,
        segments_path=data.segments_path,
        output_path=settings.bank_path,
        max_events=data.max_events,
    )


__all__ = [
    "BANK_SPLITS",
    "bank_split_city_ids",
    "build_bank",
    "build_realizer_data",
    "load_bank_payload",
    "resolve_bank_split_window",
]
