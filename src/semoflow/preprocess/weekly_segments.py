from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

CITY_COLUMN = "city_id"
SPLIT_COLUMN = "split"
SEGMENT_CITY_COLUMN = "_city_id"
SEGMENT_CACHE_FORMAT_VERSION = 2


def _read_stays(path: Path) -> tuple[pl.DataFrame, str]:
    available = set(pl.read_parquet_schema(path))
    ts_col = "local_ts" if "local_ts" in available else "timestamp"
    columns = ["device_key", ts_col, "placekey_id", SPLIT_COLUMN]
    missing = sorted(set(columns).difference(available))
    if missing:
        raise ValueError(
            "weekly segmentation requires parquet columns: "
            + ", ".join(missing)
        )
    if CITY_COLUMN in available:
        columns.append(CITY_COLUMN)
    return pl.read_parquet(path, columns=columns), ts_col


def _segment_events(
    stays: pl.DataFrame,
    *,
    ts_col: str,
    min_seq_len: int,
    min_active_days: int,
) -> pl.DataFrame:
    window = ["device_key", "_week"]
    stays = (
        stays.sort(["device_key", ts_col])
        .with_columns(pl.col(ts_col).dt.truncate("1w").alias("_week"))
        .with_columns(
            pl.int_range(0, pl.len()).over(window).alias("_pos"),
            pl.len().over(window).alias("_length"),
            pl.col(ts_col)
            .dt.date()
            .n_unique()
            .over(window)
            .alias("_active_days"),
        )
    )
    n_before = stays.select(pl.struct(window).n_unique()).item()
    stays = stays.filter(
        (pl.col("_length") >= min_seq_len)
        & (pl.col("_active_days") >= min_active_days)
    )
    logger.info(
        "weekly quality gate: %s -> %s windows (min_seq_len=%d, min_active_days=%d)",
        f"{n_before:,}",
        f"{stays.select(pl.struct(window).n_unique()).item():,}",
        min_seq_len,
        min_active_days,
    )
    return stays.with_columns(
        (pl.struct(window).rank("dense") - 1).cast(pl.UInt32).alias("_seg_id")
    )


def build_weekly_segment_cache(
    *,
    stays_path: Path,
    output: Path,
    min_seq_len: int = 5,
    min_active_days: int = 5,
    max_seq_len: int = 128,
    force: bool = False,
) -> dict[str, Any]:
    if min_seq_len < 1:
        raise ValueError("min_seq_len must be >= 1")
    if min_active_days < 1:
        raise ValueError("min_active_days must be >= 1")
    if max_seq_len < min_seq_len:
        raise ValueError("max_seq_len must be >= min_seq_len")
    spec_path = output.with_suffix(output.suffix + ".json")
    if not force:
        for path in (output, spec_path):
            if path.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing artifact: {path}"
                )

    stays, ts_col = _read_stays(stays_path)
    events = (
        _segment_events(
            stays,
            ts_col=ts_col,
            min_seq_len=min_seq_len,
            min_active_days=min_active_days,
        )
        .filter(pl.col("_pos") < max_seq_len)
        .sort(["_seg_id", "_pos"])
    )
    aggregations = [
        pl.col("placekey_id"),
        pl.col("device_key").first(),
        pl.col(ts_col).alias("_timestamps"),
        pl.col(SPLIT_COLUMN).first().alias("_split"),
    ]
    if CITY_COLUMN in stays.columns:
        aggregations.append(
            pl.col(CITY_COLUMN).first().alias(SEGMENT_CITY_COLUMN)
        )
    grouped = events.group_by("_seg_id", maintain_order=True).agg(
        *aggregations
    )

    spec = {
        "format_version": SEGMENT_CACHE_FORMAT_VERSION,
        "min_seq_len": min_seq_len,
        "max_seq_len": max_seq_len,
        "min_active_days": min_active_days,
        "split_column": SPLIT_COLUMN,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    grouped.write_parquet(temporary)
    temporary.replace(output)
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    split_counts = {
        str(row["_split"]): int(row["len"])
        for row in grouped.group_by("_split").len().iter_rows(named=True)
    }
    return {
        "output": str(output),
        "spec": str(spec_path),
        "segments": grouped.height,
        "segments_by_split": split_counts,
        **spec,
    }


__all__ = [
    "SEGMENT_CACHE_FORMAT_VERSION",
    "SEGMENT_CITY_COLUMN",
    "build_weekly_segment_cache",
]
