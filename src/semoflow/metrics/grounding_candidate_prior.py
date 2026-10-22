from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

from semoflow.metrics.prefix_time_suffix_evaluation import (
    CANDIDATE_PRIOR_FORMAT,
)
from semoflow.realizer.bank import (
    load_bank_payload,
    resolve_bank_split_window,
)


def build_prior(bank_path: Path, split: str = "train") -> dict[str, Any]:
    if split != "train":
        raise ValueError(
            "a candidate prior may only be fitted on the train split; "
            f"got {split!r}"
        )
    payload = load_bank_payload(bank_path)
    _offset, count = resolve_bank_split_window(payload, split, path=bank_path)
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("realizer bank is missing source metadata")
    skipped = source.get("skipped_missing_sid", {})
    if isinstance(skipped, dict) and int(skipped.get(split, 0)):
        raise ValueError(
            f"cannot build a {split} candidate prior from a bank with "
            f"skipped_missing_sid[{split!r}]={int(skipped.get(split, 0))}; "
            "rebuild the bank without skipped rows"
        )
    segments_value = source.get("segments_path")
    if not isinstance(segments_value, str):
        raise ValueError("bank source metadata is missing segments_path")
    max_events = int(source.get("max_events", payload["sid_codes"].size(1)))
    min_length = int(source.get("min_length", 1))

    frame = (
        pl.scan_parquet(segments_value)
        .filter(pl.col("_split") == split)
        .filter(
            pl.col("placekey_id").list.len().is_between(min_length, max_events)
        )
        .head(int(count))
        .collect()
    )
    if frame.height != int(count):
        raise ValueError(
            f"segment table yielded {frame.height} {split} rows; "
            f"expected {int(count)}"
        )
    counter: Counter[int] = Counter()
    for row in frame.iter_rows(named=True):
        counter.update(int(value) for value in row["placekey_id"])
    if not counter:
        raise ValueError("no POI occurrences found in the train window")
    return {
        "format": CANDIDATE_PRIOR_FORMAT,
        "split": split,
        "bank": str(bank_path),
        "segments_path": segments_value,
        "min_length": min_length,
        "max_events": max_events,
        "rows": int(count),
        "events": int(sum(counter.values())),
        "distinct_pois": len(counter),
        "counts": {str(key): int(value) for key, value in counter.items()},
    }


__all__ = ["build_prior"]
