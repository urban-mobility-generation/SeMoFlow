from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from semoflow.metrics.prefix_time_suffix_evaluation import (
    CANDIDATE_PRIOR_FORMAT,
)
from semoflow.realizer.bank import bank_segments, load_bank_payload


def build_prior(bank_path: Path) -> dict[str, Any]:
    """Per-POI visit counts over the bank's train rows."""
    payload = load_bank_payload(bank_path)
    frame = bank_segments(payload, "train", ["placekey_id"])
    counter = Counter(int(value) for value in frame["placekey_id"].explode())
    if not counter:
        raise ValueError("no POI occurrences found in the train window")
    return {
        "format": CANDIDATE_PRIOR_FORMAT,
        "split": "train",
        "bank": str(bank_path),
        "rows": frame.height,
        "events": sum(counter.values()),
        "distinct_pois": len(counter),
        "counts": {str(key): value for key, value in counter.items()},
    }


__all__ = ["build_prior"]
