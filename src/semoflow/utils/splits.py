from __future__ import annotations

from semoflow.utils.data import hash_fraction

GLOBAL_SPLIT_SEED = 20260712

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

SPLIT_COLUMN = "split"


def assign_split(
    key: str,
    *,
    seed: int = GLOBAL_SPLIT_SEED,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> str:
    frac = hash_fraction(str(key), seed)
    if frac < train_fraction:
        return "train"
    if frac < train_fraction + val_fraction:
        return "val"
    return "test"


__all__ = [
    "GLOBAL_SPLIT_SEED",
    "SPLIT_COLUMN",
    "TRAIN_FRACTION",
    "VAL_FRACTION",
    "assign_split",
]
