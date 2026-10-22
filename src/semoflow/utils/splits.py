from __future__ import annotations

import polars as pl

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


def with_split_column(
    frame: pl.DataFrame,
    key_column: str,
    *,
    seed: int = GLOBAL_SPLIT_SEED,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    output_column: str = SPLIT_COLUMN,
) -> pl.DataFrame:
    keys = (
        frame.select(pl.col(key_column).cast(pl.Utf8).alias("_split_key"))
        .unique()
        .with_columns(
            pl.col("_split_key")
            .map_elements(
                lambda k: assign_split(
                    k,
                    seed=seed,
                    train_fraction=train_fraction,
                    val_fraction=val_fraction,
                ),
                return_dtype=pl.Utf8,
            )
            .alias(output_column)
        )
    )
    return (
        frame.with_columns(
            pl.col(key_column).cast(pl.Utf8).alias("_split_key")
        )
        .join(keys, on="_split_key", how="left")
        .drop("_split_key")
    )


__all__ = [
    "GLOBAL_SPLIT_SEED",
    "SPLIT_COLUMN",
    "TRAIN_FRACTION",
    "VAL_FRACTION",
    "assign_split",
    "with_split_column",
]
