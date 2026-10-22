from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy import sparse


def _poi_key_column(frame: pl.DataFrame) -> str:
    if "placekey_id" in frame.columns:
        return "placekey_id"
    if "placekey" in frame.columns:
        return "placekey"
    raise ValueError("stays must contain placekey_id or placekey")


def covisit_adjacency(
    stays: pl.DataFrame,
    poi_ids: Sequence[object],
    *,
    tc_hours: float,
    ts_intermediate: int,
    min_covisits: int,
    min_shift: int = 1,
) -> sparse.csr_matrix:
    if min_shift < 1:
        raise ValueError("min_shift must be at least 1")
    if min_shift > ts_intermediate + 1:
        raise ValueError(
            "min_shift exceeds the largest offset ts_intermediate allows"
        )
    n = len(poi_ids)
    poi_col = _poi_key_column(stays)

    ts = "timestamp" if "timestamp" in stays.columns else None
    if n == 0 or not ts:
        return sparse.csr_matrix((n, n), dtype=np.float32)
    exit_expr = (
        pl.col("_start")
        + pl.duration(
            minutes=pl.col("total_dwell_min").fill_null(0).cast(pl.Int64)
        )
        if "total_dwell_min" in stays.columns
        else pl.col("_start")
    )
    base = (
        stays.lazy()
        .select(
            [
                "device_key",
                poi_col,
                ts,
                *(
                    ["total_dwell_min"]
                    if "total_dwell_min" in stays.columns
                    else []
                ),
            ]
        )
        .drop_nulls()
        .with_columns(pl.col(ts).cast(pl.Datetime).alias("_start"))
        .with_columns(
            pl.col("_start").dt.date().alias("_day"), exit_expr.alias("_exit")
        )
        .sort(["device_key", "_start"])
    )
    counts = base.group_by(poi_col).agg(pl.len().alias("n_i")).collect()
    tc_us = int(tc_hours * 3600 * 1000000)
    pairs = []
    for k in range(min_shift, ts_intermediate + 2):
        pairs.append(
            base.with_columns(
                pl.col(poi_col).shift(-k).over("device_key").alias("_dst"),
                pl.col("_start")
                .shift(-k)
                .over("device_key")
                .alias("_dst_start"),
                pl.col("_day").shift(-k).over("device_key").alias("_dst_day"),
            )
            .with_columns(
                (pl.col("_dst_start") - pl.col("_exit"))
                .dt.total_microseconds()
                .alias("_gap")
            )
            .filter(
                pl.col("_dst").is_not_null()
                & (pl.col("_dst_day") == pl.col("_day"))
                & (pl.col(poi_col) != pl.col("_dst"))
                & (pl.col("_gap") >= 0)
                & (pl.col("_gap") <= tc_us)
            )
            .select(
                pl.col(poi_col).alias("src"),
                pl.col("_dst").alias("dst"),
            )
        )
    p_df = (
        pl.concat(pairs)
        .group_by(["src", "dst"])
        .agg(pl.len().alias("n_ij"))
        .filter(pl.col("n_ij") >= min_covisits)
        .collect()
    )
    if p_df.is_empty():
        return sparse.csr_matrix((n, n), dtype=np.float32)
    directed = p_df.join(
        counts.rename({poi_col: "src"}), on="src", how="left"
    ).with_columns(
        (pl.col("n_ij") / pl.col("n_i")).cast(pl.Float32).alias("w")
    )
    total = float(counts["n_i"].sum())
    directed = (
        directed.join(
            counts.rename({poi_col: "dst", "n_i": "n_j"}),
            on="dst",
            how="left",
        )
        .with_columns(
            (
                pl.col("w")
                / (pl.col("n_j").cast(pl.Float64) / total).clip(
                    lower_bound=1e-12
                )
            )
            .cast(pl.Float32)
            .alias("w")
        )
        .drop("n_j")
    )
    lookup = pl.DataFrame(
        {
            "_key": pl.Series(poi_col, list(poi_ids)),
            "_idx": pl.Series("_idx", np.arange(n, dtype=np.int32)),
        }
    )
    valid = (
        directed.join(
            lookup.rename({"_key": "src", "_idx": "_src_idx"}),
            on="src",
            how="inner",
        )
        .join(
            lookup.rename({"_key": "dst", "_idx": "_dst_idx"}),
            on="dst",
            how="inner",
        )
        .select("_src_idx", "_dst_idx", "w")
    )
    if valid.is_empty():
        return sparse.csr_matrix((n, n), dtype=np.float32)
    return sparse.coo_matrix(
        (
            valid["w"].to_numpy().astype(np.float32),
            (
                valid["_src_idx"].to_numpy().astype(np.int32),
                valid["_dst_idx"].to_numpy().astype(np.int32),
            ),
        ),
        shape=(n, n),
    ).tocsr()


def top_k_per_row(
    adjacency: sparse.csr_matrix, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    adjacency = adjacency.tocsr()
    src_out: list[np.ndarray] = []
    dst_out: list[np.ndarray] = []
    indptr = adjacency.indptr
    indices = adjacency.indices
    values = adjacency.data
    for row in range(adjacency.shape[0]):
        start, end = int(indptr[row]), int(indptr[row + 1])
        if end <= start:
            continue
        cols = indices[start:end]
        vals = values[start:end]
        if cols.size > top_k:
            keep = np.argpartition(vals, -top_k)[-top_k:]
            cols = cols[keep]
        src_out.append(np.full(cols.size, row, dtype=np.int64))
        dst_out.append(cols.astype(np.int64))
    if not src_out:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(src_out), np.concatenate(dst_out)


__all__ = ["covisit_adjacency", "top_k_per_row"]
