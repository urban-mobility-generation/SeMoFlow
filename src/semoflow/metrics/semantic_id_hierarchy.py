from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

_EARTH_KM_PER_DEG = 111.0


def load_hierarchy_frame(
    grounding_index_path: Path,
) -> tuple[pl.DataFrame, list[str]]:
    import pickle

    with grounding_index_path.open("rb") as handle:
        index = pickle.load(handle)
    frame = index.to_frame()
    return frame, list(index.key_columns)


def codebook_utilization_by_level(
    frame: pl.DataFrame, key_columns: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level, column in enumerate(key_columns):
        counts = frame[column].value_counts().sort(column)["count"].to_numpy()
        share = counts / counts.sum()
        entropy_bits = float(-(share * np.log2(share)).sum())
        rows.append(
            {
                "level": level,
                "codes_used": int(counts.size),
                "perplexity": float(2.0**entropy_bits),
                "usage_entropy_bits": entropy_bits,
                "top1_share": float(share.max()),
            }
        )
    return rows


def _modal_purity(
    frame: pl.DataFrame, prefix: list[str], label_column: str
) -> float:
    grouped = (
        frame.group_by([*prefix, label_column])
        .len()
        .group_by(prefix)
        .agg(
            pl.col("len").max().alias("modal"),
            pl.col("len").sum().alias("total"),
        )
    )
    return float(
        cast(Any, grouped["modal"].sum()) / cast(Any, grouped["total"].sum())
    )


def _size_weighted_geo_radius_km(
    frame: pl.DataFrame,
    prefix: list[str],
    latitude_column: str,
    longitude_column: str,
) -> float:
    grouped = (
        frame.group_by(prefix)
        .agg(
            pl.col(latitude_column).std().alias("lat_std"),
            pl.col(latitude_column).mean().alias("lat_mean"),
            pl.col(longitude_column).std().alias("lon_std"),
            pl.len().alias("count"),
        )
        .filter(pl.col("count") > 1)
        .sort(prefix)
    )
    if grouped.height == 0:
        return 0.0
    lat_std = grouped["lat_std"].fill_null(0.0).to_numpy()
    lon_std = grouped["lon_std"].fill_null(0.0).to_numpy()
    lat_mean = grouped["lat_mean"].fill_null(0.0).to_numpy()
    counts = grouped["count"].to_numpy().astype(float)
    lat_km = lat_std * _EARTH_KM_PER_DEG
    lon_km = lon_std * _EARTH_KM_PER_DEG * np.cos(np.radians(lat_mean))
    radius = np.sqrt(lat_km**2 + lon_km**2)
    return float(np.average(radius, weights=counts))


def prefix_depth_hierarchy(
    frame: pl.DataFrame,
    key_columns: list[str],
    *,
    category_column: str | None,
    naics_column: str | None,
    latitude_column: str | None,
    longitude_column: str | None,
) -> list[dict[str, Any]]:
    n_pois = frame.height
    rows: list[dict[str, Any]] = []
    previous_distinct: int | None = None
    previous_bits: float | None = None

    for depth in range(1, len(key_columns) + 1):
        prefix = key_columns[:depth]
        grouped = frame.group_by(prefix).len()
        distinct = int(grouped.height)
        joint_bits = math.log2(distinct) if distinct > 0 else 0.0

        row: dict[str, Any] = {
            "depth": depth,
            "distinct_prefixes": distinct,
            "mean_pois_per_prefix": float(n_pois / distinct),
            "max_pois_per_prefix": int(cast(Any, grouped["len"].max())),
            "joint_prefix_entropy_bits_upper_bound": joint_bits,
            "branching_factor": (
                float(distinct / previous_distinct)
                if previous_distinct
                else None
            ),
            "conditional_bits_upper_bound": (
                float(joint_bits - previous_bits)
                if previous_bits is not None
                else joint_bits
            ),
        }

        if category_column is not None:
            row["category_purity"] = _modal_purity(
                frame, prefix, category_column
            )
        if naics_column is not None:
            row["naics2_purity"] = _modal_purity(frame, prefix, "_naics2")
        if latitude_column is not None and longitude_column is not None:
            row["mean_geo_radius_km"] = _size_weighted_geo_radius_km(
                frame, prefix, latitude_column, longitude_column
            )

        rows.append(row)
        previous_distinct = distinct
        previous_bits = joint_bits
    return rows


def _pick(frame: pl.DataFrame, *candidates: str) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def covisitation_coherence(
    frame: pl.DataFrame,
    key_columns: list[str],
    stays: pl.DataFrame,
    *,
    poi_key_column: str,
    depths: list[int],
) -> list[dict[str, Any]]:
    pairs = (
        stays.lazy()
        .select(["device_key", poi_key_column])
        .drop_nulls()
        .sort(["device_key"])
        .with_columns(
            pl.col(poi_key_column).shift(-1).over("device_key").alias("_dst")
        )
        .filter(
            pl.col("_dst").is_not_null()
            & (pl.col(poi_key_column) != pl.col("_dst"))
        )
        .group_by([poi_key_column, "_dst"])
        .agg(pl.len().alias("w"))
        .collect()
    )
    rows: list[dict[str, Any]] = []
    if pairs.is_empty():
        return rows

    for depth in depths:
        prefix = key_columns[:depth]
        labelled = (
            frame.select(
                [pl.col(poi_key_column), *[pl.col(c) for c in prefix]]
            )
            .with_columns(
                pl.concat_str(
                    [pl.col(c).cast(pl.Utf8) for c in prefix], separator="|"
                ).alias("_bucket")
            )
            .select([poi_key_column, "_bucket"])
        )

        joined = (
            pairs.join(labelled, on=poi_key_column, how="inner")
            .rename({"_bucket": "_src_bucket"})
            .join(
                labelled.rename({poi_key_column: "_dst"}),
                on="_dst",
                how="inner",
            )
            .rename({"_bucket": "_dst_bucket"})
        )
        if joined.is_empty():
            continue
        total = float(joined["w"].sum())
        same = joined.filter(pl.col("_src_bucket") == pl.col("_dst_bucket"))
        observed = float(same["w"].sum()) / total

        out_mass = (
            joined.group_by("_src_bucket")
            .agg(pl.col("w").sum().alias("out"))
            .sort("_src_bucket")
        )
        in_mass = (
            joined.group_by("_dst_bucket")
            .agg(pl.col("w").sum().alias("inn"))
            .sort("_dst_bucket")
        )
        marg = out_mass.join(
            in_mass,
            left_on="_src_bucket",
            right_on="_dst_bucket",
            how="full",
            maintain_order="left_right",
        ).fill_null(0)
        out_v = marg["out"].to_numpy().astype(float) / total
        in_v = marg["inn"].to_numpy().astype(float) / total
        expected = float((out_v * in_v).sum())

        per_bucket = (
            out_mass.join(
                same.group_by("_src_bucket").agg(
                    pl.col("w").sum().alias("same")
                ),
                on="_src_bucket",
                how="left",
                maintain_order="left",
            )
            .join(
                in_mass.rename({"_dst_bucket": "_src_bucket"}),
                on="_src_bucket",
                how="left",
                maintain_order="left",
            )
            .fill_null(0)
            .filter(pl.col("out") > 0)
        )
        if per_bucket.height:
            obs_share = per_bucket["same"].to_numpy().astype(
                float
            ) / per_bucket["out"].to_numpy().astype(float)
            exp_share = per_bucket["inn"].to_numpy().astype(float) / total
            keep = exp_share > 0
            macro_lift = (
                float(np.mean(obs_share[keep] / exp_share[keep]))
                if keep.any()
                else float("nan")
            )
        else:
            macro_lift = float("nan")

        rows.append(
            {
                "depth": depth,
                "covisit_pairs_scored": int(joined.height),
                "covisit_mass": total,
                "same_bucket_share_observed": observed,
                "same_bucket_share_null": expected,
                "covisit_lift_micro": (
                    observed / expected if expected > 0 else float("nan")
                ),
                "covisit_lift_macro": macro_lift,
                "buckets_scored": int(per_bucket.height),
            }
        )
    return rows


def audit_sid_hierarchy(
    grounding_index_path: Path,
    *,
    stays_path: Path | None = None,
    covisit_split: str = "val",
    label: str | None = None,
) -> dict[str, Any]:
    if stays_path is not None and covisit_split == "train":
        raise ValueError(
            "covisit_split='train' is refused: the encoder under audit is "
            "trained on train co-visitation, so scoring it against the same "
            "split measures memorisation, not coherence"
        )

    frame, key_columns = load_hierarchy_frame(grounding_index_path)
    category_column = _pick(frame, "top_category", "category")
    naics_column = _pick(frame, "naics_code", "naics")
    latitude_column = _pick(frame, "latitude", "lat")
    longitude_column = _pick(frame, "longitude", "lon", "lng")

    if naics_column is not None:
        frame = frame.with_columns(
            pl.col(naics_column).cast(pl.Utf8).str.slice(0, 2).alias("_naics2")
        )

    n_pois = frame.height
    distinct_full = int(frame.group_by(key_columns).len().height)
    report: dict[str, Any] = {
        "label": label,
        "grounding_index": str(grounding_index_path),
        "n_pois": n_pois,
        "n_codebooks": len(key_columns),
        "distinct_full_sids": distinct_full,
        "joint_sid_entropy_bits_upper_bound": (
            math.log2(distinct_full) if distinct_full else 0.0
        ),
        "mean_pois_per_full_sid": float(n_pois / distinct_full),
        "sid_injectivity": float(distinct_full / n_pois),
        "codebook_utilization": codebook_utilization_by_level(
            frame, key_columns
        ),
        "prefix_depths": prefix_depth_hierarchy(
            frame,
            key_columns,
            category_column=category_column,
            naics_column=naics_column,
            latitude_column=latitude_column,
            longitude_column=longitude_column,
        ),
    }

    if stays_path is not None:
        stays = pl.read_parquet(
            stays_path, columns=["device_key", "placekey_id", "split"]
        ).filter(pl.col("split") == covisit_split)
        report["covisitation_coherence"] = {
            "split": covisit_split,
            "stays": int(stays.height),
            "depths": covisitation_coherence(
                frame,
                key_columns,
                stays,
                poi_key_column="placekey_id",
                depths=list(range(1, len(key_columns) + 1)),
            ),
        }

    return report


__all__ = [
    "audit_sid_hierarchy",
    "codebook_utilization_by_level",
    "covisitation_coherence",
    "load_hierarchy_frame",
    "prefix_depth_hierarchy",
]
