from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl

from semoflow.cross_city.union_catalog import (
    read_union_target,
    refuse_existing,
    write_sidecar_manifest,
)
from semoflow.utils.data import hash_fraction

BUDGET_SELECTION_SEED = 20260921
INNER_ROLE_SEED = 20260922
HOLDOUT_SELECTION_SEED = 20260923

INNER_TRAIN_FRACTION = 0.8
INNER_VALIDATION_FRACTION = 0.1

DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.50, 1.00)

_INNER_ROLE_TO_SPLIT = {"train": "train", "validation": "val"}
_CODE_COLUMNS = tuple(f"sem_code_{level}" for level in range(6))

RungSegmentLayout = Literal["train", "eval"]


def rung_label(fraction: float) -> str:
    return f"pct_{fraction * 100:g}".replace(".", "p")


def _eligible_devices(visits: Path, segments: Path) -> pl.DataFrame:
    segment_devices = (
        pl.scan_parquet(segments)
        .filter(pl.col("_split") == "train")
        .group_by("device_key")
        .agg(pl.len().alias("segment_count"))
        .collect()
    )
    visit_devices = (
        pl.scan_parquet(visits)
        .filter(pl.col("split") == "train")
        .group_by(["caid", "device_key"])
        .agg(pl.len().alias("event_count"))
        .collect()
    )
    eligible = visit_devices.join(
        segment_devices, on="device_key", how="inner"
    )
    if eligible.is_empty():
        raise ValueError("no eligible target-train devices were found")
    if eligible.get_column("caid").n_unique() != eligible.height:
        raise ValueError("caid must identify a target-train device uniquely")
    return (
        eligible.with_columns(
            pl.col("caid")
            .map_elements(
                lambda value: hash_fraction(value, BUDGET_SELECTION_SEED),
                return_dtype=pl.Float64,
            )
            .alias("budget_rank"),
            pl.col("caid")
            .map_elements(
                lambda value: hash_fraction(value, INNER_ROLE_SEED),
                return_dtype=pl.Float64,
            )
            .alias("_inner_rank"),
        )
        .with_columns(
            pl.when(pl.col("_inner_rank") < INNER_TRAIN_FRACTION)
            .then(pl.lit("train"))
            .when(
                pl.col("_inner_rank")
                < INNER_TRAIN_FRACTION + INNER_VALIDATION_FRACTION
            )
            .then(pl.lit("validation"))
            .otherwise(pl.lit("test"))
            .alias("inner_role")
        )
        .drop("_inner_rank")
    )


def _rung_manifest(
    rung: pl.DataFrame, *, city: str, fraction: float, eligible: int
) -> dict[str, Any]:
    by_role = {
        str(row["inner_role"]): {
            "devices": int(row["devices"]),
            "events": int(row["events"]),
            "segments": int(row["segments"]),
        }
        for row in rung.group_by("inner_role")
        .agg(
            pl.len().alias("devices"),
            pl.col("event_count").sum().alias("events"),
            pl.col("segment_count").sum().alias("segments"),
        )
        .to_dicts()
    }
    return {
        "artifact_kind": "nested_budget_panel",
        "city": city,
        "split": "train",
        "behavioral_data": True,
        "target_behavior_used": True,
        "selection_unit": "complete_device",
        "budget_fraction": fraction,
        "budget_selection_seed": BUDGET_SELECTION_SEED,
        "inner_role_seed": INNER_ROLE_SEED,
        "inner_train_fraction": INNER_TRAIN_FRACTION,
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "eligible_devices": eligible,
        "devices": rung.height,
        "events": int(rung.get_column("event_count").sum()),
        "segments": int(rung.get_column("segment_count").sum()),
        "by_inner_role": by_role,
    }


def build_budget_ladder(
    *,
    city: str,
    visits: Path,
    segments: Path,
    output_dir: Path,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    force: bool = False,
) -> dict[str, Any]:
    ordered = sorted(set(fractions))
    if not ordered or any(not 0.0 < value <= 1.0 for value in ordered):
        raise ValueError("every budget fraction must lie in (0, 1]")

    rung_paths = [
        (
            output_dir / rung_label(fraction) / "devices.parquet",
            output_dir / rung_label(fraction) / "manifest.json",
        )
        for fraction in ordered
    ]
    ladder_path = output_dir / "ladder.json"
    refuse_existing(
        *(path for pair in rung_paths for path in pair),
        ladder_path,
        force=force,
    )

    eligible = _eligible_devices(visits, segments)
    rungs: list[dict[str, Any]] = []
    previous: set[str] = set()
    for fraction, (devices_path, manifest_path) in zip(
        ordered, rung_paths, strict=True
    ):
        rung = eligible.filter(pl.col("budget_rank") < fraction).drop(
            "budget_rank"
        )
        if rung.is_empty():
            raise ValueError(f"fraction {fraction} selected no devices")
        selected = set(rung.get_column("caid").to_list())
        if not previous <= selected:
            raise RuntimeError(
                f"rung {fraction} is not a superset of the previous rung"
            )
        previous = selected

        devices_path.parent.mkdir(parents=True, exist_ok=True)
        rung.sort("caid").write_parquet(devices_path)
        manifest = _rung_manifest(
            rung, city=city, fraction=fraction, eligible=eligible.height
        )
        manifest["devices_path"] = str(devices_path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rungs.append(manifest)

    ladder = {
        "artifact_kind": "nested_budget_ladder",
        "city": city,
        "visits": str(visits),
        "segments": str(segments),
        "eligible_devices": eligible.height,
        "eligible_events": int(eligible.get_column("event_count").sum()),
        "eligible_segments": int(eligible.get_column("segment_count").sum()),
        "nested_verified": True,
        "rungs": rungs,
    }
    ladder_path.write_text(
        json.dumps(ladder, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"ladder": str(ladder_path), "rungs": len(rungs)}


def _panel_inner_splits(panel_path: Path) -> pl.DataFrame:
    panel = pl.read_parquet(panel_path)
    for column in ("device_key", "inner_role"):
        if column not in panel.columns:
            raise ValueError(f"panel is missing {column!r}")
    selected = panel.filter(
        pl.col("inner_role").is_in(list(_INNER_ROLE_TO_SPLIT))
    ).select(
        "device_key",
        pl.col("inner_role")
        .replace_strict(_INNER_ROLE_TO_SPLIT)
        .alias("_inner_split"),
    )
    if selected.is_empty():
        raise ValueError("panel has no train or validation devices")
    return selected


def build_rung_visits(
    *,
    visits: Path,
    panel: Path,
    output: Path,
    inner_role: str = "train",
    force: bool = False,
) -> dict[str, Any]:
    refuse_existing(output, force=force)
    devices = pl.read_parquet(panel)
    if "inner_role" not in devices.columns or "caid" not in devices.columns:
        raise ValueError("panel must contain caid and inner_role columns")
    selected = devices.filter(pl.col("inner_role") == inner_role)
    if selected.is_empty():
        raise ValueError(
            f"panel has no devices with inner_role={inner_role!r}"
        )

    rows = (
        pl.scan_parquet(visits)
        .filter(pl.col("split") == "train")
        .filter(pl.col("caid").is_in(selected.get_column("caid").implode()))
        .collect()
    )
    if rows.is_empty():
        raise ValueError("the rung selected no training visits")

    output.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(output)
    manifest = {
        "artifact_kind": "budget_rung_visits",
        "source_visits": str(visits),
        "panel": str(panel),
        "inner_role": inner_role,
        "devices": int(rows.get_column("caid").n_unique()),
        "panel_devices": selected.height,
        "events": rows.height,
        "output": str(output),
    }
    write_sidecar_manifest(output, manifest)
    return manifest


def build_rung_augmented_visits(
    *,
    visits: Path,
    panel: Path,
    places_index: Path,
    union_lookup: Path,
    union_manifest: Path,
    output: Path,
    force: bool = False,
) -> dict[str, Any]:
    refuse_existing(output, force=force)
    offset, target_city_id = read_union_target(union_manifest)
    inner_splits = _panel_inner_splits(panel)
    bridge = pl.read_parquet(places_index, columns=["placekey_id", "placekey"])
    lookup = pl.read_parquet(
        union_lookup, columns=["placekey_id", "placekey", *_CODE_COLUMNS]
    ).rename({"placekey_id": "_union_placekey_id"})

    rows = (
        pl.scan_parquet(visits)
        .filter(pl.col("split") == "train")
        .join(inner_splits.lazy(), on="device_key", how="inner")
        .join(bridge.lazy(), on="placekey_id", how="inner")
        .join(lookup.lazy(), on="placekey", how="inner")
        .drop("placekey_id", "placekey")
        .rename({"_union_placekey_id": "placekey_id"})
        .with_columns(
            pl.lit(target_city_id, dtype=pl.Int32).alias("city_id"),
            pl.col("_inner_split").alias("split"),
        )
        .drop("_inner_split")
        .collect()
    )
    if rows.is_empty():
        raise ValueError("the rung selected no visits")
    ids = rows.get_column("placekey_id")
    if cast(int, ids.min()) <= offset:
        raise ValueError(
            "a rewritten placekey_id landed inside the source catalog's range"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(output)
    manifest = {
        "artifact_kind": "budget_rung_augmented_visits",
        "source_visits": str(visits),
        "panel": str(panel),
        "places_index": str(places_index),
        "union_lookup": str(union_lookup),
        "placekey_id_offset": offset,
        "target_city_id": target_city_id,
        "devices": int(rows.get_column("device_key").n_unique()),
        "events": rows.height,
        "events_by_split": {
            str(row["split"]): int(row["len"])
            for row in rows.group_by("split").len().iter_rows(named=True)
        },
        "distinct_placekey_ids": int(ids.n_unique()),
        "output": str(output),
    }
    write_sidecar_manifest(output, manifest)
    return manifest


def build_holdout_reference_panel(
    *,
    segments: Path,
    output: Path,
    split: str = "val",
    size: int = 20_000,
    min_length: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    refuse_existing(output, force=force)
    if split == "train":
        raise ValueError(
            "the held-out reference must come from an outer held-out split, "
            "not the train split the budget ladder selects within"
        )
    pool = (
        pl.read_parquet(segments)
        .filter(
            (pl.col("_split") == split)
            & (pl.col("placekey_id").list.len() >= min_length)
        )
        .select("_seg_id", "device_key")
    )
    if pool.height < size:
        raise ValueError(
            f"held-out pool has {pool.height} usable segments, fewer than the "
            f"requested {size}"
        )
    ranked = pool.with_columns(
        pl.struct("device_key", "_seg_id")
        .map_elements(
            lambda row: hash_fraction(
                f"{row['device_key']}:{row['_seg_id']}",
                HOLDOUT_SELECTION_SEED,
            ),
            return_dtype=pl.Float64,
        )
        .alias("_rank")
    ).sort("_rank", "_seg_id")
    panel = ranked.head(size).drop("_rank").sort("_seg_id")

    output.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(output)
    manifest = {
        "artifact_kind": "holdout_reference_panel",
        "source_segments": str(segments),
        "outer_split": split,
        "selection_seed": HOLDOUT_SELECTION_SEED,
        "min_length": min_length,
        "pool_segments": pool.height,
        "panel_segments": panel.height,
        "panel_devices": int(panel.get_column("device_key").n_unique()),
        "output": str(output),
    }
    write_sidecar_manifest(output, manifest)
    return manifest


def build_rung_segments(
    *,
    segments: Path,
    panel: Path,
    union_lookup: Path,
    union_manifest: Path,
    holdout_panel: Path,
    output: Path,
    layout: RungSegmentLayout = "train",
    force: bool = False,
) -> dict[str, Any]:
    if layout not in ("train", "eval"):
        raise ValueError("layout must be 'train' or 'eval'")
    refuse_existing(output, force=force)
    offset, target_city_id = read_union_target(union_manifest)
    inner_splits = _panel_inner_splits(panel)

    cache = pl.read_parquet(segments)
    missing = sorted(
        {"_seg_id", "placekey_id", "device_key", "_timestamps", "_split"}
        - set(cache.columns)
    )
    if missing:
        raise ValueError(f"segment cache is missing columns {missing}")

    outer_train = cache.filter(pl.col("_split") == "train")
    known = set(outer_train.get_column("device_key").unique().to_list())
    stray = sorted(
        set(inner_splits.get_column("device_key").to_list()) - known
    )
    if stray:
        raise ValueError(
            f"{len(stray)} panel devices are absent from the cache's outer "
            f"train split, e.g. {stray[:3]}"
        )
    rows = outer_train.join(inner_splits, on="device_key", how="inner")
    if rows.is_empty():
        raise ValueError("the rung selected no segments")

    holdout_ids = pl.read_parquet(holdout_panel, columns=["_seg_id"])
    holdout = cache.join(holdout_ids, on="_seg_id", how="inner")
    if holdout.height != holdout_ids.height:
        raise ValueError(
            f"held-out panel names {holdout_ids.height} segments but the cache "
            f"supplies {holdout.height}"
        )
    if (holdout.get_column("_split") == "train").any():
        raise ValueError(
            "a held-out reference segment comes from the outer train split the "
            "budget ladder selects within"
        )
    shared_devices = set(
        holdout.get_column("device_key").unique().to_list()
    ) & set(rows.get_column("device_key").unique().to_list())
    if shared_devices:
        raise ValueError(
            f"{len(shared_devices)} devices appear in both the rung and the "
            "held-out reference"
        )

    if layout == "eval":
        rows = rows.with_columns(
            pl.when(pl.col("_inner_split") == "val")
            .then(pl.lit("test"))
            .otherwise(pl.col("_inner_split"))
            .alias("_inner_split")
        )
        holdout_split = "val"
    else:
        holdout_split = "test"
    rows = pl.concat(
        [
            rows,
            holdout.with_columns(pl.lit(holdout_split).alias("_inner_split")),
        ],
        how="vertical",
    ).with_columns(
        pl.col("placekey_id").list.eval(pl.element() + offset),
        pl.lit(target_city_id, dtype=pl.Int32).alias("_city_id"),
        pl.col("_inner_split").alias("_split"),
    )
    rows = rows.drop("_inner_split")

    legal = set(
        pl.read_parquet(union_lookup, columns=["placekey_id"])
        .get_column("placekey_id")
        .to_list()
    )
    used = set(
        rows.select(pl.col("placekey_id").explode())
        .get_column("placekey_id")
        .unique()
        .to_list()
    )
    orphans = sorted(used - legal)
    if orphans:
        raise ValueError(
            f"{len(orphans)} shifted placekey_ids are absent from the union "
            f"lookup, e.g. {orphans[:3]}"
        )
    if min(used) <= offset:
        raise ValueError(
            "a shifted placekey_id landed inside the source catalog's range"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(output)
    manifest = {
        "artifact_kind": "budget_rung_weekly_segments",
        "layout": layout,
        "source_segments": str(segments),
        "panel": str(panel),
        "union_lookup": str(union_lookup),
        "placekey_id_offset": offset,
        "target_city_id": target_city_id,
        "devices": int(rows.get_column("device_key").n_unique()),
        "segments": rows.height,
        "segments_by_split": {
            str(row["_split"]): int(row["len"])
            for row in rows.group_by("_split").len().iter_rows(named=True)
        },
        "distinct_placekey_ids": len(used),
        "output": str(output),
    }
    write_sidecar_manifest(output, manifest)
    return manifest


__all__ = [
    "BUDGET_SELECTION_SEED",
    "DEFAULT_FRACTIONS",
    "HOLDOUT_SELECTION_SEED",
    "INNER_ROLE_SEED",
    "RungSegmentLayout",
    "build_budget_ladder",
    "build_holdout_reference_panel",
    "build_rung_augmented_visits",
    "build_rung_segments",
    "build_rung_visits",
    "rung_label",
]
