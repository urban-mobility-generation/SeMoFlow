from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from semoflow.utils.provenance import (
    ProvenanceClosure,
    load_closure,
    write_closure,
)


@dataclass(frozen=True)
class CitySubsetResult:
    output_dir: Path
    city_ids: tuple[int, ...]
    places: int
    visits: int
    quality_scores: int

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "city_ids": list(self.city_ids),
            "rows": {
                "places": self.places,
                "visits": self.visits,
                "quality_scores": self.quality_scores,
            },
        }


def derive_city_subset(
    source_dir: Path,
    output_dir: Path,
    city_ids: Sequence[int],
) -> CitySubsetResult:
    selected = tuple(sorted(set(int(city_id) for city_id in city_ids)))
    if not selected:
        raise ValueError("at least one city ID is required")
    if len(selected) != len(city_ids):
        raise ValueError("city IDs must not be repeated")

    source_places_path = source_dir / "places_index.parquet"
    source_visits_path = source_dir / "processed_visits.parquet"
    source_quality_path = source_dir / "quality_scores.parquet"
    for path in (source_places_path, source_visits_path, source_quality_path):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    places = pl.read_parquet(source_places_path)
    observed_ids = set(int(value) for value in places["city_id"].unique())
    unknown = set(selected) - observed_ids
    if unknown:
        raise ValueError(f"requested city IDs are absent: {sorted(unknown)}")

    subset_places = places.filter(pl.col("city_id").is_in(selected)).rename(
        {"placekey_id": "placekey_id_source"}
    )
    subset_places = subset_places.with_row_index(
        "placekey_id", offset=1
    ).with_columns(pl.col("placekey_id").cast(pl.Int32))
    id_map = subset_places.select("placekey_id_source", "placekey_id")
    published_places = subset_places.drop("placekey_id_source")
    expected_ids = pl.Series(
        "placekey_id", range(1, published_places.height + 1), dtype=pl.Int32
    )
    if not published_places["placekey_id"].equals(expected_ids):
        raise ValueError("derived placekey_id namespace is not contiguous")
    published_places.write_parquet(output_dir / "places_index.parquet")

    visit_columns = list(
        pl.scan_parquet(source_visits_path).collect_schema().names()
    )
    visits_path = output_dir / "processed_visits.parquet"
    (
        pl.scan_parquet(source_visits_path)
        .filter(pl.col("city_id").is_in(selected))
        .rename({"placekey_id": "placekey_id_source"})
        .join(
            id_map.lazy(),
            on="placekey_id_source",
            how="inner",
            validate="m:1",
            maintain_order="left",
        )
        .select(*visit_columns)
        .sink_parquet(visits_path)
    )
    visit_summary_frame = (
        pl.scan_parquet(visits_path)
        .select(
            pl.len().alias("rows"),
            pl.col("placekey_id").min().alias("min_placekey_id"),
            pl.col("placekey_id").max().alias("max_placekey_id"),
        )
        .collect()
    )
    visit_summary = visit_summary_frame.row(0, named=True)
    visit_city_frame = (
        pl.scan_parquet(visits_path)
        .select("city_id")
        .unique()
        .sort("city_id")
        .collect()
    )
    observed_visit_ids = tuple(
        int(value) for value in visit_city_frame.get_column("city_id")
    )
    if observed_visit_ids != selected:
        raise ValueError(
            "derived visit cities disagree: "
            f"observed={observed_visit_ids} expected={selected}"
        )
    if (
        int(visit_summary["min_placekey_id"]) < 1
        or int(visit_summary["max_placekey_id"]) > published_places.height
    ):
        raise ValueError(
            f"derived visit IDs are out of range: {visit_summary}"
        )

    retained_devices = (
        pl.scan_parquet(visits_path).select("device_key").unique()
    )
    quality_path = output_dir / "quality_scores.parquet"
    (
        pl.scan_parquet(source_quality_path)
        .join(
            retained_devices,
            on="device_key",
            how="semi",
            maintain_order="left",
        )
        .sink_parquet(quality_path)
    )
    quality_summary = pl.scan_parquet(quality_path).select(pl.len()).collect()
    quality_rows = int(quality_summary.item())

    source_closure = load_closure(source_dir)
    if source_closure is None:
        raise FileNotFoundError(
            f"source preprocessing has no provenance: {source_dir}"
        )
    components = []
    covered_city_ids: set[int] = set()
    for component in source_closure.components:
        if component.city_ids is None:
            raise ValueError("source provenance has unrestricted city IDs")
        retained_city_ids = tuple(
            sorted(set(component.city_ids).intersection(selected))
        )
        if not retained_city_ids:
            continue
        covered_city_ids.update(retained_city_ids)
        components.append(
            type(component)(
                source=component.source,
                kind=component.kind,
                city_ids=retained_city_ids,
                splits=component.splits,
                split_seed=component.split_seed,
                time_start=component.time_start,
                time_end=component.time_end,
            )
        )
    if covered_city_ids != set(selected):
        raise ValueError(
            "source provenance does not cover every requested city: "
            f"covered={sorted(covered_city_ids)} expected={list(selected)}"
        )
    write_closure(
        ProvenanceClosure(artifact=str(visits_path), components=components),
        output_dir,
    )
    result = CitySubsetResult(
        output_dir=output_dir,
        city_ids=selected,
        places=published_places.height,
        visits=int(visit_summary["rows"]),
        quality_scores=quality_rows,
    )
    manifest = {
        "command": "semoflow-artifacts subset-cities",
        "source_dir": str(source_dir),
        **result.to_dict(),
        "identity_bridge": "placekey_id remapped through source places_index",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = ["CitySubsetResult", "derive_city_subset"]
