from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

PROVENANCE_FILENAME = "provenance.json"
PROVENANCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ClosureComponent:
    source: str
    kind: str = "behavioral"
    city_ids: tuple[int, ...] | None = None
    splits: tuple[str, ...] | None = None
    split_seed: int | None = None
    time_start: str | None = None
    time_end: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("city_ids", "splits"):
            if d[k] is not None:
                d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClosureComponent:
        return cls(
            source=str(d["source"]),
            kind=str(d.get("kind", "behavioral")),
            city_ids=(
                tuple(int(c) for c in d["city_ids"])
                if d.get("city_ids") is not None
                else None
            ),
            splits=(
                tuple(str(s) for s in d["splits"])
                if d.get("splits") is not None
                else None
            ),
            split_seed=(
                int(d["split_seed"])
                if d.get("split_seed") is not None
                else None
            ),
            time_start=d.get("time_start"),
            time_end=d.get("time_end"),
        )


@dataclass
class ProvenanceClosure:
    artifact: str
    components: list[ClosureComponent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "artifact": self.artifact,
            "components": [c.to_dict() for c in self.components],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProvenanceClosure:
        return cls(
            artifact=str(d.get("artifact", "")),
            components=[
                ClosureComponent.from_dict(c) for c in d.get("components", [])
            ],
        )


def write_closure(closure: ProvenanceClosure, path: Path) -> Path:
    if path.is_dir():
        path = path / PROVENANCE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(closure.to_dict(), indent=2), encoding="utf-8")
    return path


def load_closure(path: Path) -> ProvenanceClosure | None:
    if path.is_dir():
        path = path / PROVENANCE_FILENAME
    if not path.exists():
        return None
    return ProvenanceClosure.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def closure_from_stays(
    stays: pl.DataFrame,
    *,
    artifact: str,
    source: str,
    split_seed: int | None = None,
    timestamp_column: str | None = None,
) -> ProvenanceClosure:
    city_ids: tuple[int, ...] | None = None
    if "city_id" in stays.columns:
        city_ids = tuple(
            sorted(int(c) for c in stays["city_id"].unique().to_list())
        )
    splits: tuple[str, ...] | None = None
    if "split" in stays.columns:
        splits = tuple(
            sorted(str(s) for s in stays["split"].unique().to_list())
        )
    time_start = time_end = None
    ts_col = timestamp_column
    if ts_col is None:
        for candidate in ("stay_start_time", "local_ts", "timestamp"):
            if candidate in stays.columns:
                ts_col = candidate
                break
    if ts_col is not None and ts_col in stays.columns:
        bounds = stays.select(
            pl.col(ts_col).min().alias("lo"),
            pl.col(ts_col).max().alias("hi"),
        ).row(0)
        if bounds[0] is not None:
            time_start = str(bounds[0])
            time_end = str(bounds[1])
    return ProvenanceClosure(
        artifact=artifact,
        components=[
            ClosureComponent(
                source=source,
                kind="behavioral",
                city_ids=city_ids,
                splits=splits,
                split_seed=split_seed,
                time_start=time_start,
                time_end=time_end,
            )
        ],
    )


__all__ = [
    "PROVENANCE_FILENAME",
    "ClosureComponent",
    "ProvenanceClosure",
    "closure_from_stays",
    "load_closure",
    "write_closure",
]
