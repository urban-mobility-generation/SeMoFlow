from __future__ import annotations

import dataclasses
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from semoflow.tokenization.token_space import (
    sid_code_columns,
)

CodeKey = tuple[int, ...]


@dataclass(frozen=True)
class GroundingStats:
    n_pois: int
    n_keys: int
    key_columns: list[str]
    order_agnostic: bool

    duplicate_poi_rate: float
    ambiguous_key_rate: float
    singleton_key_rate: float
    mean_candidates_per_key: float
    max_candidates_per_key: int

    category_purity_mean: float | None
    candidate_count_histogram: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _row_key(
    row: Mapping[str, Any],
    key_columns: Sequence[str],
    *,
    order_agnostic: bool,
) -> CodeKey:
    if order_agnostic:
        return tuple(sorted(int(row[col]) for col in key_columns))
    return tuple(int(row[col]) for col in key_columns)


def _purity(values: Iterable[Any]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    counts = Counter(vals)
    return counts.most_common(1)[0][1] / len(vals)


class SemanticIDGroundingIndex:
    def __init__(
        self,
        *,
        key_columns: Sequence[str],
        id_column: str,
        payload_columns: Sequence[str],
        order_agnostic: bool,
        mapping: Mapping[CodeKey, Sequence[dict[str, Any]]],
    ) -> None:
        self.key_columns = list(key_columns)
        self.id_column = id_column
        self.payload_columns = list(payload_columns)
        self.order_agnostic = bool(order_agnostic)
        self._mapping: dict[CodeKey, list[dict[str, Any]]] = {
            tuple(key): [dict(item) for item in payloads]
            for key, payloads in mapping.items()
        }

    @classmethod
    def from_frame(
        cls,
        frame: pl.DataFrame,
        *,
        key_columns: Sequence[str] | None = None,
        id_column: str = "placekey",
        payload_columns: Sequence[str] = (
            "placekey",
            "placekey_id",
            "top_category",
            "top_cat",
        ),
        order_agnostic: bool = False,
    ) -> SemanticIDGroundingIndex:
        keys = list(key_columns or sid_code_columns(frame.columns))
        if not keys:
            raise ValueError("no semantic code columns found")
        missing = [c for c in [*keys, id_column] if c not in frame.columns]
        if missing:
            raise ValueError(f"grounding frame missing columns: {missing}")
        payload_cols = [
            c
            for c in dict.fromkeys([id_column, *payload_columns])
            if c in frame.columns
        ]
        mapping: dict[CodeKey, list[dict[str, Any]]] = defaultdict(list)
        seen_by_key: dict[CodeKey, set[Any]] = defaultdict(set)
        for row in frame.select([*keys, *payload_cols]).iter_rows(named=True):
            key = _row_key(row, keys, order_agnostic=order_agnostic)
            poi_id = row[id_column]
            if poi_id in seen_by_key[key]:
                continue
            seen_by_key[key].add(poi_id)
            mapping[key].append({col: row[col] for col in payload_cols})
        return cls(
            key_columns=keys,
            id_column=id_column,
            payload_columns=payload_cols,
            order_agnostic=order_agnostic,
            mapping=mapping,
        )

    @property
    def n_keys(self) -> int:
        return len(self._mapping)

    @property
    def n_pois(self) -> int:
        return sum(len(payloads) for payloads in self._mapping.values())

    def iter_payloads(self) -> Iterable[dict[str, Any]]:
        for payloads in self._mapping.values():
            yield from payloads

    def iter_payload_groups(self) -> Iterable[Sequence[Mapping[str, Any]]]:
        yield from self._mapping.values()

    def iter_key_payload_groups(
        self,
    ) -> Iterable[tuple[CodeKey, Sequence[Mapping[str, Any]]]]:
        yield from self._mapping.items()

    def build_category_string_vocab(
        self,
        column: str = "top_category",
        *,
        fallback_columns: Sequence[str] = ("top_cat",),
    ) -> dict[str, int]:
        strings: set[str] = set()
        candidates = [column, *fallback_columns]
        for payload in self.iter_payloads():
            for candidate in candidates:
                val = payload.get(candidate)
                if isinstance(val, str) and val:
                    strings.add(val)
                    break
        return {s: i + 1 for i, s in enumerate(sorted(strings))}

    def build_naics_string_vocab(
        self,
        *,
        column: str = "naics_code",
    ) -> dict[str, int]:
        strings: set[str] = set()
        for payload in self.iter_payloads():
            val = payload.get(column)
            if isinstance(val, str) and val:
                strings.add(val)
        return {s: i + 1 for i, s in enumerate(sorted(strings))}

    def candidate_payloads(
        self, code_values: Sequence[int] | Mapping[str, int]
    ) -> list[dict[str, Any]]:
        key = self._lookup_key(code_values)
        return [dict(item) for item in self._mapping.get(key, [])]

    def packed_lookup_table(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        cached = getattr(self, "_packed_lookup_cache", None)
        if cached is not None:
            return cached
        if not self._mapping:
            empty = (
                np.empty(0, dtype=np.uint64),
                np.empty(0, dtype=np.int32),
                [],
            )
            self._packed_lookup_cache = empty
            self._packed_lookup_groups_cache = []
            return empty

        key_rows = np.asarray(list(self._mapping), dtype=np.int64)
        if key_rows.ndim != 2 or key_rows.shape[1] != len(self.key_columns):
            raise ValueError("grounding index keys have inconsistent widths")
        if np.any((key_rows < 0) | (key_rows > 255)):
            raise ValueError(
                "bulk grounding lookup supports code values in [0, 255]"
            )
        packed = np.zeros(key_rows.shape[0], dtype=np.uint64)
        for column in range(key_rows.shape[1]):
            packed |= key_rows[:, column].astype(np.uint64) << (8 * column)
        order = np.argsort(packed, kind="mergesort")
        payload_rows = list(self._mapping.values())
        table = (
            packed[order],
            np.asarray(
                [len(payload_rows[int(index)]) for index in order],
                dtype=np.int32,
            ),
            [payload_rows[int(index)][0] for index in order],
        )
        self._packed_lookup_cache = table
        self._packed_lookup_groups_cache = [
            payload_rows[int(index)] for index in order
        ]
        return table

    def packed_lookup_groups(self) -> list[Sequence[Mapping[str, Any]]]:
        cached = getattr(self, "_packed_lookup_groups_cache", None)
        if cached is None:
            self._packed_lookup_cache = None
            self.packed_lookup_table()
            cached = getattr(self, "_packed_lookup_groups_cache", None)
        if cached is None:
            raise RuntimeError("failed to build packed candidate groups")
        return cached

    def _lookup_key(
        self, code_values: Sequence[int] | Mapping[str, int]
    ) -> CodeKey:
        if isinstance(code_values, Mapping):
            return _row_key(
                cast(Mapping[str, Any], code_values),
                self.key_columns,
                order_agnostic=self.order_agnostic,
            )
        if self.order_agnostic:
            if len(code_values) != len(self.key_columns):
                raise ValueError(
                    "code_values length does not match grounding key columns"
                )
            return tuple(sorted(int(value) for value in code_values))
        return tuple(int(v) for v in code_values)

    def candidate_count(
        self, code_values: Sequence[int] | Mapping[str, int]
    ) -> int:
        key = self._lookup_key(code_values)
        return len(self._mapping.get(key, ()))

    def stats(self) -> GroundingStats:
        candidate_counts = [
            len(payloads) for payloads in self._mapping.values()
        ]
        hist = Counter(candidate_counts)
        ambiguous = [n for n in candidate_counts if n > 1]

        category_col = (
            "top_category"
            if "top_category" in self.payload_columns
            else "top_cat"
            if "top_cat" in self.payload_columns
            else None
        )
        category_purities: list[float] = []
        for payloads in self._mapping.values():
            if category_col is not None:
                cat_purity = _purity(p.get(category_col) for p in payloads)
                if cat_purity is not None:
                    category_purities.append(cat_purity)

        n_pois = self.n_pois
        n_keys = self.n_keys
        return GroundingStats(
            n_pois=n_pois,
            n_keys=n_keys,
            key_columns=list(self.key_columns),
            order_agnostic=self.order_agnostic,
            duplicate_poi_rate=1.0 - (n_keys / n_pois) if n_pois else 0.0,
            ambiguous_key_rate=len(ambiguous) / n_keys if n_keys else 0.0,
            singleton_key_rate=(
                sum(1 for n in candidate_counts if n == 1) / n_keys
                if n_keys
                else 0.0
            ),
            mean_candidates_per_key=(
                sum(candidate_counts) / n_keys if n_keys else 0.0
            ),
            max_candidates_per_key=max(candidate_counts)
            if candidate_counts
            else 0,
            category_purity_mean=(
                sum(category_purities) / len(category_purities)
                if category_purities
                else None
            ),
            candidate_count_histogram={
                int(k): int(v) for k, v in hist.items()
            },
        )

    def to_frame(self) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for key, payloads in self._mapping.items():
            for payload in payloads:
                row = dict(payload)
                for col, code in zip(self.key_columns, key, strict=True):
                    row[col] = code
                rows.append(row)
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> SemanticIDGroundingIndex:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"expected {cls.__name__}, got {type(obj).__name__}"
            )
        return obj


__all__ = ["GroundingStats", "SemanticIDGroundingIndex"]
