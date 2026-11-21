from __future__ import annotations

import functools
import pickle
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from semoflow.tokenization.token_space import sid_code_columns

CodeKey = tuple[int, ...]


def _row_key(
    row: Mapping[str, Any],
    key_columns: Sequence[str],
    *,
    order_agnostic: bool,
) -> CodeKey:
    if order_agnostic:
        return tuple(sorted(int(row[col]) for col in key_columns))
    return tuple(int(row[col]) for col in key_columns)


def pack_code_rows(rows: np.ndarray) -> np.ndarray:
    """Pack each row of byte-sized codes into one uint64 key."""
    packed = np.zeros(rows.shape[0], dtype=np.uint64)
    for column in range(rows.shape[1]):
        packed |= rows[:, column].astype(np.uint64) << (8 * column)
    return packed


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

    def iter_key_payload_groups(
        self,
    ) -> Iterable[tuple[CodeKey, Sequence[Mapping[str, Any]]]]:
        yield from self._mapping.items()

    @functools.cached_property
    def _packed_lookup(
        self,
    ) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
        """Sorted packed keys and each key's candidate payloads."""
        key_rows = np.asarray(list(self._mapping), dtype=np.int64)
        if np.any((key_rows < 0) | (key_rows > 255)):
            raise ValueError(
                "bulk grounding lookup supports code values in [0, 255]"
            )
        packed = pack_code_rows(key_rows)
        order = np.argsort(packed, kind="mergesort")
        payload_rows = list(self._mapping.values())
        return packed[order], [payload_rows[int(index)] for index in order]

    def packed_lookup_table(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Sorted packed keys and each key's first candidate payload."""
        keys, groups = self._packed_lookup
        return keys, [group[0] for group in groups]

    def packed_lookup_groups(self) -> list[list[dict[str, Any]]]:
        return self._packed_lookup[1]

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
        return obj


__all__ = ["SemanticIDGroundingIndex", "pack_code_rows"]
