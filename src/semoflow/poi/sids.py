from __future__ import annotations

import functools
import pickle
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

SID_CODE_PREFIX = "sem_code_"


def sid_code_names(n_codebooks: int) -> tuple[str, ...]:
    return tuple(f"{SID_CODE_PREFIX}{i}" for i in range(n_codebooks))


def sid_code_columns(columns: Iterable[str]) -> tuple[str, ...]:
    names = [c for c in columns if c.startswith(SID_CODE_PREFIX)]
    return tuple(sorted(names, key=lambda c: int(c.rsplit("_", 1)[-1])))


def packed_code_keys(
    codes: torch.Tensor, *, codebook_size: int
) -> torch.LongTensor:
    width = codes.shape[-1]
    powers = torch.tensor(
        [codebook_size**power for power in range(width - 1, -1, -1)],
        dtype=torch.long,
        device=codes.device,
    )
    return (codes.long() * powers).sum(dim=-1)


def find_sorted(
    sorted_keys: torch.Tensor, keys: torch.Tensor
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    positions = torch.searchsorted(sorted_keys, keys)
    safe = positions.clamp_max(max(sorted_keys.numel() - 1, 0))
    found = (positions < sorted_keys.numel()) & (sorted_keys[safe] == keys)
    return safe, found


def load_codebooks(path: str | Path) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return payload["codebooks"].detach().float().contiguous()


def codes_to_embeddings(
    codes: torch.Tensor, codebooks: torch.Tensor
) -> torch.Tensor:
    codes = codes.long()
    codebooks = codebooks.to(codes.device)
    levels = torch.arange(codebooks.shape[0], device=codes.device)
    return codebooks[levels, codes].sum(dim=-2)


def pack_code_rows(rows: np.ndarray) -> np.ndarray:
    packed = np.zeros(rows.shape[0], dtype=np.uint64)
    for column in range(rows.shape[1]):
        packed |= rows[:, column].astype(np.uint64) << (8 * column)
    return packed


class SemanticIDGroundingIndex:
    def __init__(
        self,
        *,
        key_columns: Sequence[str],
        payload_columns: Sequence[str],
        groups: Mapping[tuple[int, ...], Sequence[dict[str, Any]]],
    ) -> None:
        self.key_columns = list(key_columns)
        self.payload_columns = list(payload_columns)
        self.groups = {key: list(items) for key, items in groups.items()}

    @classmethod
    def from_frame(
        cls, frame: pl.DataFrame, *, payload_columns: Sequence[str]
    ) -> SemanticIDGroundingIndex:
        keys = list(sid_code_columns(frame.columns))
        payload_cols = list(dict.fromkeys(["placekey_id", *payload_columns]))
        groups: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
        seen: dict[tuple[int, ...], set[Any]] = defaultdict(set)
        for row in frame.select([*keys, *payload_cols]).iter_rows(named=True):
            key = tuple(int(row[col]) for col in keys)
            if row["placekey_id"] in seen[key]:
                continue
            seen[key].add(row["placekey_id"])
            groups[key].append({col: row[col] for col in payload_cols})
        return cls(
            key_columns=keys, payload_columns=payload_cols, groups=groups
        )

    @functools.cached_property
    def _packed_lookup(
        self,
    ) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
        key_rows = np.asarray(list(self.groups), dtype=np.int64)
        packed = pack_code_rows(key_rows)
        order = np.argsort(packed, kind="mergesort")
        payload_rows = list(self.groups.values())
        return packed[order], [payload_rows[int(index)] for index in order]

    def packed_lookup_table(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        keys, groups = self._packed_lookup
        return keys, [group[0] for group in groups]

    def packed_lookup_groups(self) -> list[list[dict[str, Any]]]:
        return self._packed_lookup[1]

    def to_frame(self) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for key, payloads in self.groups.items():
            for payload in payloads:
                row = dict(payload)
                for col, code in zip(self.key_columns, key, strict=True):
                    row[col] = code
                rows.append(row)
        return pl.DataFrame(rows)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "key_columns": self.key_columns,
            "payload_columns": self.payload_columns,
            "groups": self.groups,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path) -> SemanticIDGroundingIndex:
        with open(path, "rb") as f:
            return cls(**pickle.load(f))


__all__ = [
    "SemanticIDGroundingIndex",
    "codes_to_embeddings",
    "find_sorted",
    "load_codebooks",
    "pack_code_rows",
    "packed_code_keys",
    "sid_code_columns",
    "sid_code_names",
]
