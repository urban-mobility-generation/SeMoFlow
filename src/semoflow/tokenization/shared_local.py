from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import polars as pl

__all__ = [
    "semantic_id_lookup_fingerprint",
    "shared_codebook_fingerprint",
]

_ID_COLUMN = "placekey_id"


def shared_codebook_fingerprint(codebooks: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for index, codebook in enumerate(codebooks):
        tensor = codebook.detach().to("cpu").contiguous()
        digest.update(str(index).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def semantic_id_lookup_fingerprint(
    lookup: pl.DataFrame,
    code_columns: Sequence[str],
) -> str:
    _require_columns(lookup, [*code_columns], label="lookup")
    key_column = "placekey" if "placekey" in lookup.columns else _ID_COLUMN
    _require_columns(lookup, [key_column], label="lookup")
    key = lookup[key_column]
    if key.null_count():
        raise ValueError(f"lookup key {key_column!r} contains nulls")
    if key.n_unique() != lookup.height:
        raise ValueError(f"lookup key {key_column!r} is not unique")
    for column in code_columns:
        series = lookup[column]
        if not series.dtype.is_integer():
            raise ValueError(
                f"lookup column {column!r} must hold integer codes, got "
                f"{series.dtype}"
            )
        if series.null_count():
            raise ValueError(f"lookup column {column!r} contains nulls")

    canonical = lookup.select([key_column, *code_columns]).sort(key_column)
    digest = hashlib.sha256()
    digest.update(key_column.encode("utf-8"))
    digest.update(json.dumps(list(code_columns)).encode("utf-8"))
    digest.update(str(canonical.height).encode("utf-8"))
    for row in canonical.iter_rows():
        digest.update(json.dumps(list(row), separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _require_columns(
    frame: pl.DataFrame, columns: Sequence[str], *, label: str
) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
