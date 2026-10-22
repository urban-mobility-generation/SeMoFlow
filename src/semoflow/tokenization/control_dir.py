from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl
import torch

from semoflow.tokenization.decoding.grounding_index import (
    SemanticIDGroundingIndex,
)
from semoflow.tokenization.shared_local import (
    semantic_id_lookup_fingerprint,
)
from semoflow.tokenization.token_space import (
    sid_code_columns,
)

CONTROL_MANIFEST = "sid_assignment_control.json"
_COPIED_METADATA = ("config.yaml", "config_resolved.yaml", "provenance.json")


def write_control_tokenizer_dir(
    *,
    source_dir: Path,
    output_dir: Path,
    lookup: pl.DataFrame,
    control: dict[str, Any],
    manifest_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {output_dir}")
    code_cols = list(sid_code_columns(lookup.columns))
    if not code_cols:
        raise ValueError("replacement lookup carries no sem_code_* columns")
    if lookup["placekey_id"].n_unique() != lookup.height:
        raise ValueError("replacement lookup repeats placekey_id")

    source_lookup = pl.read_parquet(source_dir / "placekey_lookup.parquet")
    if set(source_lookup["placekey_id"]) != set(lookup["placekey_id"]):
        raise ValueError(
            "replacement lookup and source lookup cover different POI sets"
        )

    source_index = SemanticIDGroundingIndex.load(
        source_dir / "grounding_index.pkl"
    )
    if source_index.id_column != "placekey_id":
        raise ValueError(
            "source grounding index must be keyed by placekey_id, got "
            f"{source_index.id_column!r}"
        )
    payload = source_index.to_frame().drop(
        sid_code_columns(source_index.to_frame().columns)
    )
    grounding_frame = lookup.select(["placekey_id", *code_cols]).join(
        payload, on="placekey_id", how="left"
    )
    index = SemanticIDGroundingIndex.from_frame(
        grounding_frame,
        key_columns=code_cols,
        id_column="placekey_id",
        payload_columns=source_index.payload_columns,
        order_agnostic=source_index.order_agnostic,
    )

    codebooks = torch.load(
        source_dir / "tokenizer_codebooks.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(codebooks, dict):
        raise ValueError(
            "source tokenizer_codebooks.pt must be a dict artifact"
        )
    source_fingerprint = semantic_id_lookup_fingerprint(
        source_lookup, list(sid_code_columns(source_lookup.columns))
    )
    fingerprint = semantic_id_lookup_fingerprint(lookup, code_cols)
    control = {
        **control,
        "source_dir": str(source_dir),
        "source_lookup_fingerprint": source_fingerprint,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    lookup.write_parquet(output_dir / "placekey_lookup.parquet")
    index.save(output_dir / "grounding_index.pkl")
    torch.save(
        {
            **codebooks,
            "lookup_fingerprint": fingerprint,
            "sid_assignment": control,
        },
        output_dir / "tokenizer_codebooks.pt",
    )
    for name in _COPIED_METADATA:
        if (source_dir / name).exists():
            shutil.copy(source_dir / name, output_dir / name)

    manifest: dict[str, Any] = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "n_pois": lookup.height,
        "n_levels": len(code_cols),
        "n_full_sids": lookup.select(code_cols).n_unique(),
        "source_lookup_fingerprint": source_fingerprint,
        "lookup_fingerprint": fingerprint,
        **control,
        **(manifest_extra or {}),
    }
    (output_dir / CONTROL_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = ["CONTROL_MANIFEST", "write_control_tokenizer_dir"]
