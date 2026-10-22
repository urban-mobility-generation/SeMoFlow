from __future__ import annotations

import dataclasses
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import torch

from semoflow.tokenization.decoding.codebooks import (
    extract_codebook_bundle,
)
from semoflow.tokenization.shared_local import (
    read_shared_local_provenance,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComposedTokenizerDirResult:
    output_dir: str
    city_id: int
    n_pois: int
    n_levels: int
    shared_levels: int
    local_levels: int
    shared_codebook_fingerprint: str
    codebook_shape: tuple[int, ...]
    augmented_visit_rows: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def assemble_composed_tokenizer_dir(
    *,
    composed_lookup: Path,
    city_id: int,
    local_dir: Path,
    local_places: Path,
    shared_codebooks: Path,
    output_dir: Path,
) -> ComposedTokenizerDirResult:
    provenance = read_shared_local_provenance(
        composed_lookup.with_suffix(
            composed_lookup.suffix + ".provenance.json"
        )
    )
    spec = provenance.spec
    logger.info(
        "composed vocabulary %s fingerprint %s (%d shared + %d local levels)",
        spec["shared_sid_version"],
        spec["shared_codebook_fingerprint"][:16],
        spec["shared_levels"],
        spec["local_levels"],
    )

    lookup = pl.read_parquet(composed_lookup).filter(
        pl.col("city_id") == city_id
    )
    if lookup.is_empty():
        raise ValueError(f"composed lookup has no rows for city {city_id}")
    if "placekey" not in lookup.columns:
        raise ValueError(
            "composed lookup must include placekey so the local visit "
            "namespace can be verified"
        )
    code_cols = sorted(
        (c for c in lookup.columns if c.startswith("sem_code_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    if not code_cols:
        raise ValueError("composed lookup has no semantic-ID code columns")
    logger.info(
        "city %d: %d POIs, %d levels", city_id, lookup.height, len(code_cols)
    )
    if lookup["placekey"].null_count():
        raise ValueError("composed lookup contains null placekey values")
    if lookup["placekey"].n_unique() != lookup.height:
        raise ValueError("composed lookup repeats placekey")

    local_places_frame = pl.read_parquet(
        local_places, columns=["placekey_id", "placekey"]
    ).with_columns(pl.col("placekey_id").cast(pl.Int64))
    local_nulls = local_places_frame.null_count()
    if local_nulls["placekey_id"][0] > 0 or local_nulls["placekey"][0] > 0:
        raise ValueError("local places index contains null identity values")
    if (
        local_places_frame["placekey_id"].n_unique()
        != local_places_frame.height
    ):
        raise ValueError("local places index repeats placekey_id")
    if local_places_frame["placekey"].n_unique() != local_places_frame.height:
        raise ValueError("local places index repeats placekey")

    lookup_namespace = (
        lookup.select("placekey_id", "placekey")
        .with_columns(pl.col("placekey_id").cast(pl.Int64))
        .join(local_places_frame, on="placekey", how="left", suffix="_local")
    )
    if lookup_namespace["placekey_id_local"].null_count():
        raise ValueError(
            "composed lookup contains POIs absent from the local places "
            "namespace"
        )
    if (
        lookup_namespace["placekey_id"]
        != lookup_namespace["placekey_id_local"]
    ).any():
        raise ValueError(
            "composed lookup placekey_id values do not match the local "
            "places namespace"
        )

    src = local_dir / "augmented_processed_visits.parquet"
    visits = pl.scan_parquet(src)
    keep = [
        c
        for c in visits.collect_schema().names()
        if not c.startswith("sem_code_")
    ]
    if "placekey_id" not in keep:
        raise ValueError("augmented visits are missing placekey_id")
    if "placekey" in keep:
        keep.remove("placekey")
        keep.append("placekey")
    visit_rows = visits.select(keep)
    if "placekey" in keep:
        visit_rows = visit_rows.rename({"placekey": "visit_placekey"})
    visit_rows = visit_rows.with_columns(pl.col("placekey_id").cast(pl.Int64))
    visit_rows = visit_rows.join(
        local_places_frame.lazy(), on="placekey_id", how="left"
    )
    if "visit_placekey" in visit_rows.collect_schema().names():
        visit_rows = visit_rows.filter(
            pl.col("visit_placekey") == pl.col("placekey")
        ).drop("visit_placekey")
    composed = visit_rows.join(
        lookup.lazy().select("placekey", *code_cols),
        on="placekey",
        how="left",
    ).collect(engine="streaming")
    total_frame = visits.select(pl.len()).collect()
    total = total_frame.item()
    logger.info(
        "augmented visits: %d of %d rows retained (%.4f)",
        composed.height,
        total,
        composed.height / total if total else 0.0,
    )
    if composed.height != total:
        raise ValueError(
            "the local places namespace validation changed the visit row "
            f"count ({composed.height} of {total}); refusing to truncate"
        )
    if composed.filter(pl.col(code_cols[0]).is_null()).height:
        raise ValueError(
            "some visit rows have no composed code after the placekey bridge; "
            "refusing to write a partial augmented artifact"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup.write_parquet(output_dir / "placekey_lookup.parquet")
    composed.write_parquet(output_dir / "augmented_processed_visits.parquet")

    shared = extract_codebook_bundle(shared_codebooks).codebooks
    local = extract_codebook_bundle(
        local_dir / "tokenizer_codebooks.pt"
    ).codebooks
    if shared.shape[1:] != local.shape[1:]:
        raise ValueError(
            f"shared codebooks {tuple(shared.shape)} and local "
            f"{tuple(local.shape)} disagree beyond the level axis"
        )
    stacked = torch.cat([shared, local], dim=0)
    torch.save(
        {
            "codebooks": stacked,
            "code_fields": tuple(code_cols),
            "method": "shared_local_product",
            "product_code": True,
            "shared_levels": int(spec["shared_levels"]),
            "local_levels": int(spec["local_levels"]),
            "shared_sid_version": spec["shared_sid_version"],
            "shared_codebook_fingerprint": spec["shared_codebook_fingerprint"],
            "city_id": city_id,
            "warning": (
                "Two independent quantizers concatenated on the level axis. "
                "Summing all levels is NOT a valid reconstruction; levels "
                f"{spec['shared_levels']}.. are city-local to city {city_id}."
            ),
        },
        output_dir / "tokenizer_codebooks.pt",
    )
    for name in ("config.yaml", "config_resolved.yaml"):
        source = local_dir / name
        if source.exists():
            shutil.copy(source, output_dir / name)

    logger.info("wrote %s  codebooks %s", output_dir, tuple(stacked.shape))
    return ComposedTokenizerDirResult(
        output_dir=str(output_dir),
        city_id=city_id,
        n_pois=lookup.height,
        n_levels=len(code_cols),
        shared_levels=int(spec["shared_levels"]),
        local_levels=int(spec["local_levels"]),
        shared_codebook_fingerprint=spec["shared_codebook_fingerprint"],
        codebook_shape=tuple(stacked.shape),
        augmented_visit_rows=composed.height,
    )


__all__ = ["ComposedTokenizerDirResult", "assemble_composed_tokenizer_dir"]
