from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch

from semoflow.tokenization.cross_city_projection import Projection
from semoflow.utils.serialization import write_json

logger = logging.getLogger(__name__)

SourceTemplate = Literal["address_stripped", "full"]


@dataclass(frozen=True)
class CrossCityProjectionApplyResult:
    output_dir: str
    rows: int
    out_dim: int
    projection_fitted_cities: list[int]
    applied_cities: list[int]
    source_template: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def apply_cross_city_projection(
    *,
    projection: Path,
    embeddings: Path,
    embeddings_index: Path,
    places_index: Path,
    output_dir: Path,
    source_template: SourceTemplate,
    city_ids: list[int] | None = None,
    device: str = "cpu",
    batch_size: int = 8192,
) -> CrossCityProjectionApplyResult:
    payload = torch.load(projection, map_location="cpu", weights_only=False)
    fitted = payload.get("fitted_cities", [])
    logger.info(
        "frozen projection: in=%d hidden=%d out=%d fitted on cities %s "
        "(key=%s, behavioural supervision=%s)",
        payload["in_dim"],
        payload["hidden"],
        payload["out_dim"],
        fitted,
        payload.get("correspondence_key"),
        payload.get("behavioural_supervision"),
    )
    if source_template != "address_stripped":
        logger.warning(
            "applying a projection fitted on address-stripped text to "
            "%s-template embeddings; the input distribution differs and the "
            "result is not comparable to the fitted panel",
            source_template,
        )

    index = pl.read_parquet(embeddings_index)
    places = pl.read_parquet(places_index)
    frame = index.join(places, on="placekey_id", how="inner").sort(
        "embedding_row"
    )
    if city_ids:
        frame = frame.filter(pl.col("city_id").is_in(city_ids))
        overlap = sorted(set(city_ids) & {int(c) for c in fitted})
        if overlap:
            logger.warning(
                "cities %s were in the projection's own fit; this is not a "
                "held-out measurement for them",
                overlap,
            )
    if frame.is_empty():
        raise ValueError("no rows selected; check city_ids and the indexes")

    emb = np.load(embeddings, mmap_mode="r")
    if emb.shape[1] != payload["in_dim"]:
        raise ValueError(
            f"source embeddings are {emb.shape[1]}-d but the projection "
            f"expects {payload['in_dim']}-d"
        )

    model = Projection(
        payload["in_dim"], payload["hidden"], payload["out_dim"]
    )
    model.load_state_dict(payload["state_dict"])
    model.eval().to(device)

    rows = frame["embedding_row"].to_numpy()
    out = np.zeros((len(rows), payload["out_dim"]), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            block = np.asarray(
                emb[rows[start : start + batch_size]], dtype=np.float32
            )
            out[start : start + batch_size] = (
                model(torch.from_numpy(block).to(device)).cpu().numpy()
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "text_embeddings.npy", out)
    frame.select("placekey_id").with_columns(
        pl.arange(0, frame.height, eager=True).alias("embedding_row")
    ).select("embedding_row", "placekey_id").write_parquet(
        output_dir / "text_embeddings.index.parquet"
    )
    applied_cities = sorted(
        int(c) for c in frame["city_id"].unique().to_list()
    )
    write_json(
        output_dir / "projection_provenance.json",
        {
            "format_version": "cross_city_projection_applied_v1",
            "projection": str(projection),
            "projection_fitted_cities": fitted,
            "applied_cities": applied_cities,
            "source_embeddings": str(embeddings),
            "source_template": source_template,
            "rows": int(frame.height),
        },
    )
    logger.info("wrote %s  shape %s", output_dir, out.shape)
    return CrossCityProjectionApplyResult(
        output_dir=str(output_dir),
        rows=int(frame.height),
        out_dim=int(payload["out_dim"]),
        projection_fitted_cities=[int(c) for c in fitted],
        applied_cities=applied_cities,
        source_template=source_template,
    )


__all__ = [
    "CrossCityProjectionApplyResult",
    "SourceTemplate",
    "apply_cross_city_projection",
]
