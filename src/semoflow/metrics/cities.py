"""Per-city evaluation behind ``semoflow evaluate``.

Each city's inputs are cut from the joint artifacts; its bank, train-split
POI prior and sampled plans are then composed and scored.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, cast

import polars as pl

from semoflow.config import CITY_ID, EvaluationConfig, RunLayout
from semoflow.planner.sampling import sample_occurrence_planner
from semoflow.preprocess import SEGMENT_CITY_COLUMN
from semoflow.realizer.bank import bank_segments, build_bank, load_bank_payload
from semoflow.realizer.evaluation import (
    evaluate_planner_realizer,
    realizer_max_events,
)
from semoflow.tokenization.sids import (
    SemanticIDGroundingIndex,
)
from semoflow.utils.serialization import write_json

logger = logging.getLogger(__name__)

SEGMENTS_NAME = "segments.parquet"
LOOKUP_NAME = "placekey_lookup.parquet"
GROUNDING_NAME = "grounding_index.pkl"


def evaluate_cities(
    settings: EvaluationConfig,
    layout: RunLayout,
    cities: list[str],
    *,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Sample, compose, and score trajectories for each configured city.

    The city banks keep weeks up to the realizer's sequence length, as its
    training bank did.
    """
    max_events = realizer_max_events(layout.realizer_dir / "best.pt")
    return {
        name: _evaluate_city(
            settings,
            layout,
            CITY_ID[name],
            max_events=max_events,
            output_dir=layout.evaluation_dir / name,
            force=force,
        )
        for name in cities
    }


def _evaluate_city(
    settings: EvaluationConfig,
    layout: RunLayout,
    city_id: int,
    *,
    max_events: int,
    output_dir: Path,
    force: bool,
) -> dict[str, Any]:
    logger.info("evaluating city_id=%d output_dir=%s", city_id, output_dir)
    tokenizer = layout.tokenizer
    _extract_city_inputs(
        city_id,
        segments=layout.segments,
        lookup=tokenizer.placekey_lookup,
        grounding_index=tokenizer.grounding_index,
        output_dir=output_dir,
        force=force,
    )
    bank = output_dir / "bank.pt"
    row_counts = cast(
        dict[str, int],
        build_bank(
            lookup_path=output_dir / LOOKUP_NAME,
            segments_path=output_dir / SEGMENTS_NAME,
            output_path=bank,
            max_events=max_events,
        )["row_counts"],
    )
    composition = settings.composition
    prior = None
    if composition.candidate_selection != "first":
        prior = output_dir / "candidate_prior.json"
        write_json(prior, _build_prior(bank))

    num_samples = row_counts[settings.reference_split]
    plans = output_dir / "plans.pt"
    sampling = settings.sampling
    sample_occurrence_planner(
        checkpoint_path=layout.planner_dir / "best.pt",
        output_path=plans,
        num_samples=num_samples,
        city_id=city_id,
        batch_size=sampling.batch_size,
        device=settings.device,
        seed=settings.seed,
        temperature=sampling.temperature,
        route_temperature=sampling.route_temperature,
        force=force,
    )
    output = output_dir / "composition.json"
    report = evaluate_planner_realizer(
        realizer_checkpoint=layout.realizer_dir / "best.pt",
        planner_samples=plans,
        bank=bank,
        lookup=output_dir / LOOKUP_NAME,
        grounding_index=output_dir / GROUNDING_NAME,
        places=layout.preprocessing.places_index,
        output=output,
        max_samples=num_samples,
        batch_size=composition.batch_size,
        device=settings.device,
        num_steps=composition.num_steps,
        solver=composition.solver,
        reference_split=settings.reference_split,
        candidate_selection=composition.candidate_selection,
        candidate_prior=prior,
        catalog_temperature=composition.catalog_temperature,
        catalog_noise_coupling=composition.catalog_noise_coupling,
    )
    return {"output": output, "samples": num_samples, **report["summary"]}


def _extract_city_inputs(
    city_id: int,
    *,
    segments: Path,
    lookup: Path,
    grounding_index: Path,
    output_dir: Path,
    force: bool = False,
) -> None:
    """Write ``city_id``'s segments, SID lookup, and grounding index.

    Each output keeps the joint artifact's rows, order, and identifiers, so
    the bank, candidate prior, and composition built from them are those of
    the joint run restricted to one city.
    """
    outputs = {
        name: output_dir / name
        for name in (SEGMENTS_NAME, LOOKUP_NAME, GROUNDING_NAME)
    }
    if not force:
        for path in outputs.values():
            if path.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing artifact: {path}"
                )

    city_lookup = pl.read_parquet(lookup).filter(pl.col("city_id") == city_id)
    city_segments = pl.read_parquet(segments).filter(
        pl.col(SEGMENT_CITY_COLUMN) == city_id
    )
    if city_lookup.is_empty() or city_segments.is_empty():
        raise ValueError(f"city {city_id} has no lookup rows or segments")
    joint = SemanticIDGroundingIndex.load(grounding_index)
    city_grounding = SemanticIDGroundingIndex.from_frame(
        joint.to_frame().filter(
            pl.col("placekey_id").is_in(city_lookup["placekey_id"].implode())
        ),
        payload_columns=joint.payload_columns,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    city_segments.write_parquet(outputs[SEGMENTS_NAME])
    city_lookup.write_parquet(outputs[LOOKUP_NAME])
    city_grounding.save(outputs[GROUNDING_NAME])


def _build_prior(bank_path: Path) -> dict[str, Any]:
    """Per-POI visit counts over the bank's train rows."""
    payload = load_bank_payload(bank_path)
    frame = bank_segments(payload, "train", ["placekey_id"])
    counter = Counter(int(value) for value in frame["placekey_id"].explode())
    if not counter:
        raise ValueError("no POI occurrences found in the train window")
    return {
        "split": "train",
        "bank": str(bank_path),
        "rows": frame.height,
        "events": sum(counter.values()),
        "distinct_pois": len(counter),
        "counts": {str(key): value for key, value in counter.items()},
    }


__all__ = ["evaluate_cities"]
