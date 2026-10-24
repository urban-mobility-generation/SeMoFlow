"""Per-city sampling and composition behind ``semoflow-eval evaluate``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import polars as pl

from semoflow.config.artifacts import TokenizerArtifacts
from semoflow.config.schemas.evaluation_config import EvaluationConfig
from semoflow.metrics.city_inputs import (
    GROUNDING_NAME,
    LOOKUP_NAME,
    SEGMENTS_NAME,
    extract_city_inputs,
)
from semoflow.metrics.grounding_candidate_prior import build_prior
from semoflow.metrics.planner_realizer_composition import (
    evaluate_planner_realizer,
    realizer_contract,
)
from semoflow.planner.conditions_build import (
    build_occurrence_planner_conditions,
)
from semoflow.planner.sampling_run import sample_occurrence_planner
from semoflow.realizer.bank import build_bank
from semoflow.utils.serialization import write_json

logger = logging.getLogger(__name__)


def evaluate_cities(
    settings: EvaluationConfig, *, force: bool = False
) -> dict[str, dict[str, Any]]:
    """Sample, compose, and score trajectories for each configured city.

    The city banks keep weeks up to the realizer's sequence length, as its
    training bank did.
    """
    tokenizer = TokenizerArtifacts.from_run_dir(settings.tokenization_dir)
    max_events = realizer_contract(settings.realizer_checkpoint)["max_seq_len"]
    return {
        name: _evaluate_city(
            settings,
            city_id,
            tokenizer=tokenizer,
            max_events=max_events,
            output_dir=settings.output_dir / name,
            force=force,
        )
        for name, city_id in settings.cities.items()
    }


def _evaluate_city(
    settings: EvaluationConfig,
    city_id: int,
    *,
    tokenizer: TokenizerArtifacts,
    max_events: int,
    output_dir: Path,
    force: bool,
) -> dict[str, Any]:
    logger.info("evaluating city_id=%d output_dir=%s", city_id, output_dir)
    extract_city_inputs(
        city_id,
        segments=settings.segments_path,
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
        write_json(prior, build_prior(bank))

    # One plan per reference week.
    num_samples = row_counts[settings.reference_split]
    conditions = output_dir / "conditions.pt"
    build_occurrence_planner_conditions(
        checkpoint_path=settings.planner_checkpoint,
        rows=pl.DataFrame({"city_id": [city_id] * num_samples}),
        output_path=conditions,
        force=force,
    )
    plans = output_dir / "plans.pt"
    sampling = settings.sampling
    sample_occurrence_planner(
        checkpoint_path=settings.planner_checkpoint,
        output_path=plans,
        num_samples=num_samples,
        batch_size=sampling.batch_size,
        conditions_path=conditions,
        device=settings.device,
        seed=settings.seed,
        temperature=sampling.temperature,
        route_temperature=sampling.route_temperature,
        force=force,
    )
    output = output_dir / "composition.json"
    report = evaluate_planner_realizer(
        realizer_checkpoint=settings.realizer_checkpoint,
        planner_samples=plans,
        planner_checkpoint=settings.planner_checkpoint,
        bank=bank,
        codebooks=tokenizer.tokenizer_codebooks,
        lookup=output_dir / LOOKUP_NAME,
        grounding_index=output_dir / GROUNDING_NAME,
        places=settings.places_path,
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
        brief_metrics=composition.brief_metrics,
    )
    return {"output": output, "samples": num_samples, **report["summary"]}


__all__ = ["evaluate_cities"]
