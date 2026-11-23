from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl
import torch

from semoflow.config import CITY_ID, EvaluationConfig, RunLayout
from semoflow.data import (
    SEGMENT_CITY_COLUMN,
    bank_segments,
    bank_split_city_ids,
    bank_split_window,
    build_bank,
    load_bank_payload,
)
from semoflow.evaluation.scoring import (
    SampledPlan,
    aligned_reference_event_hours,
    load_candidate_prior,
    load_places,
    load_reference,
    metric_block,
    plan_from_planner_samples,
)
from semoflow.planner.sampling import sample_occurrence_planner
from semoflow.poi.sids import SemanticIDGroundingIndex
from semoflow.realizer.catalog import SuffixCatalog
from semoflow.realizer.model import (
    PrefixTimeSuffixLatentFlowRealizer,
    load_realizer_checkpoint,
    realizer_max_events,
)
from semoflow.realizer.sampling import sample_prefix_time_suffix_latent_flow
from semoflow.utils import normalize_value, write_json

logger = logging.getLogger(__name__)


SEGMENTS_NAME = "segments.parquet"


LOOKUP_NAME = "placekey_lookup.parquet"


GROUNDING_NAME = "grounding_index.pkl"


def evaluate_cities(
    settings: EvaluationConfig,
    layout: RunLayout,
    cities: list[str],
) -> dict[str, dict[str, Any]]:
    max_events = realizer_max_events(layout.realizer_dir / "best.pt")
    return {
        name: _evaluate_city(
            settings,
            layout,
            CITY_ID[name],
            max_events=max_events,
            output_dir=layout.evaluation_dir / name,
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
) -> dict[str, Any]:
    logger.info("evaluating city_id=%d output_dir=%s", city_id, output_dir)
    tokenizer = layout.tokenizer
    _extract_city_inputs(
        city_id,
        segments=layout.segments,
        lookup=tokenizer.placekey_lookup,
        grounding_index=tokenizer.grounding_index,
        output_dir=output_dir,
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
        force=True,
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
) -> None:
    city_lookup = pl.read_parquet(lookup).filter(pl.col("city_id") == city_id)
    city_segments = pl.read_parquet(segments).filter(
        pl.col(SEGMENT_CITY_COLUMN) == city_id
    )
    joint = SemanticIDGroundingIndex.load(grounding_index)
    city_grounding = SemanticIDGroundingIndex.from_frame(
        joint.to_frame().filter(
            pl.col("placekey_id").is_in(city_lookup["placekey_id"].implode())
        ),
        payload_columns=joint.payload_columns,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    city_segments.write_parquet(output_dir / SEGMENTS_NAME)
    city_lookup.write_parquet(output_dir / LOOKUP_NAME)
    city_grounding.save(output_dir / GROUNDING_NAME)


def _build_prior(bank_path: Path) -> dict[str, Any]:
    payload = load_bank_payload(bank_path)
    frame = bank_segments(payload, "train", ["placekey_id"])
    counter = Counter(int(value) for value in frame["placekey_id"].explode())
    return {
        "split": "train",
        "bank": str(bank_path),
        "rows": frame.height,
        "events": sum(counter.values()),
        "distinct_pois": len(counter),
        "counts": {str(key): value for key, value in counter.items()},
    }


_NESTED_SUMMARY_KEYS = {
    "time_location_joint_jsd": (
        "temporal_spatial",
        "time_location",
        "joint_jsd",
    ),
    "gap_distance_joint_jsd": (
        "temporal_spatial",
        "gap_distance",
        "joint_jsd",
    ),
    "inter_return_time_jsd": ("elapsed_recurrence", "inter_return_time_jsd"),
    "return_index_time_joint_jsd": (
        "elapsed_recurrence",
        "return_index_time_joint_jsd",
    ),
}


def summary(metrics: Mapping[str, Any]) -> dict[str, object]:
    out: dict[str, object] = {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, dict)
    }
    for key, (section, *path) in _NESTED_SUMMARY_KEYS.items():
        node: Any = metrics[section]
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        out[key] = node
    return out


def _catalog(
    model: PrefixTimeSuffixLatentFlowRealizer,
    codebooks: torch.Tensor,
    payload: dict[str, Any],
    *,
    lookup_path: Path,
) -> SuffixCatalog:
    offset, count = bank_split_window(payload, "train")
    rows = slice(offset, offset + count)
    return SuffixCatalog.from_lookup(
        lookup_path,
        codebooks,
        prefix_depth=model.prefix_depth,
        observed_codes=payload["sid_codes"][rows][
            payload["valid_mask"][rows].bool()
        ],
        with_city=model.city_vocab > 0,
    )


@dataclass(frozen=True)
class _Arm:
    name: str
    prefix: torch.Tensor
    times: torch.Tensor
    input_valid: torch.Tensor
    supported: torch.Tensor
    hours: torch.Tensor
    seed: int
    city_id: torch.Tensor | None

    def support(self) -> dict[str, int]:
        valid, supported = self.input_valid, self.supported
        return {
            "input_events": int(valid.sum()),
            "supported_events": int(supported.sum()),
            "rows_without_support": int((~supported.any(dim=1)).sum()),
        }


def _sample_arm(
    model: PrefixTimeSuffixLatentFlowRealizer,
    arm: _Arm,
    *,
    catalog: SuffixCatalog,
    checkpoint: Mapping[str, Any],
    catalog_temperature: float,
    batch_size: int,
    device: torch.device,
    num_steps: int,
    solver: str,
    catalog_noise_coupling: str,
) -> tuple[torch.Tensor, dict[str, int]]:
    prefix, times, valid = arm.prefix, arm.times, arm.supported.clone()
    empty = ~valid.any(dim=1)
    if empty.any():
        prefix, times = prefix.clone(), times.clone()
        prefix[empty, 0] = catalog.codes[0, : model.prefix_depth].cpu()
        times[empty, 0] = 0
        valid[empty, 0] = True
    sampled = sample_prefix_time_suffix_latent_flow(
        model,
        prefix.to(device),
        times.to(device),
        valid.to(device),
        catalog=catalog,
        tail_mean=checkpoint["tail_mean"].float().to(device),
        tail_whitener=checkpoint["tail_whitener"].float().to(device),
        catalog_prior_weight=float(
            checkpoint["config"]["catalog_prior_weight"]
        ),
        catalog_temperature=catalog_temperature,
        num_steps=num_steps,
        solver=solver,
        generator=torch.Generator(device=device).manual_seed(arm.seed),
        max_batch_size=batch_size,
        city_id=None if arm.city_id is None else arm.city_id.to(device),
        catalog_noise_coupling=catalog_noise_coupling,
    )
    full_sids = sampled.full_sids.cpu()
    full_sids[empty] = 0
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return full_sids, {
        "constrained_events": sampled.city_constrained_events,
        "fallback_events": sampled.city_constraint_fallback_events,
    }


@dataclass(frozen=True)
class _Reference:
    codes: torch.Tensor
    times: torch.Tensor
    valid: torch.Tensor
    placekey_ids: torch.Tensor
    hours: torch.Tensor

    @property
    def rows(self) -> int:
        return int(self.codes.size(0))


def _load_reference_and_plan(
    payload: dict[str, Any],
    planner_samples: Path,
    *,
    split: str,
    width: int,
    max_samples: int,
) -> tuple[_Reference, SampledPlan]:
    codes, times, valid, placekey_ids = load_reference(
        payload, split=split, max_samples=max_samples
    )
    plan = plan_from_planner_samples(
        torch.load(planner_samples, map_location="cpu", weights_only=False),
        width=width,
        max_rows=max_samples,
    )
    n_rows = min(codes.size(0), plan.prefix.size(0))
    codes, times, valid, placekey_ids = (
        tensor[:n_rows] for tensor in (codes, times, valid, placekey_ids)
    )
    hours = aligned_reference_event_hours(
        payload, split=split, valid_mask=valid
    )
    return _Reference(codes, times, valid, placekey_ids, hours), plan


def _arms(
    model: PrefixTimeSuffixLatentFlowRealizer,
    catalog: SuffixCatalog,
    reference: _Reference,
    plan: SampledPlan,
    payload: dict[str, Any],
    split: str,
) -> tuple[_Arm, _Arm]:
    n_rows = reference.rows
    gold_city = sampled_city = None
    if model.city_vocab > 0:
        gold_city = bank_split_city_ids(payload, split, rows=n_rows)
        sampled_city = plan.city_id[:n_rows]
    gold_prefix = reference.codes[..., : model.prefix_depth]
    sampled_prefix = plan.prefix[:n_rows]
    sampled_valid = plan.valid[:n_rows]
    return (
        _Arm(
            name="gold_prefix",
            prefix=gold_prefix,
            times=reference.times,
            input_valid=reference.valid,
            supported=reference.valid & catalog.supports(gold_prefix).cpu(),
            hours=reference.hours,
            seed=7000,
            city_id=gold_city,
        ),
        _Arm(
            name="sampled_plan",
            prefix=sampled_prefix,
            times=plan.time_tokens[:n_rows],
            input_valid=sampled_valid,
            supported=sampled_valid & catalog.supports(sampled_prefix).cpu(),
            hours=plan.event_hours[:n_rows],
            seed=17000,
            city_id=sampled_city,
        ),
    )


def _score_arms(
    reference: _Reference,
    arms: tuple[_Arm, _Arm],
    generated: dict[str, torch.Tensor],
    *,
    grounding_index: Path,
    places: Path,
    candidate_selection: str,
    candidate_prior: Path | None,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    grounding = SemanticIDGroundingIndex.load(grounding_index)
    coordinates, geohashes, placekey_vocab_size = load_places(places)
    weights = (
        None
        if candidate_prior is None
        else load_candidate_prior(candidate_prior)
    )
    evaluations: dict[str, dict[str, Any]] = {}
    for arm in arms:
        arm_started = time.perf_counter()
        evaluations[arm.name] = metric_block(
            reference.codes,
            reference.valid,
            generated[arm.name],
            arm.supported,
            grounding=grounding,
            coordinates=coordinates,
            geohashes=geohashes,
            placekey_vocab_size=placekey_vocab_size,
            reference_placekey_ids=reference.placekey_ids,
            real_time_tokens=reference.times,
            generated_time_tokens=arm.times,
            real_event_hours=reference.hours,
            generated_event_hours=arm.hours,
            candidate_selection=candidate_selection,
            candidate_weights=weights,
            compute_device=device,
        )
        logger.info(
            "%s metrics on %d rows in %.1fs",
            arm.name,
            reference.rows,
            time.perf_counter() - arm_started,
        )
    return evaluations


def evaluate_planner_realizer(
    *,
    realizer_checkpoint: Path,
    planner_samples: Path,
    bank: Path,
    lookup: Path,
    grounding_index: Path,
    places: Path,
    output: Path,
    max_samples: int,
    batch_size: int,
    device: str,
    num_steps: int,
    solver: str,
    candidate_selection: str,
    candidate_prior: Path | None,
    catalog_temperature: float | None,
    reference_split: str,
    catalog_noise_coupling: str,
) -> dict[str, Any]:
    if candidate_selection != "first" and candidate_prior is None:
        raise ValueError("--candidate-selection requires --candidate-prior")
    started = time.perf_counter()
    resolved_device = torch.device(device)
    model, checkpoint = load_realizer_checkpoint(
        realizer_checkpoint, resolved_device
    )
    temperature = float(
        checkpoint["config"]["catalog_temperature"]
        if catalog_temperature is None
        else catalog_temperature
    )
    payload = load_bank_payload(bank)
    reference, plan = _load_reference_and_plan(
        payload,
        planner_samples,
        split=reference_split,
        width=model.max_seq_len,
        max_samples=max_samples,
    )
    catalog = _catalog(
        model,
        checkpoint["codebooks"].float(),
        payload,
        lookup_path=lookup,
    )
    arms = _arms(model, catalog, reference, plan, payload, reference_split)
    sampled = {
        arm.name: _sample_arm(
            model,
            arm,
            catalog=catalog,
            checkpoint=checkpoint,
            catalog_temperature=temperature,
            batch_size=batch_size,
            device=resolved_device,
            num_steps=num_steps,
            solver=solver,
            catalog_noise_coupling=catalog_noise_coupling,
        )
        for arm in arms
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gold_prefix_full_sids": sampled["gold_prefix"][0],
            "sampled_plan_full_sids": sampled["sampled_plan"][0],
            "sampled_plan_valid_mask": arms[1].supported,
            "sampled_plan_input_valid_mask": arms[1].input_valid,
        },
        output.with_name("generated_full_sids.pt"),
    )
    evaluations = _score_arms(
        reference,
        arms,
        {name: full_sids for name, (full_sids, _) in sampled.items()},
        grounding_index=grounding_index,
        places=places,
        candidate_selection=candidate_selection,
        candidate_prior=candidate_prior,
        device=resolved_device,
    )
    result: dict[str, Any] = {
        "checkpoint": str(realizer_checkpoint),
        "planner_samples": str(planner_samples),
        "reference_bank": str(bank),
        "reference_split": reference_split,
        "rows": reference.rows,
        "sampling": {"solver": solver, "num_steps": num_steps},
        "decoding": {
            "catalog_temperature": temperature,
            "catalog_noise_coupling": catalog_noise_coupling,
            "candidate_selection": candidate_selection,
            "candidate_prior": None
            if candidate_prior is None
            else str(candidate_prior),
        },
        "catalog_support": {arm.name: arm.support() for arm in arms},
        "city_constraint": {
            name: stats for name, (_, stats) in sampled.items()
        },
        "evaluations": evaluations,
        "summary": {
            name: summary(metrics) for name, metrics in evaluations.items()
        },
    }
    result = cast(dict[str, Any], normalize_value(result))
    write_json(output, result)
    logger.info(
        "composition complete  output=%s  elapsed=%.1fs",
        output,
        time.perf_counter() - started,
    )
    return result


__all__ = [
    "evaluate_cities",
    "evaluate_planner_realizer",
    "summary",
]
