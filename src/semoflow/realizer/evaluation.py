"""Compose planner samples with the realizer and score both arms.

The gold-prefix arm completes the reference split's own prefixes; the
sampled-plan arm completes the planner's samples.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl
import torch

from semoflow.metrics.evaluation import (
    aligned_reference_event_hours,
    load_candidate_prior,
    load_places,
    load_reference,
    metric_block,
    plan_from_planner_samples,
)
from semoflow.realizer.bank import (
    bank_split_city_ids,
    bank_split_window,
    load_bank_payload,
)
from semoflow.realizer.model import PrefixTimeSuffixLatentFlowRealizer
from semoflow.realizer.sampling import (
    CATALOG_NOISE_COUPLINGS,
    PrefixTailCatalog,
    sample_prefix_time_suffix_latent_flow,
)
from semoflow.tokenization.sids import (
    SemanticIDGroundingIndex,
    sid_code_columns,
)
from semoflow.utils.serialization import normalize_value, write_json

LOGGER = logging.getLogger(__name__)

# Joint JSDs nested in an arm's metrics, lifted into its summary.
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


def load_realizer_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[PrefixTimeSuffixLatentFlowRealizer, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    state = checkpoint["model_state_dict"]
    codebooks = checkpoint["codebooks"].float().contiguous()
    prefix_depth = int(checkpoint["prefix_depth"])
    model = PrefixTimeSuffixLatentFlowRealizer(
        max_seq_len=int(state["event_position_embedding.weight"].size(0)),
        prefix_depth=prefix_depth,
        num_codebooks=int(checkpoint["num_codebooks"]),
        codebook_size=int(checkpoint["codebook_size"]),
        time_vocab=int(checkpoint["time_vocab"]),
        latent_dim=int(checkpoint["latent_dim"]),
        model_dim=int(config["model_dim"]),
        condition_depth=int(config["condition_depth"]),
        depth=int(config["depth"]),
        num_heads=int(config["num_heads"]),
        mlp_ratio=float(config["mlp_ratio"]),
        dropout=float(config["dropout"]),
        prefix_plan_codebooks=codebooks[:prefix_depth],
        city_vocab=int(config.get("city_vocab", 0)),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def realizer_max_events(checkpoint: Path) -> int:
    """The number of events per trajectory a realizer checkpoint models."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    position = payload["model_state_dict"]["event_position_embedding.weight"]
    return int(position.shape[0])


def summary(metrics: Mapping[str, Any]) -> dict[str, object]:
    """An arm's scalar metrics, with the nested joint JSDs lifted out.

    A metric the arm could not score is ``None``.
    """
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
    places_path: Path,
) -> PrefixTailCatalog:
    """The legal SID catalog generation decodes into, with the train-split
    log-count prior and, for a city-conditioned realizer, each SID's city."""
    lookup = pl.read_parquet(lookup_path)
    columns = list(sid_code_columns(lookup.columns))
    catalog_codes = torch.from_numpy(
        lookup.select(columns).to_numpy().astype("int64")
    )
    offset, count = bank_split_window(payload, "train")
    rows = slice(offset, offset + count)
    observed = payload["sid_codes"][rows][payload["valid_mask"][rows].bool()]
    city = None
    if model.city_vocab > 0:
        places = pl.read_parquet(
            places_path, columns=["placekey_id", "city_id"]
        )
        city_of_sid = (
            lookup.select("placekey_id", *columns)
            .join(places, on="placekey_id", how="left", maintain_order="left")
            .drop_nulls("city_id")
            .unique(columns, keep="first", maintain_order=True)
            .drop("placekey_id")
        )
        city_column = (
            lookup.select(columns)
            .join(city_of_sid, on=columns, how="left", maintain_order="left")[
                "city_id"
            ]
            .fill_null(-1)
        )
        missing = int((city_column == -1).sum())
        if missing:
            LOGGER.warning(
                "%d catalog rows have no city and cannot satisfy the "
                "constraint",
                missing,
            )
        city = torch.from_numpy(city_column.to_numpy().astype("int64"))
    return PrefixTailCatalog.from_codes(
        catalog_codes,
        codebooks,
        prefix_depth=model.prefix_depth,
        observed_codes=observed,
        candidate_city=city,
    )


@dataclass(frozen=True)
class _Arm:
    """One conditioning source for the realizer: gold or sampled prefixes."""

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
    catalog: PrefixTailCatalog,
    checkpoint: Mapping[str, Any],
    catalog_temperature: float,
    batch_size: int,
    device: torch.device,
    num_steps: int,
    solver: str,
    catalog_noise_coupling: str,
) -> tuple[torch.Tensor, dict[str, int]]:
    """The arm's full SIDs, and how the city constraint applied.

    Rows without a supported event are sampled from a placeholder first
    event and returned all zero.
    """
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
            checkpoint["config"].get("catalog_prior_weight", 0.25)
        ),
        catalog_temperature=catalog_temperature,
        num_steps=num_steps,
        solver=solver,
        device=device,
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
    candidate_prior: Path | None = None,
    catalog_temperature: float | None = None,
    reference_split: str = "val",
    catalog_noise_coupling: str = "independent",
) -> dict[str, Any]:
    """Complete planner samples with the realizer, score both arms against
    the reference split, and write the report to ``output``."""
    if max_samples < 1 or batch_size < 1:
        raise ValueError("max_samples and batch_size must be positive")
    if catalog_noise_coupling not in CATALOG_NOISE_COUPLINGS:
        raise ValueError(
            f"catalog_noise_coupling must be one of {CATALOG_NOISE_COUPLINGS}"
        )
    if candidate_selection != "first" and candidate_prior is None:
        raise ValueError("--candidate-selection requires --candidate-prior")
    started = time.perf_counter()
    resolved_device = torch.device(device)
    model, checkpoint = load_realizer_checkpoint(
        realizer_checkpoint, resolved_device
    )
    temperature = float(
        checkpoint["config"].get("catalog_temperature", 1.0)
        if catalog_temperature is None
        else catalog_temperature
    )
    if temperature <= 0.0:
        raise ValueError("catalog temperature must be positive")

    payload = load_bank_payload(bank)
    real_codes, real_times, real_valid, real_placekey_ids = load_reference(
        payload, split=reference_split, max_samples=max_samples
    )
    plan = plan_from_planner_samples(
        torch.load(planner_samples, map_location="cpu", weights_only=False),
        width=model.max_seq_len,
        prefix_depth=model.prefix_depth,
        max_rows=max_samples,
    )
    n_rows = min(real_codes.size(0), plan.prefix.size(0))
    real_codes, real_times, real_valid, real_placekey_ids = (
        tensor[:n_rows]
        for tensor in (real_codes, real_times, real_valid, real_placekey_ids)
    )
    real_hours = aligned_reference_event_hours(
        payload, split=reference_split, valid_mask=real_valid
    )
    catalog = _catalog(
        model,
        checkpoint["codebooks"].float(),
        payload,
        lookup_path=lookup,
        places_path=places,
    )

    gold_city = sampled_city = None
    if model.city_vocab > 0:
        if plan.city_id is None:
            raise ValueError(
                "planner samples carry no city_ids; a city-conditioned "
                "realizer cannot be evaluated on them"
            )
        gold_city = bank_split_city_ids(payload, reference_split, rows=n_rows)
        sampled_city = plan.city_id[:n_rows]
    gold_prefix = real_codes[..., : model.prefix_depth]
    sampled_prefix = plan.prefix[:n_rows]
    sampled_valid = plan.valid[:n_rows]
    arms = (
        _Arm(
            name="gold_prefix",
            prefix=gold_prefix,
            times=real_times,
            input_valid=real_valid,
            supported=real_valid & catalog.supports(gold_prefix).cpu(),
            hours=real_hours,
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
            real_codes,
            real_valid,
            sampled[arm.name][0],
            arm.supported,
            grounding=grounding,
            coordinates=coordinates,
            geohashes=geohashes,
            placekey_vocab_size=placekey_vocab_size,
            reference_placekey_ids=real_placekey_ids,
            real_time_tokens=real_times,
            generated_time_tokens=arm.times,
            real_event_hours=real_hours,
            generated_event_hours=arm.hours,
            candidate_selection=candidate_selection,
            candidate_weights=weights,
            compute_device=resolved_device,
        )
        LOGGER.info(
            "%s metrics on %d rows in %.1fs",
            arm.name,
            n_rows,
            time.perf_counter() - arm_started,
        )

    result: dict[str, Any] = {
        "checkpoint": str(realizer_checkpoint),
        "planner_samples": str(planner_samples),
        "reference_bank": str(bank),
        "reference_split": reference_split,
        "rows": n_rows,
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
    LOGGER.info(
        "composition complete  output=%s  elapsed=%.1fs",
        output,
        time.perf_counter() - started,
    )
    return result


__all__ = [
    "evaluate_planner_realizer",
    "load_realizer_checkpoint",
    "realizer_max_events",
    "summary",
]
