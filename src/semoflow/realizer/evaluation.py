from __future__ import annotations

import dataclasses
import functools
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import torch

from semoflow.metrics.prefix_time_suffix_evaluation import (
    aligned_reference_event_hours,
    load_candidate_prior,
    load_lookup,
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
from semoflow.realizer.model import (
    MODEL_TYPE,
    PrefixTimeSuffixLatentFlowRealizer,
)
from semoflow.realizer.sampling import (
    CATALOG_NOISE_COUPLINGS,
    PrefixTailCatalog,
    PrefixTimeSuffixLatentFlowSample,
    sample_prefix_time_suffix_latent_flow,
)
from semoflow.tokenization.decoding.codebooks import (
    MAX_STANDALONE_CODEBOOK_ARTIFACT_BYTES,
    SIDCodebookBundle,
    codebook_bundle_metadata,
    extract_codebook_bundle,
    validate_codebook_compatibility,
)
from semoflow.tokenization.decoding.grounding_index import (
    SemanticIDGroundingIndex,
)

LOGGER = logging.getLogger(__name__)


def load_realizer_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[PrefixTimeSuffixLatentFlowRealizer, SIDCodebookBundle, dict]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    state = checkpoint["model_state_dict"]
    bundle = SIDCodebookBundle(
        codebooks=checkpoint["codebooks"].float().cpu().contiguous(),
        code_fields=tuple(
            str(field) for field in checkpoint["codebook_fields"]
        ),
    )
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
        prefix_plan_codebooks=bundle.codebooks[:prefix_depth],
        city_vocab=int(config.get("city_vocab", 0)),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, bundle, checkpoint


def _catalog_city_ids(
    lookup_path: Path,
    places_path: Path,
    catalog_codes: torch.Tensor,
) -> torch.Tensor:
    width = int(catalog_codes.shape[-1])
    columns = [f"sem_code_{level}" for level in range(width)]
    lookup = pl.read_parquet(lookup_path, columns=["placekey_id", *columns])
    places = pl.read_parquet(places_path, columns=["placekey_id", "city_id"])
    city_of_poi = {
        int(row["placekey_id"]): int(row["city_id"])
        for row in places.iter_rows(named=True)
        if row["city_id"] is not None
    }
    first: dict[tuple[int, ...], int] = {}
    for row in lookup.iter_rows(named=True):
        key = tuple(int(row[column]) for column in columns)
        if key in first:
            continue
        city = city_of_poi.get(int(row["placekey_id"]))
        if city is not None:
            first[key] = city
    missing = 0
    cities = []
    for code_row in catalog_codes.tolist():
        city = first.get(tuple(int(v) for v in code_row))
        if city is None:
            missing += 1
            cities.append(-1)
        else:
            cities.append(city)
    if missing:
        LOGGER.warning(
            "%d catalog rows have no city and cannot satisfy the constraint",
            missing,
        )
    return torch.tensor(cities, dtype=torch.long)


def _catalog_supported_events(
    catalog: PrefixTailCatalog,
    prefix_codes: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.BoolTensor:
    powers = torch.tensor(
        [
            catalog.codebook_size**power
            for power in range(catalog.prefix_depth - 1, -1, -1)
        ],
        dtype=torch.long,
    )
    keys = (prefix_codes[..., : catalog.prefix_depth].long() * powers).sum(
        dim=-1
    )
    catalog_keys = catalog.prefix_keys.cpu().long()
    positions = torch.searchsorted(catalog_keys, keys.reshape(-1).cpu())
    in_range = positions < catalog_keys.numel()
    safe_positions = positions.clamp_max(max(catalog_keys.numel() - 1, 0))
    supported = in_range & (
        catalog_keys[safe_positions] == keys.reshape(-1).cpu()
    )
    supported = supported.view_as(valid_mask)
    return valid_mask.bool().cpu() & supported


def _sample(
    model: PrefixTimeSuffixLatentFlowRealizer,
    prefix: torch.Tensor,
    times: torch.Tensor,
    valid: torch.BoolTensor,
    *,
    catalog: PrefixTailCatalog,
    tail_mean: torch.Tensor,
    tail_whitener: torch.Tensor,
    catalog_prior_weight: float,
    catalog_temperature: float,
    batch_size: int,
    device: torch.device,
    num_steps: int,
    solver: str,
    seed: int,
    city_id: torch.Tensor | None = None,
    catalog_noise_coupling: str = "independent",
) -> PrefixTimeSuffixLatentFlowSample:
    input_valid = valid.bool().cpu()
    empty_rows = ~input_valid.any(dim=1)
    sampling_prefix = prefix
    sampling_times = times
    sampling_valid = input_valid
    if empty_rows.any():
        sampling_prefix = prefix.clone()
        sampling_times = times.clone()
        sampling_valid = input_valid.clone()
        fallback_prefix = catalog.codes[0, : model.prefix_depth].cpu()
        sampling_prefix[empty_rows, 0] = fallback_prefix
        sampling_times[empty_rows, 0] = 0
        sampling_valid[empty_rows, 0] = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sampled = sample_prefix_time_suffix_latent_flow(
        model,
        sampling_prefix.to(device),
        sampling_times.to(device),
        sampling_valid.to(device),
        catalog=catalog,
        tail_mean=tail_mean,
        tail_whitener=tail_whitener,
        catalog_prior_weight=catalog_prior_weight,
        catalog_temperature=catalog_temperature,
        num_steps=num_steps,
        solver=solver,
        device=device,
        generator=torch.Generator(device=device).manual_seed(seed),
        max_batch_size=batch_size,
        city_id=None if city_id is None else city_id.to(device),
        catalog_noise_coupling=catalog_noise_coupling,
    )
    sampled_full_sids = sampled.full_sids.cpu()
    sampled_full_sids[empty_rows] = 0
    sampled_cpu = dataclasses.replace(sampled, full_sids=sampled_full_sids)
    if device.type == "cuda":
        LOGGER.info(
            "latent-flow arm peak CUDA memory allocated=%.2f GiB reserved=%.2f GiB",
            torch.cuda.max_memory_allocated(device) / 2**30,
            torch.cuda.max_memory_reserved(device) / 2**30,
        )
    del sampled
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sampled_cpu


@dataclass(frozen=True)
class _Arm:
    """One conditioning source for the realizer: gold or sampled prefixes."""

    name: str
    prefix: torch.Tensor
    times: torch.Tensor
    input_valid: torch.Tensor
    supported: torch.Tensor
    seed: int
    city_id: torch.Tensor | None


def _catalog_support(arms: Sequence[_Arm]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for arm in arms:
        valid, supported = arm.input_valid, arm.supported
        counts[f"{arm.name}_input_events"] = int(valid.sum().item())
        counts[f"{arm.name}_supported_events"] = int(supported.sum().item())
        counts[f"{arm.name}_unsupported_events"] = int(
            (valid & ~supported).sum().item()
        )
        counts[f"{arm.name}_rows_without_support"] = int(
            (~supported.any(dim=1)).sum().item()
        )
    return counts


@dataclass(frozen=True)
class _Decoding:
    tail_mean: torch.Tensor
    tail_whitener: torch.Tensor
    prior_weight: float
    checkpoint_temperature: float
    temperature: float

    def report(self) -> dict[str, Any]:
        return {
            "calibrated_legal_catalog_posterior": True,
            "catalog_prior_weight": self.prior_weight,
            "catalog_temperature": self.temperature,
            "checkpoint_catalog_temperature": self.checkpoint_temperature,
        }


def _decoding_settings(
    checkpoint: Mapping[str, Any],
    temperature_override: float | None,
    device: torch.device,
) -> _Decoding:
    config = checkpoint.get("config", {})
    checkpoint_temperature = float(config.get("catalog_temperature", 1.0))
    temperature = (
        checkpoint_temperature
        if temperature_override is None
        else float(temperature_override)
    )
    if temperature <= 0.0:
        raise ValueError("catalog temperature must be positive")

    def _on_device(key: str) -> torch.Tensor:
        return checkpoint[key].float().to(device)

    return _Decoding(
        tail_mean=_on_device("tail_mean"),
        tail_whitener=_on_device("tail_whitener"),
        prior_weight=float(config.get("catalog_prior_weight", 0.25)),
        checkpoint_temperature=checkpoint_temperature,
        temperature=temperature,
    )


def _check_options(
    *,
    max_samples: int,
    batch_size: int,
    reference_split: str,
    catalog_noise_coupling: str,
) -> None:
    if max_samples < 1 or batch_size < 1:
        raise ValueError("max_samples and batch_size must be positive")
    if catalog_noise_coupling not in CATALOG_NOISE_COUPLINGS:
        raise ValueError(
            f"catalog_noise_coupling must be one of {CATALOG_NOISE_COUPLINGS}"
        )
    if reference_split not in {"val", "test"}:
        raise ValueError("reference_split must be 'val' or 'test'")


def _generation_catalog(
    model: PrefixTimeSuffixLatentFlowRealizer,
    payload: dict[str, Any],
    *,
    bank_path: Path,
    lookup_path: Path,
    places_path: Path,
    codebooks_path: Path,
    expected_bundle: SIDCodebookBundle,
) -> tuple[PrefixTailCatalog, SIDCodebookBundle]:
    """The legal SID catalog generation decodes into, with its train-split
    log-count prior, and its codebooks."""
    catalog_codes = torch.from_numpy(load_lookup(lookup_path)).long()
    offset, count = bank_split_window(payload, "train")
    rows = slice(offset, offset + count)
    observed_codes = payload["sid_codes"][rows][
        payload["valid_mask"][rows].bool()
    ]
    bundle = extract_codebook_bundle(
        codebooks_path,
        max_artifact_bytes=MAX_STANDALONE_CODEBOOK_ARTIFACT_BYTES,
    )
    validate_codebook_compatibility(
        expected_bundle.codebooks,
        expected_bundle.code_fields,
        bundle,
    )
    catalog_city = (
        _catalog_city_ids(lookup_path, places_path, catalog_codes)
        if model.city_vocab > 0
        else None
    )
    catalog = PrefixTailCatalog.from_codes(
        catalog_codes,
        bundle.codebooks,
        prefix_depth=model.prefix_depth,
        observed_codes=observed_codes,
        candidate_city=catalog_city,
    )
    return catalog, bundle


def _conditioning_arms(
    model: PrefixTimeSuffixLatentFlowRealizer,
    catalog: PrefixTailCatalog,
    payload: dict[str, Any],
    *,
    real: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    sampled: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    sampled_city: torch.Tensor | None,
    bank_path: Path,
    reference_split: str,
) -> tuple[_Arm, _Arm]:
    """Gold reference prefixes and sampled plans, with their conditioning.

    ``real`` and ``sampled`` are ``(codes, time_tokens, valid_mask)``.
    """
    real_codes, real_times, real_valid = real
    sampled_prefix, sampled_times, sampled_valid = sampled
    gold_prefix = real_codes[..., : model.prefix_depth]
    gold_city = None
    if model.city_vocab > 0:
        if sampled_city is None:
            raise ValueError(
                "planner samples carry no city_ids; a city-conditioned "
                "realizer cannot be evaluated on them"
            )
        gold_city = bank_split_city_ids(
            payload, reference_split, rows=int(real_valid.shape[0])
        )
    gold = _Arm(
        name="gold_prefix",
        prefix=gold_prefix,
        times=real_times,
        input_valid=real_valid,
        supported=_catalog_supported_events(catalog, gold_prefix, real_valid),
        seed=7000,
        city_id=gold_city,
    )
    planned = _Arm(
        name="sampled_plan",
        prefix=sampled_prefix,
        times=sampled_times,
        input_valid=sampled_valid,
        supported=_catalog_supported_events(
            catalog, sampled_prefix, sampled_valid
        ),
        seed=17000,
        city_id=sampled_city,
    )
    return gold, planned


def _city_constraint_report(
    model: PrefixTimeSuffixLatentFlowRealizer,
    samples: Mapping[str, PrefixTimeSuffixLatentFlowSample],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "enabled": any(
            sample.city_constrained_events for sample in samples.values()
        ),
        "requested": model.city_vocab > 0,
    }
    for name, sample in samples.items():
        report[f"{name}_constrained_events"] = sample.city_constrained_events
        report[f"{name}_fallback_events"] = (
            sample.city_constraint_fallback_events
        )
    return report


def evaluate(
    *,
    checkpoint_path: Path,
    bank_path: Path,
    planner_samples: Mapping[str, Any],
    codebooks_path: Path,
    lookup_path: Path,
    grounding_path: Path,
    places_path: Path,
    output_path: Path,
    max_samples: int,
    batch_size: int,
    device: str,
    num_steps: int,
    solver: str,
    reference_split: str = "val",
    candidate_selection: str = "first",
    candidate_prior_path: Path | None = None,
    catalog_temperature_override: float | None = None,
    catalog_noise_coupling: str = "independent",
) -> dict[str, Any]:
    _check_options(
        max_samples=max_samples,
        batch_size=batch_size,
        reference_split=reference_split,
        catalog_noise_coupling=catalog_noise_coupling,
    )
    candidate_weights: Mapping[int, float] | None = (
        load_candidate_prior(candidate_prior_path)
        if candidate_prior_path is not None
        else None
    )
    if candidate_selection != "first" and candidate_weights is None:
        raise ValueError("--candidate-selection requires --candidate-prior")
    resolved_device = torch.device(device)
    model, expected_bundle, checkpoint = load_realizer_checkpoint(
        checkpoint_path, resolved_device
    )
    decoding = _decoding_settings(
        checkpoint, catalog_temperature_override, resolved_device
    )
    payload = load_bank_payload(bank_path)
    real_codes, real_times, real_valid, real_placekey_ids = load_reference(
        payload, split=reference_split, max_samples=max_samples
    )
    plan = plan_from_planner_samples(
        planner_samples,
        width=model.max_seq_len,
        prefix_depth=model.prefix_depth,
        max_rows=max_samples,
    )
    n_rows = min(real_codes.size(0), plan.prefix.size(0))
    real_codes = real_codes[:n_rows]
    real_times = real_times[:n_rows]
    real_valid = real_valid[:n_rows]
    real_placekey_ids = real_placekey_ids[:n_rows]
    real_hours = aligned_reference_event_hours(
        payload, split=reference_split, valid_mask=real_valid
    )
    sampled_hours = plan.event_hours[:n_rows]
    catalog, bundle = _generation_catalog(
        model,
        payload,
        bank_path=bank_path,
        lookup_path=lookup_path,
        places_path=places_path,
        codebooks_path=codebooks_path,
        expected_bundle=expected_bundle,
    )
    grounding = SemanticIDGroundingIndex.load(grounding_path)
    coordinates, geohashes, placekey_vocab_size = load_places(places_path)

    gold, planned = _conditioning_arms(
        model,
        catalog,
        payload,
        real=(real_codes, real_times, real_valid),
        sampled=(
            plan.prefix[:n_rows],
            plan.time_tokens[:n_rows],
            plan.valid[:n_rows],
        ),
        sampled_city=None if plan.city_id is None else plan.city_id[:n_rows],
        bank_path=bank_path,
        reference_split=reference_split,
    )
    arms = (gold, planned)

    sample_arm = functools.partial(
        _sample,
        model,
        catalog=catalog,
        tail_mean=decoding.tail_mean,
        tail_whitener=decoding.tail_whitener,
        catalog_prior_weight=decoding.prior_weight,
        catalog_temperature=decoding.temperature,
        batch_size=batch_size,
        device=resolved_device,
        num_steps=num_steps,
        solver=solver,
        catalog_noise_coupling=catalog_noise_coupling,
    )
    samples = {
        arm.name: sample_arm(
            arm.prefix,
            arm.times,
            arm.supported,
            seed=arm.seed,
            city_id=arm.city_id,
        )
        for arm in arms
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gold_prefix_full_sids": samples[gold.name].full_sids.cpu(),
            "sampled_plan_full_sids": samples[planned.name].full_sids.cpu(),
            "sampled_plan_valid_mask": planned.supported,
            "sampled_plan_input_valid_mask": planned.input_valid,
            "decoder": "calibrated_legal_catalog_posterior",
        },
        output_path.with_name("generated_full_sids.pt"),
    )

    arm_metrics = functools.partial(
        metric_block,
        real_codes,
        real_valid,
        grounding=grounding,
        coordinates=coordinates,
        geohashes=geohashes,
        placekey_vocab_size=placekey_vocab_size,
        reference_placekey_ids=real_placekey_ids,
        real_time_tokens=real_times,
        real_event_hours=real_hours,
        candidate_selection=candidate_selection,
        candidate_weights=candidate_weights,
        compute_device=resolved_device,
    )
    evaluations: dict[str, dict[str, Any]] = {}
    for arm in arms:
        label = arm.name.replace("_", "-")
        started_at = time.perf_counter()
        LOGGER.info(
            "computing %s metrics on %s rows using %s",
            label,
            len(real_codes),
            resolved_device,
        )
        evaluations[arm.name] = arm_metrics(
            samples[arm.name].full_sids.cpu(),
            arm.supported,
            generated_time_tokens=arm.times,
            generated_event_hours=(
                real_hours if arm is gold else sampled_hours
            ),
        )
        LOGGER.info(
            "%s metrics complete in %.1fs",
            label,
            time.perf_counter() - started_at,
        )

    result: dict[str, Any] = {
        "model_type": MODEL_TYPE,
        "checkpoint": str(checkpoint_path),
        "reference_bank": str(bank_path),
        "codebooks": str(codebooks_path),
        "codebook_metadata": codebook_bundle_metadata(bundle),
        "lookup": str(lookup_path),
        "grounding_index": str(grounding_path),
        "rows": n_rows,
        "device": str(resolved_device),
        "city_constraint": _city_constraint_report(model, samples),
        "sampling": {"solver": solver, "num_steps": num_steps},
        "candidate_selection": candidate_selection,
        "generation_candidate_sid_count": int(catalog.codes.shape[0]),
        "generation_catalog_support": _catalog_support(arms),
        "candidate_prior": (
            None if candidate_prior_path is None else str(candidate_prior_path)
        ),
        "decoding": {
            **decoding.report(),
            "catalog_noise_coupling": catalog_noise_coupling,
        },
        "metric_protocol": {
            "undefined_metric_value": None,
            "ground_before_poi_metrics": True,
            "reference_poi_source": (
                f"aligned_{reference_split}_segment_placekey_ids"
            ),
            "generated_poi_source": (
                "canonical_first_candidate_per_full_sid"
                if candidate_selection == "first"
                else f"train_prior_candidate_policy:{candidate_selection}"
            ),
        },
        "evaluations": evaluations,
    }
    return result


__all__ = ["evaluate", "load_realizer_checkpoint"]
