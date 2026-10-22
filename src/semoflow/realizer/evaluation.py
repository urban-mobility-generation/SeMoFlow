from __future__ import annotations

import dataclasses
import functools
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl
import torch

from semoflow.metrics.prefix_time_suffix_evaluation import (
    accuracy,
    compute_tail_error_metrics,
    compute_tail_level_error_metrics,
    dynamics,
    load_candidate_prior,
    load_lookup,
    load_places,
    load_reference,
    load_sampled_plan,
    metric_block,
)
from semoflow.realizer.bank import (
    bank_split_city_ids,
    load_bank_payload,
    resolve_bank_split_window,
)
from semoflow.realizer.model import (
    PrefixTimeSuffixLatentFlowRealizer,
    model_type,
)
from semoflow.realizer.sampling import (
    PrefixTailCatalog,
    PrefixTimeSuffixLatentFlowSample,
    sample_prefix_time_suffix_latent_flow,
)
from semoflow.realizer.spatial import (
    prefix_centroid_table,
    prefix_hop_distance_km,
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
from semoflow.training.utils.checkpoint import (
    load_checkpoint,
)

LOGGER = logging.getLogger(__name__)


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    PrefixTimeSuffixLatentFlowRealizer,
    torch.Tensor,
    torch.Tensor,
    SIDCodebookBundle,
    dict[str, Any],
]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint is missing the model config")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing model_state_dict")
    position = state.get("event_position_embedding.weight")
    if not isinstance(position, torch.Tensor):
        raise ValueError("checkpoint is missing event position embeddings")
    tail_mean = checkpoint.get("tail_mean")
    tail_std = checkpoint.get("tail_std")
    if not isinstance(tail_mean, torch.Tensor) or not isinstance(
        tail_std, torch.Tensor
    ):
        raise ValueError("checkpoint is missing tail normalization statistics")
    checkpoint_codebooks = checkpoint.get("codebooks")
    checkpoint_fields = checkpoint.get("codebook_fields")
    if not isinstance(checkpoint_codebooks, torch.Tensor):
        raise ValueError(
            "checkpoint is missing the embedded training codebooks; "
            "retrain the realizer with the compatible artifact contract"
        )
    if not isinstance(checkpoint_fields, (list, tuple)):
        raise ValueError("checkpoint is missing codebook_fields")
    expected_bundle = SIDCodebookBundle(
        codebooks=checkpoint_codebooks.float().cpu().contiguous(),
        code_fields=tuple(str(field) for field in checkpoint_fields),
    )
    expected_shape = (
        int(checkpoint["num_codebooks"]),
        int(checkpoint["codebook_size"]),
        int(checkpoint["latent_dim"]),
    )
    if tuple(expected_bundle.codebooks.shape) != expected_shape:
        raise ValueError(
            "checkpoint codebook shape does not match its model metadata: "
            f"{tuple(expected_bundle.codebooks.shape)} != {expected_shape}"
        )
    stored_metadata = checkpoint.get("codebook_metadata")
    if stored_metadata != codebook_bundle_metadata(expected_bundle):
        raise ValueError("checkpoint codebook metadata is inconsistent")
    prefix_depth = int(checkpoint["prefix_depth"])
    prefix_embedding_mode = str(config.get("prefix_embedding_mode", "learned"))
    tail_representation = str(config.get("tail_representation", "summed"))
    catalog_decoder_mode = str(config.get("catalog_decoder_mode", "nearest"))
    latent_target_scope = str(config.get("latent_target_scope", "suffix"))
    prefix_plan_codebooks = None
    if prefix_embedding_mode == "hybrid_frozen_codebook":
        prefix_plan_codebooks = expected_bundle.codebooks[:prefix_depth]
    model = PrefixTimeSuffixLatentFlowRealizer(
        max_seq_len=int(position.size(0)),
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
        prefix_embedding_mode=prefix_embedding_mode,
        prefix_plan_codebooks=prefix_plan_codebooks,
        tail_representation=tail_representation,
        latent_target_scope=latent_target_scope,
        catalog_decoder_mode=catalog_decoder_mode,
        city_vocab=int(config.get("city_vocab", 0)),
        spatial_conditioning=bool(config.get("spatial_conditioning", False)),
        spatial_num_basis=int(config.get("spatial_num_basis", 8)),
        spatial_max_km=float(config.get("spatial_max_km", 50.0)),
    ).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    return (
        model,
        tail_mean.float().to(device),
        tail_std.float().to(device),
        expected_bundle,
        checkpoint,
    )


def _catalog_city_ids(
    lookup_path: Path,
    places_path: Path,
    catalog_codes: torch.Tensor,
    *,
    allowed_place_ids: set[int] | None = None,
) -> torch.Tensor:
    width = int(catalog_codes.shape[-1])
    columns = [f"sem_code_{level}" for level in range(width)]
    lookup = pl.read_parquet(lookup_path, columns=["placekey_id", *columns])
    places = pl.read_parquet(places_path, columns=["placekey_id", "city_id"])
    city_of_poi = {
        int(row["placekey_id"]): int(row["city_id"])
        for row in places.iter_rows(named=True)
        if row["city_id"] is not None
        and (
            allowed_place_ids is None
            or int(row["placekey_id"]) in allowed_place_ids
        )
    }
    first: dict[tuple[int, ...], int] = {}
    for row in lookup.iter_rows(named=True):
        key = tuple(int(row[column]) for column in columns)
        if key in first:
            continue
        place_id = int(row["placekey_id"])
        if allowed_place_ids is not None and place_id not in allowed_place_ids:
            continue
        city = city_of_poi.get(place_id)
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


def _train_catalog_grounding(
    grounding: SemanticIDGroundingIndex,
    train_place_ids: set[int],
) -> SemanticIDGroundingIndex:
    mapping: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for key, payloads in grounding.iter_key_payload_groups():
        supported = [
            dict(payload)
            for payload in payloads
            if payload.get("placekey_id") is not None
            and int(payload["placekey_id"]) in train_place_ids
        ]
        if supported:
            mapping[tuple(int(value) for value in key)] = supported
    if not mapping:
        raise ValueError("train-only grounding has no supported POIs")
    return SemanticIDGroundingIndex(
        key_columns=grounding.key_columns,
        id_column=grounding.id_column,
        payload_columns=grounding.payload_columns,
        order_agnostic=grounding.order_agnostic,
        mapping=mapping,
    )


def _catalog_supported_events(
    catalog: PrefixTailCatalog,
    prefix_codes: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.BoolTensor:
    if prefix_codes.shape[:2] != valid_mask.shape:
        raise ValueError("prefix codes and valid mask are misaligned")
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


def _sampled_plan_city_ids(
    sampled_plan_path: Path, expected_rows: int
) -> torch.Tensor:
    payload = torch.load(
        sampled_plan_path, map_location="cpu", weights_only=False
    )
    city = payload.get("city_id")
    if city is None:
        raise ValueError(
            f"{sampled_plan_path} carries no city_id; a city-conditioned "
            "checkpoint cannot be evaluated on it"
        )
    city = torch.as_tensor(city).long()
    if city.ndim != 1 or int(city.shape[0]) < expected_rows:
        raise ValueError("sampled plan city_id does not align with its rows")
    return city[:expected_rows]


def _sample(
    model: PrefixTimeSuffixLatentFlowRealizer,
    prefix: torch.Tensor,
    times: torch.Tensor,
    valid: torch.BoolTensor,
    *,
    catalog: PrefixTailCatalog,
    tail_mean: torch.Tensor,
    tail_std: torch.Tensor,
    tail_whitener: torch.Tensor | None,
    tail_dewhitener: torch.Tensor | None,
    catalog_prior_weight: float,
    catalog_temperature: float,
    global_candidate_count: int,
    batch_size: int,
    device: torch.device,
    num_steps: int,
    solver: str,
    seed: int,
    city_id: torch.Tensor | None = None,
    hop_distance_km: torch.Tensor | None = None,
    trace_indices: torch.LongTensor | None = None,
) -> PrefixTimeSuffixLatentFlowSample:
    input_valid = valid.bool().cpu()
    empty_rows = ~input_valid.any(dim=1)
    sampling_prefix = prefix
    sampling_times = times
    sampling_valid = input_valid
    if empty_rows.any():
        if trace_indices is not None and bool(empty_rows[trace_indices].any()):
            raise ValueError(
                "flow traces cannot include rows with no train-catalog support"
            )
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
        tail_std=tail_std,
        tail_whitener=tail_whitener,
        tail_dewhitener=tail_dewhitener,
        catalog_prior_weight=catalog_prior_weight,
        catalog_temperature=catalog_temperature,
        global_candidate_count=global_candidate_count,
        num_steps=num_steps,
        solver=solver,
        device=device,
        generator=torch.Generator(device=device).manual_seed(seed),
        max_batch_size=batch_size,
        city_id=None if city_id is None else city_id.to(device),
        hop_distance_km=(
            None if hop_distance_km is None else hop_distance_km.to(device)
        ),
        trace_indices=(
            None if trace_indices is None else trace_indices.to(device)
        ),
    )
    sampled_full_sids = sampled.full_sids.cpu()
    sampled_tail_embeddings = sampled.tail_embeddings.cpu()
    if empty_rows.any():
        sampled_full_sids[empty_rows] = 0
        sampled_tail_embeddings[empty_rows] = 0
    sampled_cpu = dataclasses.replace(
        sampled,
        full_sids=sampled_full_sids,
        tail_embeddings=sampled_tail_embeddings,
        valid_mask=input_valid,
        tail_level_embeddings=(
            None
            if sampled.tail_level_embeddings is None
            else sampled.tail_level_embeddings.cpu()
        ),
    )
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


def _catalog_support(
    gold_valid: torch.Tensor,
    gold_supported: torch.Tensor,
    sampled_valid: torch.Tensor,
    sampled_supported: torch.Tensor,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for arm, valid, supported in (
        ("gold_prefix", gold_valid, gold_supported),
        ("sampled_plan", sampled_valid, sampled_supported),
    ):
        counts[f"{arm}_input_events"] = int(valid.sum().item())
        counts[f"{arm}_supported_events"] = int(supported.sum().item())
        counts[f"{arm}_unsupported_events"] = int(
            (valid & ~supported).sum().item()
        )
        counts[f"{arm}_rows_without_support"] = int(
            (~supported.any(dim=1)).sum().item()
        )
    return counts


def _write_flow_trace(
    path: Path,
    sample: PrefixTimeSuffixLatentFlowSample,
    trace_rows: torch.Tensor,
    *,
    model: PrefixTimeSuffixLatentFlowRealizer,
    prefix: torch.Tensor,
    times: torch.Tensor,
    valid: torch.Tensor,
    solver: str,
    num_steps: int,
    seed: int,
    checkpoint_path: Path,
    sampled_plan_path: Path,
) -> None:
    if sample.normalized_flow_path is None or sample.flow_times is None:
        raise RuntimeError(
            "latent-flow sampler did not return the requested trace"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "artifact_type": "sampled_plan_latent_flow_path",
            "normalized_flow_path": sample.normalized_flow_path,
            "flow_times": sample.flow_times,
            "trace_indices": trace_rows,
            "prefix_codes": prefix.index_select(0, trace_rows).cpu(),
            "time_tokens": times.index_select(0, trace_rows).cpu(),
            "valid_mask": valid.index_select(0, trace_rows).cpu(),
            "full_sids": sample.full_sids.index_select(
                0, trace_rows.to(sample.full_sids.device)
            ).cpu(),
            "tail_representation": model.tail_representation,
            "flow_latent_dim": model.flow_latent_dim,
            "prefix_depth": model.prefix_depth,
            "solver": solver,
            "num_steps": num_steps,
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "sampled_plan": str(sampled_plan_path),
        },
        path,
    )


def evaluate(
    *,
    checkpoint_path: Path,
    bank_path: Path,
    sampled_plan_path: Path,
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
    catalog_prior_source: str = "target_train",
    generation_catalog_scope: str = "full_city",
    flow_trace_rows: int = 0,
    flow_path_output: Path | None = None,
    brief_metrics: bool = False,
) -> dict[str, Any]:
    if max_samples < 1 or batch_size < 1:
        raise ValueError("max_samples and batch_size must be positive")
    if reference_split not in {"val", "test"}:
        raise ValueError("reference_split must be 'val' or 'test'")
    if flow_trace_rows < 0:
        raise ValueError("flow_trace_rows must be non-negative")
    if flow_path_output is not None and flow_trace_rows == 0:
        raise ValueError("flow_path_output requires flow_trace_rows > 0")
    if catalog_prior_source not in {"target_train", "uniform"}:
        raise ValueError(
            "catalog_prior_source must be 'target_train' or 'uniform'"
        )
    if generation_catalog_scope not in {"full_city", "train_observed"}:
        raise ValueError(
            "generation_catalog_scope must be 'full_city' or 'train_observed'"
        )
    if generation_catalog_scope == "train_observed" and (
        candidate_prior_path is None or catalog_prior_source != "target_train"
    ):
        raise ValueError(
            "train_observed generation requires the train candidate prior "
            "and target_train catalog prior source"
        )
    candidate_weights: Mapping[int, float] | None = (
        load_candidate_prior(candidate_prior_path)
        if candidate_prior_path is not None
        else None
    )
    if candidate_selection != "first" and candidate_weights is None:
        raise ValueError("--candidate-selection requires --candidate-prior")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    model, tail_mean, tail_std, expected_bundle, checkpoint = _load_model(
        checkpoint_path, resolved_device
    )
    tail_whitener = checkpoint.get("tail_whitener")
    tail_dewhitener = checkpoint.get("tail_dewhitener")
    if tail_whitener is not None:
        tail_whitener = tail_whitener.float().to(resolved_device)
    if tail_dewhitener is not None:
        tail_dewhitener = tail_dewhitener.float().to(resolved_device)
    checkpoint_config = checkpoint.get("config", {})
    if not isinstance(checkpoint_config, dict):
        raise ValueError("checkpoint config must be a dictionary")
    catalog_prior_weight = float(
        checkpoint_config.get("catalog_prior_weight", 0.25)
    )
    global_candidate_count = int(
        checkpoint_config.get("catalog_candidate_count", 64)
    )
    checkpoint_catalog_temperature = float(
        checkpoint_config.get("catalog_temperature", 1.0)
    )
    catalog_temperature = (
        checkpoint_catalog_temperature
        if catalog_temperature_override is None
        else float(catalog_temperature_override)
    )
    if catalog_temperature <= 0.0:
        raise ValueError("catalog temperature must be positive")
    if (
        catalog_temperature_override is not None
        and model.catalog_decoder_mode != "calibrated"
    ):
        raise ValueError(
            "--catalog-temperature applies only to the calibrated "
            "legal-catalog posterior decoder"
        )
    payload = load_bank_payload(bank_path)
    real_codes, real_times, real_valid, real_placekey_ids = load_reference(
        bank_path,
        max_samples=max_samples,
        split=reference_split,
        payload=payload,
    )
    sampled_prefix, sampled_times, sampled_valid, sampled_protocol = (
        load_sampled_plan(
            sampled_plan_path,
            max_samples=max_samples,
            expected_width=model.max_seq_len,
            prefix_depth=model.prefix_depth,
        )
    )
    n_rows = min(real_codes.size(0), sampled_prefix.size(0))
    real_codes = real_codes[:n_rows]
    real_times = real_times[:n_rows]
    real_valid = real_valid[:n_rows]
    real_placekey_ids = real_placekey_ids[:n_rows]
    sampled_prefix = sampled_prefix[:n_rows]
    sampled_times = sampled_times[:n_rows]
    sampled_valid = sampled_valid[:n_rows]
    flow_trace_indices = (
        torch.arange(min(flow_trace_rows, n_rows), dtype=torch.long)
        if flow_trace_rows
        else None
    )

    catalog_codes = torch.from_numpy(load_lookup(lookup_path)).long()
    observed_codes = None
    if catalog_prior_source == "target_train":
        train_offset, train_count = resolve_bank_split_window(
            payload, "train", path=bank_path
        )
        train_codes = payload["sid_codes"][
            train_offset : train_offset + train_count
        ]
        train_valid = payload["valid_mask"][
            train_offset : train_offset + train_count
        ].bool()
        observed_codes = train_codes[train_valid]
    train_place_ids = (
        set(int(value) for value in candidate_weights)
        if generation_catalog_scope == "train_observed"
        and candidate_weights is not None
        else None
    )
    if generation_catalog_scope == "train_observed":
        if observed_codes is None or observed_codes.numel() == 0:
            raise ValueError("train-observed catalog requires train SID rows")
        catalog_codes = torch.unique(observed_codes.long(), dim=0)
    bundle = extract_codebook_bundle(
        codebooks_path,
        max_artifact_bytes=MAX_STANDALONE_CODEBOOK_ARTIFACT_BYTES,
    )
    validate_codebook_compatibility(
        expected_bundle.codebooks,
        expected_bundle.code_fields,
        bundle,
    )
    catalog_city = None
    if getattr(model, "city_vocab", 0) > 0:
        catalog_city = _catalog_city_ids(
            lookup_path,
            places_path,
            catalog_codes,
            allowed_place_ids=train_place_ids,
        )
    catalog = PrefixTailCatalog.from_codes(
        catalog_codes,
        bundle.codebooks,
        prefix_depth=model.prefix_depth,
        observed_codes=observed_codes,
        candidate_city=catalog_city,
        target_level_start=model.target_level_start,
    )
    gold_supported_valid = _catalog_supported_events(
        catalog,
        real_codes[..., : model.prefix_depth],
        real_valid,
    )
    sampled_supported_valid = _catalog_supported_events(
        catalog,
        sampled_prefix,
        sampled_valid,
    )
    catalog_support = _catalog_support(
        real_valid,
        gold_supported_valid,
        sampled_valid,
        sampled_supported_valid,
    )
    grounding = SemanticIDGroundingIndex.load(grounding_path)
    generated_grounding = (
        _train_catalog_grounding(grounding, train_place_ids)
        if train_place_ids is not None
        else grounding
    )
    coordinates, geohashes, placekey_vocab_size = load_places(places_path)

    gold_city = None
    sampled_city = None
    if getattr(model, "city_vocab", 0) > 0:
        gold_city = bank_split_city_ids(
            payload,
            bank_path,
            split=reference_split,
            valid_mask=real_valid,
            time_tokens=real_times,
        )
        sampled_city = _sampled_plan_city_ids(
            sampled_plan_path, sampled_prefix.shape[0]
        )

    gold_hop = None
    sampled_hop = None
    if getattr(model, "spatial_conditioning", False):
        assert places_path is not None
        assert lookup_path is not None
        centroids = prefix_centroid_table(
            lookup_path,
            places_path,
            real_codes.shape[-1],
            model.prefix_depth,
        )
        gold_hop = prefix_hop_distance_km(
            real_codes[..., : model.prefix_depth], real_valid, centroids
        )
        sampled_hop = prefix_hop_distance_km(
            sampled_prefix, sampled_valid, centroids
        )

    sample_arm = functools.partial(
        _sample,
        model,
        catalog=catalog,
        tail_mean=tail_mean,
        tail_std=tail_std,
        tail_whitener=tail_whitener,
        tail_dewhitener=tail_dewhitener,
        catalog_prior_weight=catalog_prior_weight,
        catalog_temperature=catalog_temperature,
        global_candidate_count=global_candidate_count,
        batch_size=batch_size,
        device=resolved_device,
        num_steps=num_steps,
        solver=solver,
    )
    gold_sample = sample_arm(
        real_codes[..., : model.prefix_depth],
        real_times,
        gold_supported_valid,
        seed=7000,
        city_id=gold_city,
        hop_distance_km=gold_hop,
    )
    sampled_seed = 17000
    sampled_sample = sample_arm(
        sampled_prefix,
        sampled_times,
        sampled_supported_valid,
        seed=sampled_seed,
        city_id=sampled_city,
        hop_distance_km=sampled_hop,
        trace_indices=flow_trace_indices,
    )

    flow_path = None
    if flow_trace_indices is not None:
        flow_path = flow_path_output or output_path.with_name(
            "sampled_plan_flow_path.pt"
        )
        _write_flow_trace(
            flow_path,
            sampled_sample,
            flow_trace_indices,
            model=model,
            prefix=sampled_prefix,
            times=sampled_times,
            valid=sampled_valid,
            solver=solver,
            num_steps=num_steps,
            seed=sampled_seed,
            checkpoint_path=checkpoint_path,
            sampled_plan_path=sampled_plan_path,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gold_prefix_full_sids": gold_sample.full_sids.cpu(),
            "sampled_plan_full_sids": sampled_sample.full_sids.cpu(),
            "sampled_plan_valid_mask": sampled_supported_valid,
            "sampled_plan_input_valid_mask": sampled_valid,
            "decoder": (
                "calibrated_legal_catalog_posterior"
                if model.catalog_decoder_mode == "calibrated"
                else "nearest_legal_catalog_tail_embedding"
            ),
        },
        output_path.with_name("generated_full_sids.pt"),
    )

    arm_metrics = functools.partial(
        metric_block,
        real_codes,
        real_valid,
        grounding=grounding,
        generated_grounding=generated_grounding,
        coordinates=coordinates,
        geohashes=geohashes,
        placekey_vocab_size=placekey_vocab_size,
        reference_placekey_ids=real_placekey_ids,
        motif_day_keys_real=real_times // 6,
        candidate_selection=candidate_selection,
        candidate_weights=candidate_weights,
        compute_device=resolved_device,
        brief_metrics=brief_metrics,
    )
    evaluations: dict[str, dict[str, Any]] = {}
    for arm, sample, supported, arm_times in (
        ("gold_prefix", gold_sample, gold_supported_valid, real_times),
        (
            "sampled_plan",
            sampled_sample,
            sampled_supported_valid,
            sampled_times,
        ),
    ):
        started_at = time.perf_counter()
        LOGGER.info(
            "computing %s metrics on %s rows using %s",
            arm.replace("_", "-"),
            len(real_codes),
            resolved_device,
        )
        evaluations[arm] = arm_metrics(
            sample.full_sids.cpu(),
            supported,
            motif_day_keys_generated=arm_times // 6,
        )
        LOGGER.info(
            "%s metrics complete in %.1fs",
            arm.replace("_", "-"),
            time.perf_counter() - started_at,
        )
    gold_metrics = evaluations["gold_prefix"]
    sampled_metrics = evaluations["sampled_plan"]
    if not brief_metrics:
        gold_metrics["accuracy"] = accuracy(
            gold_sample.full_sids.cpu(),
            real_codes,
            gold_supported_valid,
            model.prefix_depth,
        )
        sampled_metrics["accuracy"] = None
        if gold_sample.tail_level_embeddings is None:
            gold_metrics["tail_error"] = compute_tail_error_metrics(
                gold_sample.tail_embeddings,
                real_codes,
                gold_supported_valid,
                bundle.codebooks,
                tail_mean,
                tail_std,
                tail_whitener=tail_whitener,
                prefix_depth=model.target_level_start,
            )
        else:
            gold_metrics["tail_error"] = compute_tail_level_error_metrics(
                gold_sample.tail_level_embeddings,
                real_codes,
                gold_supported_valid,
                bundle.codebooks,
                tail_mean,
                tail_std,
                prefix_depth=model.prefix_depth,
            )
        sampled_metrics["tail_error"] = None
        for metrics, generated, valid, input_events in (
            (
                gold_metrics,
                gold_sample,
                gold_supported_valid,
                catalog_support["gold_prefix_input_events"],
            ),
            (
                sampled_metrics,
                sampled_sample,
                sampled_supported_valid,
                catalog_support["sampled_plan_input_events"],
            ),
        ):
            metrics["sampling_dynamics"] = dynamics(
                generated.full_sids.cpu(),
                valid,
                generated_grounding,
            )
            metrics["sampling_dynamics"]["catalog_validity_rate"] = float(
                valid.sum().item() / max(int(input_events), 1)
            )

    result: dict[str, Any] = {
        "model_type": model_type(model.latent_target_scope),
        "model_variant": {
            "prefix_embedding_mode": model.prefix_embedding_mode,
            "tail_representation": model.tail_representation,
            "latent_target_scope": model.latent_target_scope,
            "catalog_decoder_mode": model.catalog_decoder_mode,
            "tail_normalization": checkpoint.get(
                "tail_normalization", "diagonal"
            ),
        },
        "checkpoint": str(checkpoint_path),
        "reference_bank": str(bank_path),
        "sampled_plan": str(sampled_plan_path),
        "codebooks": str(codebooks_path),
        "codebook_metadata": codebook_bundle_metadata(bundle),
        "lookup": str(lookup_path),
        "grounding_index": str(grounding_path),
        "rows": n_rows,
        "device": str(resolved_device),
        "city_constraint": {
            "enabled": bool(
                gold_sample.city_constrained_events
                or sampled_sample.city_constrained_events
            ),
            "requested": getattr(model, "city_vocab", 0) > 0,
            "gold_prefix_constrained_events": (
                gold_sample.city_constrained_events
            ),
            "gold_prefix_fallback_events": (
                gold_sample.city_constraint_fallback_events
            ),
            "sampled_plan_constrained_events": (
                sampled_sample.city_constrained_events
            ),
            "sampled_plan_fallback_events": (
                sampled_sample.city_constraint_fallback_events
            ),
        },
        "sampling": {"solver": solver, "num_steps": num_steps},
        "flow_trace": {
            "enabled": flow_path is not None,
            "rows": 0
            if flow_trace_indices is None
            else int(flow_trace_indices.numel()),
            "path": None if flow_path is None else str(flow_path),
            "state_space": "normalized_continuous_suffix_latent",
            "decoded_only_at": "t=0",
        },
        "candidate_selection": candidate_selection,
        "catalog_prior_source": catalog_prior_source,
        "generation_catalog_scope": generation_catalog_scope,
        "generation_candidate_poi_count": (
            None if train_place_ids is None else len(train_place_ids)
        ),
        "generation_candidate_sid_count": int(catalog.codes.shape[0]),
        "generation_catalog_support": catalog_support,
        "candidate_prior": (
            None if candidate_prior_path is None else str(candidate_prior_path)
        ),
        "decoding": {
            "nearest_legal_catalog_tail_embedding": model.catalog_decoder_mode
            == "nearest",
            "calibrated_legal_catalog_posterior": model.catalog_decoder_mode
            == "calibrated",
            "catalog_prior_weight": catalog_prior_weight,
            "global_candidate_count": global_candidate_count,
            "catalog_temperature": catalog_temperature,
            "checkpoint_catalog_temperature": checkpoint_catalog_temperature,
            "a4_quota_or_injective_assignment": False,
        },
        "metric_protocol": {
            "brief_metrics": brief_metrics,
            "ground_before_poi_metrics": True,
            "reference_poi_source": (
                f"aligned_{reference_split}_segment_placekey_ids"
            ),
            "generated_poi_source": (
                "canonical_first_candidate_per_full_sid"
                if candidate_selection == "first"
                else f"train_prior_candidate_policy:{candidate_selection}"
            ),
            "generated_grounding_scope": generation_catalog_scope,
            "sid_code_metrics_are_secondary_diagnostics": True,
            "gold_prefix_tail_error": "continuous_tail_embedding_before_catalog_decode",
        },
        "sampled_plan_protocol": sampled_protocol,
        "evaluations": evaluations,
        "limitations": [
            "sampled_plan is unpaired with the validation reference; exact target-POI accuracy is not applicable",
            "tail_error and suffix accuracy are reported only for the row-aligned gold-prefix condition",
            "the model receives fixed planner lengths and does not generate an EOS or length distribution",
            "the catalog decoder uses only catalog SIDs whose prefix exactly matches the input prefix",
            "generated ambiguous SIDs use the grounding index canonical first-candidate policy; this is not a learned POI disambiguator",
            "train_observed restricts generation candidates only; existing tokenizer and checkpoints may still have been built using the full city catalog",
            "prefix events without a train-observed legal continuation are excluded from generated events and counted in generation_catalog_support",
        ],
    }
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return result
