from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

from semoflow.realizer.model import MODEL_TYPE
from semoflow.realizer.time_tokens import (
    N_PART_OF_DAY_BINS,
    PART_OF_DAY_START_MINUTES,
)
from semoflow.utils.serialization import normalize_mapping, write_json

logger = logging.getLogger(__name__)


def _require_tensor(payload: Mapping[str, object], key: str) -> torch.Tensor:
    return cast(torch.Tensor, payload[key]).detach().cpu()


def realizer_contract(checkpoint: Path) -> dict[str, int]:
    logger.info("loading realizer contract checkpoint=%s", checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"compose needs a {MODEL_TYPE} checkpoint; "
            f"got {payload.get('model_type')!r}"
        )
    position = payload["model_state_dict"]["event_position_embedding.weight"]
    contract = {
        "max_seq_len": int(position.shape[0]),
        "prefix_depth": int(payload["prefix_depth"]),
        "codebook_size": int(payload["codebook_size"]),
        "time_vocab": int(payload["time_vocab"]),
    }
    logger.info(
        "realizer contract max_seq_len=%d prefix_depth=%d codebook_size=%d "
        "time_vocab=%d",
        contract["max_seq_len"],
        contract["prefix_depth"],
        contract["codebook_size"],
        contract["time_vocab"],
    )
    return contract


def _verify_planner_checkpoint(
    samples: Mapping[str, object], planner_checkpoint: Path
) -> None:
    logger.info("verifying planner checkpoint=%s", planner_checkpoint)
    checkpoint = torch.load(
        planner_checkpoint, map_location="cpu", weights_only=False
    )
    config = checkpoint.get("config")
    prefix_codes = _require_tensor(samples, "prefix_codes")
    if prefix_codes.shape[-1] != int(config["prefix_depth"]):
        raise ValueError(
            "planner sample prefix depth disagrees with checkpoint"
        )
    max_occurrences = int(config["max_occurrences"])
    logger.info(
        "planner checkpoint verified prefix_depth=%d max_occurrences=%d "
        "sample_width=%d",
        int(config["prefix_depth"]),
        max_occurrences,
        prefix_codes.size(1),
    )


def _convert_occurrence_samples(
    samples: Mapping[str, object],
    *,
    contract: Mapping[str, int],
    max_samples: int,
) -> dict[str, object]:
    prefix_codes = _require_tensor(samples, "prefix_codes").long()
    lengths = _require_tensor(samples, "lengths").long()
    event_time = samples.get("event_time_minutes")
    if not isinstance(event_time, torch.Tensor):
        raise ValueError(
            "planner samples must contain event_time_minutes for the "
            "time-conditioned realizer"
        )
    event_time = event_time.detach().cpu().float()
    city_ids = samples.get("city_ids")
    if city_ids is not None:
        city_ids = cast(torch.Tensor, city_ids).detach().cpu().long()
    if prefix_codes.size(0) < max_samples:
        raise ValueError(
            f"planner samples contain {prefix_codes.size(0)} rows, "
            f"but --max-samples={max_samples}"
        )
    max_seq_len = contract["max_seq_len"]
    if bool((lengths[:max_samples] > max_seq_len).any()):
        raise ValueError(
            "planner samples contain a trajectory longer than the realizer "
            f"max_seq_len={max_seq_len}; use a matching realizer checkpoint"
        )
    valid = torch.arange(prefix_codes.size(1))[None, :] < lengths[:, None]
    if bool(
        (prefix_codes[:max_samples][valid[:max_samples]] < 0).any()
    ) or bool(
        (
            prefix_codes[:max_samples][valid[:max_samples]]
            >= contract["codebook_size"]
        ).any()
    ):
        raise ValueError(
            "planner sample prefix codes exceed realizer vocabulary"
        )
    if contract["time_vocab"] != 42:
        raise ValueError(
            "planner sample event-time conversion requires realizer time_vocab=42"
        )

    logger.info(
        "adapting occurrence planner samples rows=%d source_width=%d "
        "target_width=%d selected_rows=%d mean_length=%.2f",
        prefix_codes.size(0),
        prefix_codes.size(1),
        contract["max_seq_len"],
        max_samples,
        float(lengths[:max_samples].float().mean()),
    )

    prefix_codes = prefix_codes[:max_samples]
    lengths = lengths[:max_samples]
    event_time = event_time[:max_samples]
    valid = valid[:max_samples]
    clamped = event_time.clamp(0.0, 10079.999)
    day = (clamped / 1440.0).long()
    # The realizer and reference bank use the unequal circadian bins of
    # realizer/time_tokens.py, so the adapter must bucket by the same starts.
    part_of_day = torch.bucketize(
        clamped % 1440.0,
        torch.tensor(PART_OF_DAY_START_MINUTES[1:], dtype=clamped.dtype),
        right=True,
    )
    time_tokens = (
        (day * N_PART_OF_DAY_BINS + part_of_day).long().masked_fill(~valid, -1)
    )
    # Continuous planner clock for elapsed-time recurrence, unclamped and
    # untokenized; NaN marks inactive positions.
    event_hours = (event_time.double() / 60.0).masked_fill(
        ~valid, float("nan")
    )
    event_hours = event_hours[:, : contract["max_seq_len"]]
    time_tokens = time_tokens[:, : contract["max_seq_len"]]
    prefix_codes = prefix_codes[:, : contract["max_seq_len"]]
    valid = valid[:, : contract["max_seq_len"]]
    width = prefix_codes.size(1)
    if width < contract["max_seq_len"]:
        padding = contract["max_seq_len"] - width
        prefix_codes = torch.nn.functional.pad(
            prefix_codes, (0, 0, 0, padding)
        )
        time_tokens = torch.nn.functional.pad(
            time_tokens, (0, padding), value=-1
        )
        event_hours = torch.nn.functional.pad(
            event_hours, (0, padding), value=float("nan")
        )
        valid = torch.nn.functional.pad(valid, (0, padding), value=False)
    adapted: dict[str, object] = {
        "schema_version": 2,
        "artifact_type": "sampled_prefix_time_plan",
        "prefix_codes": prefix_codes,
        "time_tokens": time_tokens,
        "event_hours": event_hours,
        "valid_mask": valid,
        "row_lengths": lengths,
        "protocol": {
            "planner_source": "occurrence_planner_samples",
            "time_source": "event_time_minutes_to_circadian_weekly_time_token",
            "time_is_reference_copied": False,
            "prefix_depth": contract["prefix_depth"],
            "max_events": contract["max_seq_len"],
            "selected_rows": max_samples,
            "reference_row_alignment": False,
        },
    }
    if city_ids is not None:
        adapted["city_id"] = city_ids[:max_samples]
    return adapted


def _prepare_sampled_plan(
    planner_samples: Path,
    *,
    realizer_checkpoint: Path,
    planner_checkpoint: Path | None,
    max_samples: int,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None, dict[str, Any]]:
    payload = torch.load(
        planner_samples, map_location="cpu", weights_only=False
    )
    contract = realizer_contract(realizer_checkpoint)
    artifact_type = payload.get("artifact_type")
    logger.info(
        "preparing planner input path=%s artifact_type=%s max_samples=%d",
        planner_samples,
        artifact_type,
        max_samples,
    )
    if planner_checkpoint is not None:
        if artifact_type != "occurrence_planner_samples":
            raise ValueError(
                "--planner-checkpoint verification requires an "
                "occurrence_planner_samples artifact"
            )
        _verify_planner_checkpoint(payload, planner_checkpoint)
    if artifact_type == "sampled_prefix_time_plan":
        logger.info("using pre-adapted sampled prefix-time plan")
        return (
            planner_samples,
            None,
            {
                "input_artifact_type": artifact_type,
                "adapted": False,
                "realizer_contract": contract,
            },
        )
    if artifact_type != "occurrence_planner_samples":
        raise ValueError(
            "--planner-samples must be occurrence_planner_samples or "
            "sampled_prefix_time_plan"
        )
    adapted = _convert_occurrence_samples(
        payload, contract=contract, max_samples=max_samples
    )
    temporary = tempfile.TemporaryDirectory(prefix="semoflow-compose-")
    adapted_path = Path(temporary.name) / "sampled_prefix_time_plan.pt"
    torch.save(adapted, adapted_path)
    logger.info("wrote temporary adapted plan path=%s", adapted_path)
    return (
        adapted_path,
        temporary,
        {
            "input_artifact_type": artifact_type,
            "adapted": True,
            "realizer_contract": contract,
        },
    )


def _comparison_summary(metrics: Mapping[str, object]) -> dict[str, object]:
    decoded = metrics.get("decoded_poi_metrics")
    if not isinstance(decoded, dict):
        return {}
    decoded = cast(Mapping[str, object], decoded)
    trajectory = decoded.get("trajectory_metrics")
    spatial = decoded.get("spatial_metrics")
    poi = decoded.get("poi_distribution")
    identity = decoded.get("identity_dynamics")
    trajectory_map = (
        cast(dict[str, object], trajectory)
        if isinstance(trajectory, dict)
        else None
    )
    spatial_map = (
        cast(dict[str, object], spatial) if isinstance(spatial, dict) else None
    )
    poi_map = cast(dict[str, object], poi) if isinstance(poi, dict) else None
    identity_map = (
        cast(dict[str, object], identity)
        if isinstance(identity, dict)
        else None
    )
    identity_poi = (
        cast(dict[str, object], identity_map["poi"])
        if identity_map is not None
        and isinstance(identity_map.get("poi"), dict)
        else None
    )
    recurrence_distances = (
        cast(dict[str, object], identity_poi["distribution_distances"])
        if identity_poi is not None
        and isinstance(identity_poi.get("distribution_distances"), dict)
        else None
    )
    recurrence_metric = (
        cast(
            dict[str, object],
            recurrence_distances["return_gap_distribution_jsd"],
        )
        if recurrence_distances is not None
        and isinstance(
            recurrence_distances.get("return_gap_distribution_jsd"), dict
        )
        else None
    )
    return {
        "poi_frequency_jsd": (
            poi_map.get("poi_frequency_jsd") if poi_map is not None else None
        ),
        "geohash_jsd": (
            trajectory_map.get("geohash_spatial_jsd")
            if trajectory_map is not None
            else None
        ),
        "travel_jsd": (
            trajectory_map.get("travel_dist_jsd")
            if trajectory_map is not None
            else None
        ),
        "radius_jsd": (
            spatial_map.get("radius_of_gyration_jsd")
            if spatial_map is not None
            else None
        ),
        "mobility_motif_jsd": (
            trajectory_map.get("mobility_motif_jsd")
            if trajectory_map is not None
            else None
        ),
        "mobility_recurrence_pattern_jsd": (
            recurrence_metric.get("point_estimate")
            if recurrence_metric is not None
            else None
        ),
        "multiplicity_jsd": _decoded_lookup(
            metrics,
            "identity_dynamics",
            "poi",
            "distribution_distances",
            "multiplicity_distribution_jsd",
            "point_estimate",
        ),
        "time_location_joint_jsd": _decoded_lookup(
            metrics, "temporal_spatial", "time_location", "joint_jsd"
        ),
        "gap_distance_joint_jsd": _decoded_lookup(
            metrics, "temporal_spatial", "gap_distance", "joint_jsd"
        ),
        "inter_return_time_jsd": _decoded_lookup(
            metrics, "elapsed_recurrence", "inter_return_time_jsd"
        ),
        "return_index_time_joint_jsd": _decoded_lookup(
            metrics, "elapsed_recurrence", "return_index_time_joint_jsd"
        ),
    }


def _decoded_lookup(metrics: Mapping[str, object], *path: str) -> object:
    """Nested value under ``decoded_poi_metrics``, or ``None`` if absent."""
    node: object = metrics.get("decoded_poi_metrics")
    for key in path:
        if not isinstance(node, dict):
            return None
        node = cast(Mapping[str, object], node).get(key)
    return node


def brief_summary(metrics: Mapping[str, object]) -> dict[str, object]:
    summary = _comparison_summary(metrics)
    decoded = metrics.get("decoded_poi_metrics")
    decoded_map = (
        cast(Mapping[str, object], decoded)
        if isinstance(decoded, dict)
        else None
    )
    poi = (
        decoded_map.get("poi_distribution")
        if decoded_map is not None
        else None
    )
    poi_map = (
        cast(Mapping[str, object], poi) if isinstance(poi, dict) else None
    )
    summary["poi_coverage"] = (
        poi_map.get("poi_coverage") if poi_map is not None else None
    )
    return summary


def evaluate_planner_realizer(
    *,
    realizer_checkpoint: Path,
    planner_samples: Path,
    planner_checkpoint: Path | None,
    bank: Path,
    codebooks: Path,
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
    brief_metrics: bool = False,
    catalog_noise_coupling: str = "independent",
) -> dict[str, Any]:
    started = time.perf_counter()
    logger.info(
        "planner+realizer evaluation start planner_samples=%s "
        "realizer_checkpoint=%s max_samples=%d batch_size=%d device=%s "
        "solver=%s num_steps=%d",
        planner_samples,
        realizer_checkpoint,
        max_samples,
        batch_size,
        device,
        solver,
        num_steps,
    )
    if reference_split not in {"val", "test"}:
        raise ValueError("reference_split must be 'val' or 'test'")
    logger.info(
        "composition inputs validated bank=%s codebooks=%s lookup=%s "
        "grounding_index=%s places=%s",
        bank,
        codebooks,
        lookup,
        grounding_index,
        places,
    )

    sampled_plan, temporary, adapter = _prepare_sampled_plan(
        planner_samples,
        realizer_checkpoint=realizer_checkpoint,
        planner_checkpoint=planner_checkpoint,
        max_samples=max_samples,
    )
    try:
        from semoflow.realizer.evaluation import evaluate

        logger.info("starting latent-flow gold-prefix and sampled-plan arms")
        result = evaluate(
            checkpoint_path=realizer_checkpoint,
            bank_path=bank,
            sampled_plan_path=sampled_plan,
            codebooks_path=codebooks,
            lookup_path=lookup,
            grounding_path=grounding_index,
            places_path=places,
            output_path=output,
            max_samples=max_samples,
            batch_size=batch_size,
            device=device,
            num_steps=num_steps,
            solver=solver,
            reference_split=reference_split,
            candidate_selection=candidate_selection,
            candidate_prior_path=candidate_prior,
            catalog_temperature_override=catalog_temperature,
            brief_metrics=brief_metrics,
            catalog_noise_coupling=catalog_noise_coupling,
        )
        logger.info(
            "latent-flow evaluator finished rows=%s elapsed_seconds=%.1f",
            result.get("rows"),
            time.perf_counter() - started,
        )
    except Exception:
        logger.exception("planner+realizer evaluation failed")
        raise
    finally:
        if temporary is not None:
            temporary.cleanup()

    result["planner_samples"] = str(planner_samples)
    result["sampled_plan"] = str(planner_samples)
    result["planner_checkpoint"] = (
        None if planner_checkpoint is None else str(planner_checkpoint)
    )
    result["composition_protocol"] = {
        "input_adapter": adapter,
        "reference_plan_source": f"realizer_bank_{reference_split}_split",
        "generated_plan_source": "planner_samples",
        "reference_row_alignment": False,
        "brief_metrics": brief_metrics,
    }
    result["summary"] = {
        arm: (
            brief_summary(metrics)
            if brief_metrics
            else _comparison_summary(metrics)
        )
        for arm, metrics in result.get("evaluations", {}).items()
        if isinstance(metrics, dict)
    }
    result = normalize_mapping(result)
    write_json(output, result)
    logger.info(
        "planner+realizer evaluation complete output=%s elapsed_seconds=%.1f",
        output,
        time.perf_counter() - started,
    )
    return result


__all__ = [
    "brief_summary",
    "evaluate_planner_realizer",
    "realizer_contract",
]
