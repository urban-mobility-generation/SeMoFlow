from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import torch

from semoflow.metrics.trajectory.elapsed_recurrence import (
    ReturnEvents,
    elapsed_recurrence_jsd,
)
from semoflow.metrics.trajectory.grounding import (
    compute_decoded_sid_poi_metrics,
    ground_sid_tokens_to_poi_tokens,
)
from semoflow.metrics.trajectory.scoring import (
    compute_trajectory_metrics,
    summarize_mobility_metrics,
)
from semoflow.metrics.trajectory.temporal_spatial import (
    events_from_place_ids,
    temporal_spatial_jsd,
)
from semoflow.realizer.bank import (
    load_bank_payload,
    resolve_bank_split_window,
)
from semoflow.realizer.objective import normalize_tail
from semoflow.realizer.time_tokens import N_PART_OF_DAY_BINS
from semoflow.tokenization.decoding.codebooks import (
    codes_to_embeddings_from_codebooks,
)
from semoflow.tokenization.decoding.grounding_index import (
    SemanticIDGroundingIndex,
)
from semoflow.tokenization.token_space import (
    REAL_TOKEN_OFFSET,
    apply_trajectory_valid_mask,
    sid_code_columns,
    to_token_fields,
)


def load_lookup(path: Path) -> np.ndarray:
    columns = list(
        sid_code_columns(pl.scan_parquet(path).collect_schema().names())
    )
    if not columns:
        raise ValueError(f"lookup carries no sem_code_* columns: {path}")
    frame = pl.read_parquet(path, columns=columns)
    return np.stack(
        [frame[name].to_numpy() for name in columns],
        axis=1,
    ).astype(np.int64)


def load_places(
    path: Path,
) -> tuple[dict[int, tuple[float, float]], dict[int, str], int]:
    """In-range POI coordinates, geohashes and the placekey vocab size."""
    frame = pl.read_parquet(
        path, columns=["placekey_id", "latitude", "longitude", "geohash_5"]
    )
    located = frame.filter(
        pl.col("latitude").is_between(-90.0, 90.0)
        & pl.col("longitude").is_between(-180.0, 180.0)
    )
    coordinates = {
        int(placekey_id): (float(latitude), float(longitude))
        for placekey_id, latitude, longitude, _ in located.iter_rows()
    }
    geohashes = {
        int(placekey_id): str(geohash)
        for placekey_id, *_, geohash in frame.iter_rows()
        if geohash is not None
    }
    vocab_size = int(frame["placekey_id"].to_numpy().max(initial=0)) + 1
    return coordinates, geohashes, vocab_size


def coordinate_less_selection_metrics(
    selected_placekey_ids: torch.Tensor,
    valid: torch.BoolTensor,
    coordinates: Mapping[int, tuple[float, float]],
) -> dict[str, float | int]:
    selected = selected_placekey_ids.long().reshape(-1).cpu().tolist()
    available = torch.tensor(
        [int(placekey_id) in coordinates for placekey_id in selected],
        dtype=torch.bool,
    ).view_as(valid)
    active = valid.bool().cpu()
    unavailable = active & ~available
    events = int(active.sum().item())
    unavailable_count = int(unavailable.sum().item())
    effective_count = events - unavailable_count
    return {
        "events": events,
        "effective_coordinate_selection_count": effective_count,
        "effective_coordinate_selection_rate": float(
            effective_count / max(events, 1)
        ),
        "coordinate_less_selection_count": unavailable_count,
        "coordinate_less_selection_rate": float(
            unavailable_count / max(events, 1)
        ),
    }


def load_reference(
    bank_path: Path,
    *,
    max_samples: int,
    split: str = "val",
    payload: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, torch.Tensor]:
    """The leading ``max_samples`` bank rows of a split, with their POIs."""
    if payload is None:
        payload = load_bank_payload(bank_path)
    if split not in {"val", "test"}:
        raise ValueError("reference split must be 'val' or 'test'")
    offset, count = resolve_bank_split_window(payload, split, path=bank_path)
    selected_count = min(int(count), max_samples)
    if selected_count == 0:
        raise ValueError("reference selection is empty")
    rows = slice(offset, offset + selected_count)
    sid_codes = payload["sid_codes"][rows].long()
    time_tokens = payload["time_tokens"][rows].long()
    valid_mask = payload["valid_mask"][rows].bool()

    source = payload["source"]
    segments_value = source.get("segments_path")
    segments_path = Path(segments_value)
    skipped = source.get("skipped_missing_sid", {})
    if isinstance(skipped, dict) and int(skipped.get(split, 0)):
        raise ValueError(
            "cannot reconstruct aligned reference POIs when the bank "
            f"skipped {split} segments with missing SIDs"
        )
    max_events = int(source.get("max_events", sid_codes.size(1)))
    min_length = int(source.get("min_length", 1))
    frame = (
        pl.scan_parquet(segments_path)
        .filter(pl.col("_split") == split)
        .filter(
            pl.col("placekey_id").list.len().is_between(min_length, max_events)
        )
        .head(selected_count)
        .collect()
    )
    if frame.height != selected_count:
        raise ValueError(
            f"reference segment metadata has {frame.height} rows; "
            f"expected {selected_count}"
        )
    placekey_ids = torch.zeros_like(valid_mask, dtype=torch.long)
    for row_index, row in enumerate(frame.iter_rows(named=True)):
        ids = [int(value) for value in row["placekey_id"]]
        length = int(valid_mask[row_index].sum().item())
        if len(ids) != length:
            raise ValueError(
                "reference segment and bank valid lengths do not align at "
                f"row {row_index}: {len(ids)} != {length}"
            )
        placekey_ids[row_index, :length] = torch.tensor(ids, dtype=torch.long)
    return sid_codes, time_tokens, valid_mask, placekey_ids


def aligned_reference_event_hours(
    payload: dict[str, Any],
    *,
    split: str,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Observed event times in nominal local hours from each row's week start.

    Segments are naive local timestamps grouped into Monday-truncated weeks,
    so each row's origin is Monday 00:00 of its first event's week and its
    times lie in [0, 168) h. Rows are the first ``valid_mask.size(0)`` rows of
    the split, selected exactly as :func:`load_reference` selects them.
    Inactive positions are NaN.
    """
    source = payload["source"]
    max_events = int(source.get("max_events", valid_mask.size(1)))
    min_length = int(source.get("min_length", 1))
    frame = cast(
        pl.DataFrame,
        pl.scan_parquet(Path(source["segments_path"]))
        .filter(pl.col("_split") == split)
        .filter(
            pl.col("placekey_id").list.len().is_between(min_length, max_events)
        )
        .head(int(valid_mask.size(0)))
        .select("_timestamps")
        .collect(),
    )
    lengths = torch.from_numpy(
        frame["_timestamps"].list.len().to_numpy().astype(np.int64)
    )
    if not torch.equal(lengths, valid_mask.sum(dim=1).long()):
        raise ValueError("reference timestamps do not align with bank rows")
    exploded = frame.with_columns(
        pl.col("_timestamps").list.first().dt.truncate("1w").alias("week")
    ).explode("_timestamps")
    hours = (
        exploded["_timestamps"] - exploded["week"]
    ).dt.total_microseconds().to_numpy() / 3.6e9
    out = torch.full(valid_mask.shape, float("nan"), dtype=torch.float64)
    rows = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    starts = torch.cumsum(lengths, 0) - lengths
    columns = torch.arange(int(lengths.sum())) - torch.repeat_interleave(
        starts, lengths
    )
    out[rows, columns] = torch.from_numpy(hours).double()
    return out


def sampled_plan_event_hours(
    path: Path, *, max_samples: int
) -> torch.Tensor | None:
    """The plan's continuous event hours, or ``None`` for a token-only plan."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hours = payload.get("event_hours") if isinstance(payload, dict) else None
    return None if hours is None else hours[:max_samples].double()


def elapsed_recurrence_block(
    real_poi: torch.Tensor,
    real_valid: torch.Tensor,
    real_hours: torch.Tensor | None,
    generated_poi: torch.Tensor,
    generated_valid: torch.Tensor,
    generated_hours: torch.Tensor | None,
) -> dict[str, Any]:
    """Elapsed-time recurrence on grounded POIs, or why it is unavailable.

    Events that ground to no POI (ID 0) are dropped, as the index
    recurrence metric drops them.
    """
    missing = [
        side
        for side, hours in (
            ("real", real_hours),
            ("generated", generated_hours),
        )
        if hours is None
    ]
    if missing:
        return {"unavailable": f"no continuous clock: {', '.join(missing)}"}
    assert real_hours is not None and generated_hours is not None

    def events(poi, valid, hours) -> ReturnEvents:
        if hours.shape != valid.shape:
            raise ValueError("event hours do not align with the valid mask")
        poi = poi.cpu().numpy()
        return ReturnEvents(
            place_ids=poi,
            hours=hours.cpu().double().numpy(),
            valid=valid.cpu().numpy().astype(bool) & (poi != 0),
        )

    return elapsed_recurrence_jsd(
        events(real_poi, real_valid, real_hours),
        events(generated_poi, generated_valid, generated_hours),
    )


def load_sampled_plan(
    path: Path,
    *,
    max_samples: int,
    expected_width: int,
    prefix_depth: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    prefix = payload["prefix_codes"].long()
    times = payload["time_tokens"].long()
    valid = payload["valid_mask"].bool()
    if prefix.size(1) != expected_width:
        raise ValueError(
            f"sampled plan width {prefix.size(1)} does not match model width "
            f"{expected_width}"
        )
    if prefix.size(0) < max_samples:
        raise ValueError("sampled plan artifact has too few rows")
    protocol = payload.get("protocol")
    if protocol.get("reference_row_alignment") is not False:
        raise ValueError("sampled plan must declare unpaired evaluation")
    return (
        prefix[:max_samples],
        times[:max_samples],
        valid[:max_samples],
        protocol,
    )


def dynamics(
    codes: torch.Tensor,
    valid: torch.BoolTensor,
    grounding: SemanticIDGroundingIndex,
) -> dict[str, float]:
    repeat_masses: list[float] = []
    adjacent_rates: list[float] = []
    support = 0
    widths: list[int] = []
    for row in range(codes.size(0)):
        sequence = [
            tuple(int(value) for value in sid)
            for sid in codes[row][valid[row]].tolist()
        ]
        if not sequence:
            continue
        repeat_masses.append(
            float((len(sequence) - len(set(sequence))) / len(sequence))
        )
        if len(sequence) > 1:
            adjacent_rates.append(
                float(
                    sum(
                        left == right
                        for left, right in itertools.pairwise(sequence)
                    )
                    / (len(sequence) - 1)
                )
            )
        for sid in sequence:
            candidates = grounding.candidate_payloads(sid)
            widths.append(len(candidates))
            support += bool(candidates)
    events = int(valid.sum().item())
    return {
        "events": events,
        "catalog_support_rate": float(support / max(events, 1)),
        "candidate_width_mean": float(np.mean(widths) if widths else 0.0),
        "candidate_width_p50": float(
            np.percentile(widths, 50) if widths else 0.0
        ),
        "full_sid_repeat_mass_mean": float(
            np.mean(repeat_masses) if repeat_masses else 0.0
        ),
        "adjacent_full_sid_copy_rate_mean": float(
            np.mean(adjacent_rates) if adjacent_rates else 0.0
        ),
    }


def accuracy(
    generated: torch.Tensor,
    target: torch.Tensor,
    valid: torch.BoolTensor,
    prefix_depth: int,
) -> dict[str, Any]:
    suffix_equal = generated[..., prefix_depth:] == target[..., prefix_depth:]
    events = int(valid.sum().item())
    levels = (suffix_equal & valid[..., None]).sum(dim=(0, 1))
    full = ((generated == target).all(dim=-1) & valid).sum()
    return {
        "suffix_level_accuracy": [
            float(value / max(events, 1)) for value in levels
        ],
        "full_sid_accuracy": float(full / max(events, 1)),
    }


def compute_tail_error_metrics(
    generated_tail: torch.Tensor,
    target_codes: torch.Tensor,
    valid: torch.BoolTensor,
    codebooks: torch.Tensor,
    tail_whitener: torch.Tensor,
    *,
    prefix_depth: int,
) -> dict[str, float]:

    device = generated_tail.device
    valid = valid.to(device=device, dtype=torch.bool)
    target_codes = target_codes.to(device=device, dtype=torch.long).clone()
    target_codes[~valid] = 0
    codebooks = codebooks.to(device=device).float()
    target_tail = codes_to_embeddings_from_codebooks(
        target_codes[..., prefix_depth:],
        codebooks[prefix_depth:],
    )
    raw_error = generated_tail.float() - target_tail.float()
    normalized_error = normalize_tail(
        raw_error,
        torch.zeros(raw_error.shape[-1], device=device),
        tail_whitener.to(device=device).float(),
    )
    active_raw = raw_error[valid]
    active_normalized = normalized_error[valid]
    events = int(valid.sum().item())
    if events == 0:
        return {
            "events": 0.0,
            "tail_rmse": 0.0,
            "tail_mae": 0.0,
            "normalized_tail_rmse": 0.0,
            "normalized_tail_mae": 0.0,
        }
    return {
        "events": float(events),
        "tail_rmse": float(active_raw.square().mean().sqrt().item()),
        "tail_mae": float(active_raw.abs().mean().item()),
        "normalized_tail_rmse": float(
            active_normalized.square().mean().sqrt().item()
        ),
        "normalized_tail_mae": float(active_normalized.abs().mean().item()),
    }


CANDIDATE_PRIOR_FORMAT = "grounding_candidate_prior_v1"


def load_candidate_prior(path: Path) -> dict[int, float]:
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != CANDIDATE_PRIOR_FORMAT:
        raise ValueError(
            f"{path} is not a {CANDIDATE_PRIOR_FORMAT} artifact; "
            f"got {payload.get('format')!r}"
        )
    if payload.get("split") != "train":
        raise ValueError(
            f"candidate prior must be fitted on the train split; "
            f"{path} declares {payload.get('split')!r}"
        )
    counts = payload.get("counts")
    return {int(key): float(value) for key, value in counts.items()}


def metric_block(
    real_codes: torch.Tensor,
    real_valid: torch.BoolTensor,
    generated_codes: torch.Tensor,
    generated_valid: torch.BoolTensor,
    *,
    grounding: SemanticIDGroundingIndex,
    coordinates: dict[int, tuple[float, float]],
    geohashes: dict[int, str],
    placekey_vocab_size: int,
    real_time_tokens: torch.Tensor,
    generated_time_tokens: torch.Tensor,
    reference_placekey_ids: torch.Tensor | None = None,
    cluster_ids: list[str] | None = None,
    candidate_selection: str = "first",
    candidate_weights: Mapping[int, float] | None = None,
    compute_device: torch.device | str | None = None,
    brief_metrics: bool = False,
    real_event_hours: torch.Tensor | None = None,
    generated_event_hours: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute the grounding-first metric block for one arm.

    ``real_time_tokens``/``generated_time_tokens`` are the 42 weekly time
    tokens aligned with the codes. They give the motif day keys and the joint
    temporal-spatial JSDs. ``real_event_hours``/``generated_event_hours`` are
    continuous event times in hours; when both are given the elapsed-time
    recurrence JSDs are scored, otherwise that section records why it is
    unavailable. Times are never derived from tokens.
    """
    motif_day_keys_real = real_time_tokens // N_PART_OF_DAY_BINS
    motif_day_keys_generated = generated_time_tokens // N_PART_OF_DAY_BINS
    fields = [f"sem_code_{level}" for level in range(real_codes.size(-1))]
    real_tokens = apply_trajectory_valid_mask(
        to_token_fields(real_codes, fields), real_valid
    )
    generated_tokens = apply_trajectory_valid_mask(
        to_token_fields(generated_codes, fields), generated_valid
    )
    category_string_vocab = grounding.build_category_string_vocab()
    naics_string_vocab = grounding.build_naics_string_vocab()
    real_poi, real_stats = ground_sid_tokens_to_poi_tokens(
        real_tokens,
        grounding,
        category_string_vocab=category_string_vocab,
        naics_string_vocab=naics_string_vocab,
        preferred_placekey_ids=reference_placekey_ids,
    )
    generated_poi, generated_stats = ground_sid_tokens_to_poi_tokens(
        generated_tokens,
        grounding,
        category_string_vocab=category_string_vocab,
        naics_string_vocab=naics_string_vocab,
        candidate_selection=candidate_selection,
        candidate_weights=candidate_weights,
    )
    max_code = int(
        max(
            real_codes.max().item() if real_codes.numel() else 0,
            generated_codes.max().item() if generated_codes.numel() else 0,
        )
    )
    field_vocab = {field: max_code + REAL_TOKEN_OFFSET + 1 for field in fields}
    sid_metrics = (
        None
        if brief_metrics
        else compute_trajectory_metrics(
            real_tokens,
            generated_tokens,
            field_vocab,
            motif_day_keys_real=motif_day_keys_real,
            motif_day_keys_gen=motif_day_keys_generated,
            compute_device=compute_device,
        )
    )
    grounded = compute_decoded_sid_poi_metrics(
        real_poi,
        generated_poi,
        real_stats=real_stats,
        gen_stats=generated_stats,
        placekey_latlon=coordinates,
        geohash_lookup=geohashes,
        placekey_vocab_size=placekey_vocab_size,
        category_vocab_size=len(category_string_vocab) + 1,
        cluster_ids=cluster_ids,
        motif_day_keys_real=motif_day_keys_real,
        motif_day_keys_gen=motif_day_keys_generated,
        compute_device=compute_device,
        brief_metrics=brief_metrics,
    )
    grounded["temporal_spatial"] = temporal_spatial_jsd(
        events_from_place_ids(
            real_poi["placekey_id"].cpu().numpy(),
            real_time_tokens.cpu().numpy(),
            real_valid.cpu().numpy(),
            coordinates=coordinates,
            geohashes=geohashes,
        ),
        events_from_place_ids(
            generated_poi["placekey_id"].cpu().numpy(),
            generated_time_tokens.cpu().numpy(),
            generated_valid.cpu().numpy(),
            coordinates=coordinates,
            geohashes=geohashes,
        ),
    )
    grounded["elapsed_recurrence"] = elapsed_recurrence_block(
        real_poi["placekey_id"],
        real_valid,
        real_event_hours,
        generated_poi["placekey_id"],
        generated_valid,
        generated_event_hours,
    )
    result: dict[str, Any] = {"decoded_poi_metrics": grounded}
    if sid_metrics is not None:
        result.update(
            {
                "sid_code_metrics": sid_metrics.to_dict(),
                "sid_code_summary": summarize_mobility_metrics(sid_metrics),
                "coordinate_guard": coordinate_less_selection_metrics(
                    generated_poi["placekey_id"],
                    generated_valid,
                    coordinates,
                ),
            }
        )
    return result


__all__ = [
    "accuracy",
    "aligned_reference_event_hours",
    "compute_tail_error_metrics",
    "coordinate_less_selection_metrics",
    "dynamics",
    "elapsed_recurrence_block",
    "load_lookup",
    "load_places",
    "load_reference",
    "load_sampled_plan",
    "metric_block",
    "sampled_plan_event_hours",
]
