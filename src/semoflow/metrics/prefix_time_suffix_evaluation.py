from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from semoflow.metrics.trajectory.grounding import (
    compute_decoded_sid_poi_metrics,
    ground_sid_tokens_to_poi_tokens,
)
from semoflow.metrics.trajectory.scoring import (
    compute_trajectory_metrics,
    summarize_mobility_metrics,
)
from semoflow.realizer.bank import (
    load_bank_payload,
    resolve_bank_split_window,
)
from semoflow.realizer.objective import (
    normalize_tail,
    suffix_tail_level_embeddings,
)
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
    if selected_placekey_ids.shape != valid.shape:
        raise ValueError(
            "selected_placekey_ids and valid must have the same shape; got "
            f"{tuple(selected_placekey_ids.shape)} and {tuple(valid.shape)}"
        )
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
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    selected_count = min(int(count), max_samples)
    if selected_count == 0:
        raise ValueError("reference selection is empty")
    rows = slice(offset, offset + selected_count)
    sid_codes = payload["sid_codes"][rows].long()
    time_tokens = payload["time_tokens"][rows].long()
    valid_mask = payload["valid_mask"][rows].bool()

    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("realizer bank is missing source metadata")
    segments_value = source.get("segments_path")
    if not isinstance(segments_value, str):
        raise ValueError(
            "realizer bank source metadata is missing segments_path; "
            "cannot ground reference POIs"
        )
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


def load_sampled_plan(
    path: Path,
    *,
    max_samples: int,
    expected_width: int,
    prefix_depth: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("sampled plan artifact must be a dictionary")
    prefix = payload["prefix_codes"].long()
    times = payload["time_tokens"].long()
    valid = payload["valid_mask"].bool()
    if prefix.ndim != 3 or prefix.shape[-1] != prefix_depth:
        raise ValueError("sampled prefix artifact has an invalid prefix shape")
    if times.shape != valid.shape or prefix.shape[:2] != valid.shape:
        raise ValueError("sampled prefix artifact tensors are misaligned")
    if prefix.size(1) != expected_width:
        raise ValueError(
            f"sampled plan width {prefix.size(1)} does not match model width "
            f"{expected_width}"
        )
    if prefix.size(0) < max_samples:
        raise ValueError("sampled plan artifact has too few rows")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("sampled plan artifact is missing protocol metadata")
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
    tail_mean: torch.Tensor,
    tail_std: torch.Tensor,
    *,
    tail_whitener: torch.Tensor | None = None,
    prefix_depth: int,
) -> dict[str, float]:
    if generated_tail.ndim != 3 or target_codes.ndim != 3:
        raise ValueError("tail and target codes must be rank-3 tensors")
    if generated_tail.shape[:2] != target_codes.shape[:2]:
        raise ValueError("tail and target rows are not aligned")
    if valid.shape != target_codes.shape[:2]:
        raise ValueError("valid mask does not align with target codes")
    if not 0 < prefix_depth < target_codes.shape[-1]:
        raise ValueError("prefix_depth must be within the SID width")
    if tail_mean.shape != tail_std.shape or tail_mean.ndim != 1:
        raise ValueError("tail statistics must have shape [latent_dim]")
    if (tail_std <= 0).any() or not torch.isfinite(tail_std).all():
        raise ValueError("tail_std must be finite and strictly positive")

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
    if tail_whitener is None:
        normalized_error = raw_error / tail_std.to(device=device).float()
    else:
        normalized_error = normalize_tail(
            raw_error,
            torch.zeros_like(tail_mean, device=device),
            torch.ones_like(tail_std, device=device),
            tail_whitener=tail_whitener.to(device=device).float(),
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


def compute_tail_level_error_metrics(
    generated_tail_levels: torch.Tensor,
    target_codes: torch.Tensor,
    valid: torch.BoolTensor,
    codebooks: torch.Tensor,
    tail_mean: torch.Tensor,
    tail_std: torch.Tensor,
    *,
    prefix_depth: int,
) -> dict[str, Any]:
    if generated_tail_levels.ndim != 4 or target_codes.ndim != 3:
        raise ValueError("factorized tail and target codes have invalid ranks")
    if generated_tail_levels.shape[:2] != target_codes.shape[:2]:
        raise ValueError("factorized tail and target rows are not aligned")
    if valid.shape != target_codes.shape[:2]:
        raise ValueError("valid mask does not align with target codes")
    if tail_mean.shape != tail_std.shape or tail_mean.ndim != 2:
        raise ValueError("factorized tail statistics must have shape [R, D]")
    if generated_tail_levels.shape[-2:] != tuple(tail_mean.shape):
        raise ValueError(
            "factorized tail and statistics have incompatible shapes"
        )
    if (tail_std <= 0).any() or not torch.isfinite(tail_std).all():
        raise ValueError("tail_std must be finite and strictly positive")

    device = generated_tail_levels.device
    valid = valid.to(device=device, dtype=torch.bool)
    target_codes = target_codes.to(device=device, dtype=torch.long).clone()
    target_codes[~valid] = 0
    target_levels = suffix_tail_level_embeddings(
        target_codes,
        codebooks.to(device=device).float(),
        prefix_depth=prefix_depth,
    ).float()
    raw_error = generated_tail_levels.float() - target_levels
    normalized_error = raw_error / tail_std.to(device=device).float()
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
            "per_level": [],
        }

    def _summary(
        raw: torch.Tensor, normalized: torch.Tensor
    ) -> dict[str, float]:
        return {
            "tail_rmse": float(raw.square().mean().sqrt().item()),
            "tail_mae": float(raw.abs().mean().item()),
            "normalized_tail_rmse": float(
                normalized.square().mean().sqrt().item()
            ),
            "normalized_tail_mae": float(normalized.abs().mean().item()),
        }

    per_level = [
        _summary(active_raw[:, level], active_normalized[:, level])
        for level in range(generated_tail_levels.size(-2))
    ]
    return {
        "events": float(events),
        **_summary(active_raw, active_normalized),
        "per_level": per_level,
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
    if not isinstance(counts, dict) or not counts:
        raise ValueError(f"{path} has no candidate counts")
    return {int(key): float(value) for key, value in counts.items()}


def metric_block(
    real_codes: torch.Tensor,
    real_valid: torch.BoolTensor,
    generated_codes: torch.Tensor,
    generated_valid: torch.BoolTensor,
    *,
    grounding: SemanticIDGroundingIndex,
    generated_grounding: SemanticIDGroundingIndex | None = None,
    coordinates: dict[int, tuple[float, float]],
    geohashes: dict[int, str],
    placekey_vocab_size: int,
    reference_placekey_ids: torch.Tensor | None = None,
    cluster_ids: list[str] | None = None,
    candidate_selection: str = "first",
    candidate_weights: Mapping[int, float] | None = None,
    motif_day_keys_real: torch.Tensor | None = None,
    motif_day_keys_generated: torch.Tensor | None = None,
    compute_device: torch.device | str | None = None,
    brief_metrics: bool = False,
) -> dict[str, Any]:
    generated_grounding = generated_grounding or grounding
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
        generated_grounding,
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
    "compute_tail_error_metrics",
    "compute_tail_level_error_metrics",
    "coordinate_less_selection_metrics",
    "dynamics",
    "load_lookup",
    "load_places",
    "load_reference",
    "load_sampled_plan",
    "metric_block",
]
