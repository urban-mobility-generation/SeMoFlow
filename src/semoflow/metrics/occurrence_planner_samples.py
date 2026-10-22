from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch

from semoflow.metrics.divergence import (
    jensen_shannon_divergence,
)
from semoflow.planner.artifacts import (
    SAMPLES_ARTIFACT_TYPE,
    OccurrencePlannerCheckpoint,
    load_occurrence_planner_checkpoint,
    load_occurrence_planner_dataset,
)
from semoflow.planner.data import (
    collate_occurrence_tensors,
)
from semoflow.training.utils.runtime import parse_device_spec

_SAMPLE_FIELDS = {
    "artifact_type",
    "prefix_codes",
    "gap_tokens",
    "lengths",
    "model_log_probability",
    "sampling_log_probability",
    "length_was_sampled",
    "city_ids",
    "context_features",
    "context_schema",
    "event_time_minutes",
    "seed",
    "temperature",
    "length_temperature",
    "gap_temperature",
    "prefix_temperature",
    "route_temperature",
}


LENGTH_BINS = ((3, 8), (8, 11), (11, 14), (14, 18), (18, 31))


def histogram_jsd(left: Counter[Any], right: Counter[Any]) -> float:
    keys = tuple(set(left) | set(right))
    left_values = np.asarray([left[key] for key in keys], dtype=np.float64)
    right_values = np.asarray([right[key] for key in keys], dtype=np.float64)
    return jensen_shannon_divergence(left_values, right_values)


@dataclass(frozen=True, slots=True)
class OccurrencePlannerSamplesEvaluationConfig:
    samples: Path
    checkpoint: Path
    reference: Path
    output: Path
    reference_split: str = "validation"
    reference_limit: int | None = None
    evaluation_seed: int = 0
    device: str = "cpu"
    batch_size: int = 1024

    def validate(self) -> None:
        if not self.samples.is_file():
            raise ValueError(
                f"planner samples artifact does not exist: {self.samples}"
            )
        if not self.checkpoint.is_file():
            raise ValueError(
                f"planner checkpoint does not exist: {self.checkpoint}"
            )
        if not self.reference.is_file():
            raise ValueError(
                f"planner reference dataset does not exist: {self.reference}"
            )
        if self.reference_split not in {"train", "validation"}:
            raise ValueError("reference_split must be 'train' or 'validation'")
        if self.reference_limit is not None and self.reference_limit < 1:
            raise ValueError("reference_limit must be positive")
        if isinstance(self.evaluation_seed, bool) or not isinstance(
            self.evaluation_seed, int
        ):
            raise ValueError("evaluation_seed must be an integer")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if isinstance(self.batch_size, bool) or not isinstance(
            self.batch_size, int
        ):
            raise ValueError("batch_size must be an integer")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        input_paths = {
            self.samples.resolve(),
            self.checkpoint.resolve(),
            self.reference.resolve(),
        }
        if self.output.resolve() in input_paths:
            raise ValueError(
                "evaluation output must differ from its input artifacts"
            )


def _resolve_router_diagnostics_device(spec: str) -> torch.device:
    try:
        device, device_ids = parse_device_spec(spec)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid evaluation device: {spec!r}") from exc
    if device_ids:
        raise ValueError(
            "planner-samples evaluation accepts one device, not a GPU list"
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "evaluation device requests CUDA, but CUDA is unavailable"
            )
        if (
            device.index is not None
            and device.index >= torch.cuda.device_count()
        ):
            raise ValueError(
                f"evaluation device {device} is outside the visible CUDA devices"
            )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError(
            "evaluation device requests MPS, but MPS is unavailable"
        )
    return device


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _require_tensor(
    payload: Mapping[str, object],
    key: str,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = payload[key]
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"planner samples {key} must be a tensor")
    if value.device.type != "cpu" or value.dtype != dtype:
        raise ValueError(f"planner samples {key} must be a CPU {dtype} tensor")
    return value


def _require_finite_positive(payload: Mapping[str, object], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"planner samples {key} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"planner samples {key} must be finite and positive")
    return parsed


def _load_samples(path: Path) -> Mapping[str, object]:
    try:
        raw: Any = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"could not load planner samples artifact: {path}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ValueError("planner samples artifact must be a mapping")
    raw = dict(raw)
    raw.pop("format_version", None)
    raw.pop("checkpoint_sha256", None)
    raw.pop("conditions_sha256", None)
    if set(raw) != _SAMPLE_FIELDS:
        raise ValueError(
            "planner samples artifact keys are incomplete or unsupported"
        )
    if raw["artifact_type"] != SAMPLES_ARTIFACT_TYPE:
        raise ValueError("planner samples artifact_type is unsupported")
    return raw


@dataclass(frozen=True, slots=True)
class _PopulationStats:
    summary: dict[str, Any]
    prefix_counts: Counter[tuple[int, ...]]
    transition_counts: Counter[tuple[tuple[int, ...], tuple[int, ...]]]
    transition_counts_by_return_type: dict[
        str, Counter[tuple[tuple[int, ...], tuple[int, ...]]]
    ]
    transition_event_counts_by_return_type: dict[str, int]
    gap_counts: Counter[int]
    signature_digests: tuple[bytes, ...]
    by_prefix_depth: dict[int, _PrefixScaleStats]


@dataclass(frozen=True, slots=True)
class _PrefixScaleStats:
    summary: dict[str, Any]
    prefix_counts: Counter[tuple[int, ...]]
    transition_counts: Counter[tuple[tuple[int, ...], tuple[int, ...]]]
    signature_digests: tuple[bytes, ...]


def _signature_digest(signature: tuple[tuple[int, ...], ...]) -> bytes:
    values = np.asarray(signature, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.digest()


def _entropy_bits(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return float(
        -sum(
            (count / total) * math.log2(count / total)
            for count in counts.values()
        )
    )


def _prefix_scale_stats(
    prefix_codes: torch.Tensor,
    lengths: torch.Tensor,
    *,
    codebook_size: int,
) -> dict[int, _PrefixScaleStats]:
    sample_count, _, prefix_depth = prefix_codes.shape
    output: dict[int, _PrefixScaleStats] = {}
    for depth in range(1, prefix_depth + 1):
        repeat_mass = np.empty(sample_count, dtype=np.float64)
        unique_prefixes = np.empty(sample_count, dtype=np.float64)
        prefix_counts: Counter[tuple[int, ...]] = Counter()
        transition_counts: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = (
            Counter()
        )
        level_counts = [Counter[int]() for _ in range(depth)]
        signature_digests: list[bytes] = []
        adjacent_repeat_count = 0
        adjacent_pair_count = 0
        non_adjacent_return_count = 0
        event_count = 0

        for row_index, raw_length in enumerate(lengths.tolist()):
            length = int(raw_length)
            signature = tuple(
                tuple(int(value) for value in row)
                for row in prefix_codes[row_index, :length, :depth].tolist()
            )
            prefix_counts.update(signature)
            transition_counts.update(pairwise(signature))
            signature_digests.append(_signature_digest(signature))
            repeat_mass[row_index] = (length - len(set(signature))) / length
            unique_prefixes[row_index] = len(set(signature))
            event_count += length
            if length > 1:
                adjacent_pair_count += length - 1
                adjacent_repeat_count += sum(
                    left == right for left, right in pairwise(signature)
                )
            seen: set[tuple[int, ...]] = set()
            previous: tuple[int, ...] | None = None
            for value in signature:
                if value in seen and value != previous:
                    non_adjacent_return_count += 1
                seen.add(value)
                previous = value
            for level in range(depth):
                level_counts[level].update(
                    int(row[level])
                    for row in prefix_codes[row_index, :length].tolist()
                )

        summary = {
            "prefix_depth": depth,
            "event_count": event_count,
            "unique_prefix_count": len(prefix_counts),
            "prefix_utilization": len(prefix_counts) / (codebook_size**depth),
            "mean_repeat_mass": float(repeat_mass.mean()),
            "std_repeat_mass": float(repeat_mass.std()),
            "p05_repeat_mass": float(np.quantile(repeat_mass, 0.05)),
            "median_repeat_mass": float(np.median(repeat_mass)),
            "p95_repeat_mass": float(np.quantile(repeat_mass, 0.95)),
            "mean_unique_prefixes": float(unique_prefixes.mean()),
            "adjacent_repeat_share": (
                adjacent_repeat_count / adjacent_pair_count
                if adjacent_pair_count
                else 0.0
            ),
            "non_adjacent_prefix_return_rate": (
                non_adjacent_return_count / event_count if event_count else 0.0
            ),
            "non_adjacent_prefix_return_event_count": (
                non_adjacent_return_count
            ),
            "codebook_entropy_bits": [
                _entropy_bits(counts) for counts in level_counts
            ],
            "unique_complete_plan_rate": len(set(signature_digests))
            / len(signature_digests),
        }
        output[depth] = _PrefixScaleStats(
            summary=summary,
            prefix_counts=prefix_counts,
            transition_counts=transition_counts,
            signature_digests=tuple(signature_digests),
        )
    return output


def _population_stats(
    prefix_codes: torch.Tensor,
    gap_tokens: torch.Tensor,
    lengths: torch.Tensor,
    *,
    codebook_size: int,
) -> _PopulationStats:
    length_values = lengths.numpy().astype(np.float64, copy=False)
    repeat_mass = np.empty(len(length_values), dtype=np.float64)
    unique_prefixes = np.empty(len(length_values), dtype=np.float64)
    prefix_counts: Counter[tuple[int, ...]] = Counter()
    transition_counts: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = (
        Counter()
    )
    transition_counts_by_return_type = {
        "explore": Counter(),
        "immediate_return": Counter(),
        "older_return": Counter(),
    }
    transition_event_counts_by_return_type = {
        "explore": 0,
        "immediate_return": 0,
        "older_return": 0,
    }
    gap_counts: Counter[int] = Counter()
    signature_digests: list[bytes] = []

    for row_index, raw_length in enumerate(lengths.tolist()):
        length = int(raw_length)
        signature = tuple(
            tuple(int(value) for value in row)
            for row in prefix_codes[row_index, :length].tolist()
        )
        prefix_counts.update(signature)
        transition_counts.update(pairwise(signature))
        seen: set[tuple[int, ...]] = set()
        for step, current in enumerate(signature):
            if step == 0:
                seen.add(current)
                continue
            previous = signature[step - 1]
            transition = (previous, current)
            if current == previous:
                return_type = "immediate_return"
            elif current in seen:
                return_type = "older_return"
            else:
                return_type = "explore"
            transition_counts_by_return_type[return_type][transition] += 1
            transition_event_counts_by_return_type[return_type] += 1
            seen.add(current)
        gap_counts.update(
            int(value) for value in gap_tokens[row_index, :length].tolist()
        )
        repeat_mass[row_index] = (length - len(set(signature))) / length
        unique_prefixes[row_index] = len(set(signature))
        signature_digests.append(_signature_digest(signature))

    repeat_mass_by_length_bin = []
    for lower, upper in LENGTH_BINS:
        mask = (length_values >= lower) & (length_values < upper)
        repeat_mass_by_length_bin.append(
            {
                "lower_length_inclusive": lower,
                "upper_length_exclusive": upper,
                "trajectory_count": int(mask.sum()),
                "mean_repeat_mass": (
                    None if not mask.any() else float(repeat_mass[mask].mean())
                ),
                "std_repeat_mass": (
                    None if not mask.any() else float(repeat_mass[mask].std())
                ),
            }
        )

    summary = {
        "trajectory_count": len(length_values),
        "occurrence_count": int(length_values.sum()),
        "mean_length": float(length_values.mean()),
        "p05_length": float(np.quantile(length_values, 0.05)),
        "median_length": float(np.median(length_values)),
        "p95_length": float(np.quantile(length_values, 0.95)),
        "mean_repeat_mass": float(repeat_mass.mean()),
        "std_repeat_mass": float(repeat_mass.std()),
        "p05_repeat_mass": float(np.quantile(repeat_mass, 0.05)),
        "median_repeat_mass": float(np.median(repeat_mass)),
        "p95_repeat_mass": float(np.quantile(repeat_mass, 0.95)),
        "repeat_mass_by_length_bin": repeat_mass_by_length_bin,
        "mean_unique_prefixes": float(unique_prefixes.mean()),
        "unique_complete_plan_rate": len(set(signature_digests))
        / len(signature_digests),
    }
    by_prefix_depth = _prefix_scale_stats(
        prefix_codes,
        lengths,
        codebook_size=codebook_size,
    )
    full_scale = by_prefix_depth[prefix_codes.shape[-1]].summary
    summary.update(
        {
            "adjacent_repeat_share": full_scale["adjacent_repeat_share"],
            "non_adjacent_prefix_return_rate": full_scale[
                "non_adjacent_prefix_return_rate"
            ],
            "non_adjacent_prefix_return_event_count": full_scale[
                "non_adjacent_prefix_return_event_count"
            ],
            "codebook_entropy_bits": full_scale["codebook_entropy_bits"],
        }
    )
    return _PopulationStats(
        summary=summary,
        prefix_counts=prefix_counts,
        transition_counts=transition_counts,
        transition_counts_by_return_type=transition_counts_by_return_type,
        transition_event_counts_by_return_type=(
            transition_event_counts_by_return_type
        ),
        gap_counts=gap_counts,
        signature_digests=tuple(signature_digests),
        by_prefix_depth=by_prefix_depth,
    )


def _train_signature_overlap_metrics(
    generated_digests: tuple[bytes, ...],
    train_prefix_codes: torch.Tensor,
    train_lengths: torch.Tensor,
) -> dict[str, float]:
    generated_set = set(generated_digests)
    matched: set[bytes] = set()
    for row_index, raw_length in enumerate(train_lengths.tolist()):
        length = int(raw_length)
        signature = tuple(
            tuple(int(value) for value in row)
            for row in train_prefix_codes[row_index, :length].tolist()
        )
        digest = _signature_digest(signature)
        if digest in generated_set:
            matched.add(digest)
    return {
        "unique_train_signature_overlap": float(
            len(matched) / len(generated_set)
        ),
        "sample_train_copy_rate": float(
            sum(digest in matched for digest in generated_digests)
            / len(generated_digests)
        ),
    }


def _population_metrics(
    generated: _PopulationStats,
    reference: _PopulationStats,
    *,
    train_prefix_codes: torch.Tensor | None = None,
    train_lengths: torch.Tensor | None = None,
    generated_name: str = "generated",
    reference_name: str = "reference",
) -> dict[str, Any]:
    generated_mean_length = generated.summary["mean_length"]
    reference_mean_length = reference.summary["mean_length"]
    generated_p95_length = generated.summary["p95_length"]
    reference_p95_length = reference.summary["p95_length"]
    by_prefix_depth: dict[str, dict[str, Any]] = {}
    for depth in sorted(generated.by_prefix_depth):
        generated_scale = generated.by_prefix_depth[depth]
        reference_scale = reference.by_prefix_depth[depth]
        by_prefix_depth[str(depth)] = {
            generated_name: generated_scale.summary,
            reference_name: reference_scale.summary,
            "comparisons": {
                "prefix_histogram_jsd": histogram_jsd(
                    generated_scale.prefix_counts,
                    reference_scale.prefix_counts,
                ),
                "prefix_transition_histogram_jsd": histogram_jsd(
                    generated_scale.transition_counts,
                    reference_scale.transition_counts,
                ),
                "mean_repeat_mass_delta": float(
                    generated_scale.summary["mean_repeat_mass"]
                    - reference_scale.summary["mean_repeat_mass"]
                ),
                "adjacent_repeat_share_delta": float(
                    generated_scale.summary["adjacent_repeat_share"]
                    - reference_scale.summary["adjacent_repeat_share"]
                ),
                "non_adjacent_prefix_return_rate_delta": float(
                    generated_scale.summary["non_adjacent_prefix_return_rate"]
                    - reference_scale.summary[
                        "non_adjacent_prefix_return_rate"
                    ]
                ),
            },
        }
    transition_by_return_type: dict[str, Any] = {}
    for return_type in ("explore", "immediate_return", "older_return"):
        generated_counts = generated.transition_counts_by_return_type[
            return_type
        ]
        reference_counts = reference.transition_counts_by_return_type[
            return_type
        ]
        transition_by_return_type[return_type] = {
            "generated_transition_count": generated.transition_event_counts_by_return_type[
                return_type
            ],
            "reference_transition_count": reference.transition_event_counts_by_return_type[
                return_type
            ],
            "transition_histogram_jsd": (
                None
                if not generated_counts or not reference_counts
                else histogram_jsd(generated_counts, reference_counts)
            ),
        }

    result: dict[str, Any] = {
        generated_name: generated.summary,
        reference_name: reference.summary,
        "comparisons": {
            "mean_length_error_to_reference": float(
                abs(generated_mean_length - reference_mean_length)
            ),
            "p95_length_error_to_reference": float(
                abs(generated_p95_length - reference_p95_length)
            ),
            "prefix_histogram_jsd": histogram_jsd(
                generated.prefix_counts, reference.prefix_counts
            ),
            "prefix_transition_histogram_jsd": histogram_jsd(
                generated.transition_counts, reference.transition_counts
            ),
            "gap_histogram_jsd": histogram_jsd(
                generated.gap_counts, reference.gap_counts
            ),
        },
        "by_prefix_depth": by_prefix_depth,
        "transition_by_return_type": transition_by_return_type,
    }
    if train_prefix_codes is not None and train_lengths is not None:
        result["comparisons"].update(
            _train_signature_overlap_metrics(
                generated.signature_digests,
                train_prefix_codes,
                train_lengths,
            )
        )
    return result


def _time_values(
    event_time_minutes: torch.Tensor,
    gap_tokens: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    steps = event_time_minutes.shape[1]
    valid = torch.arange(steps)[None, :] < lengths[:, None]
    times = event_time_minutes[valid].numpy().astype(np.float64, copy=False)
    durations = event_time_minutes.clone()
    durations[:, 0] = event_time_minutes[:, 0]
    if steps > 1:
        durations[:, 1:] = (
            event_time_minutes[:, 1:] - event_time_minutes[:, :-1]
        )
    duration_values = durations[valid].numpy().astype(np.float64, copy=False)
    valid_gaps = gap_tokens[valid].numpy()
    by_gap = {
        int(token): duration_values[valid_gaps == token]
        for token in np.unique(valid_gaps)
    }
    return times, duration_values, by_gap


def _continuous_summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {
            "count": 0,
            "mean_minutes": None,
            "std_minutes": None,
            "quantiles_minutes": {
                "p05": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "p95": None,
            },
        }
    return {
        "count": int(values.size),
        "mean_minutes": float(values.mean()),
        "std_minutes": float(values.std()),
        "quantiles_minutes": {
            name: float(np.quantile(values, quantile))
            for name, quantile in (
                ("p05", 0.05),
                ("p25", 0.25),
                ("p50", 0.50),
                ("p75", 0.75),
                ("p95", 0.95),
            )
        },
    }


def _wasserstein_1(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    support = np.unique(np.concatenate((left_sorted, right_sorted)))
    if support.size < 2:
        return 0.0
    intervals = np.diff(support)
    left_cdf = (
        np.searchsorted(left_sorted, support[:-1], side="right")
        / left_sorted.size
    )
    right_cdf = (
        np.searchsorted(right_sorted, support[:-1], side="right")
        / right_sorted.size
    )
    return float(np.sum(np.abs(left_cdf - right_cdf) * intervals))


def _ks_distance(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    support = np.unique(np.concatenate((left_sorted, right_sorted)))
    left_cdf = np.searchsorted(left_sorted, support, side="right") / left.size
    right_cdf = (
        np.searchsorted(right_sorted, support, side="right") / right.size
    )
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _continuous_distribution_pair(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    left_summary = _continuous_summary(left)
    right_summary = _continuous_summary(right)
    quantile_errors: dict[str, float | None] = {}
    for name in ("p05", "p25", "p50", "p75", "p95"):
        left_value = left_summary["quantiles_minutes"][name]
        right_value = right_summary["quantiles_minutes"][name]
        quantile_errors[name] = (
            None
            if left_value is None or right_value is None
            else float(left_value - right_value)
        )
    return {
        left_name: left_summary,
        right_name: right_summary,
        "wasserstein_1_minutes": _wasserstein_1(left, right),
        "ks_distance": _ks_distance(left, right),
        "mean_delta_minutes": (
            None
            if left.size == 0 or right.size == 0
            else float(left.mean() - right.mean())
        ),
        "quantile_delta_minutes": quantile_errors,
    }


def _time_distribution_metrics(
    left_event_time_minutes: torch.Tensor | None,
    left_gap_tokens: torch.Tensor,
    left_lengths: torch.Tensor,
    right_event_time_minutes: torch.Tensor | None,
    right_gap_tokens: torch.Tensor,
    right_lengths: torch.Tensor,
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any] | None:
    if left_event_time_minutes is None or right_event_time_minutes is None:
        return None
    left_times, left_durations, left_by_gap = _time_values(
        left_event_time_minutes,
        left_gap_tokens,
        left_lengths,
    )
    right_times, right_durations, right_by_gap = _time_values(
        right_event_time_minutes,
        right_gap_tokens,
        right_lengths,
    )
    gap_metrics: dict[str, Any] = {}
    for token in sorted(set(left_by_gap) | set(right_by_gap)):
        gap_metrics[str(token)] = _continuous_distribution_pair(
            left_by_gap.get(token, np.empty(0, dtype=np.float64)),
            right_by_gap.get(token, np.empty(0, dtype=np.float64)),
            left_name=left_name,
            right_name=right_name,
        )
    return {
        "event_time_minutes": _continuous_distribution_pair(
            left_times,
            right_times,
            left_name=left_name,
            right_name=right_name,
        ),
        "duration_minutes": _continuous_distribution_pair(
            left_durations,
            right_durations,
            left_name=left_name,
            right_name=right_name,
        ),
        "duration_by_gap_token": gap_metrics,
    }


@dataclass(frozen=True, slots=True)
class _ReferencePanels:
    available_count: int
    reference_count: int
    floor_count: int
    reference_prefix_codes: torch.Tensor
    reference_gap_tokens: torch.Tensor
    reference_lengths: torch.Tensor
    reference_event_time_minutes: torch.Tensor | None
    reference_city_ids: torch.Tensor
    floor_left_prefix_codes: torch.Tensor
    floor_left_gap_tokens: torch.Tensor
    floor_left_lengths: torch.Tensor
    floor_left_event_time_minutes: torch.Tensor | None
    floor_right_prefix_codes: torch.Tensor
    floor_right_gap_tokens: torch.Tensor
    floor_right_lengths: torch.Tensor
    floor_right_event_time_minutes: torch.Tensor | None
    train_prefix_codes: torch.Tensor
    train_lengths: torch.Tensor
    reference_context_features: torch.Tensor | None


def _load_reference_tensors(
    config: OccurrencePlannerSamplesEvaluationConfig,
    checkpoint: OccurrencePlannerCheckpoint,
) -> _ReferencePanels:
    dataset = load_occurrence_planner_dataset(config.reference)
    if not torch.equal(
        dataset.catalog_city_ids, checkpoint.catalog_city_ids
    ) or not torch.equal(
        dataset.city_legal_prefix_mask, checkpoint.city_legal_prefix_mask
    ):
        raise ValueError(
            "planner reference city legality does not match checkpoint"
        )
    if (
        dataset.config.codebook_size != checkpoint.config.codebook_size
        or dataset.config.gap_vocab != checkpoint.config.gap_vocab
        or dataset.config.prefix_depth != checkpoint.config.prefix_depth
        or dataset.config.max_occurrences != checkpoint.config.max_occurrences
        or dataset.config.context_dim != checkpoint.config.context_dim
        or dataset.config.temporal_state_enabled
        != checkpoint.config.temporal_state_enabled
    ):
        raise ValueError(
            "planner reference dataset configuration is incompatible with checkpoint"
        )

    reference_split = getattr(dataset, config.reference_split)
    available_count = reference_split.size
    rng = np.random.default_rng(config.evaluation_seed)
    permutation = rng.permutation(available_count)
    if config.reference_limit is None:
        reference_indices = np.arange(available_count, dtype=np.int64)
        floor_count = available_count // 2
        if floor_count < 1:
            raise ValueError(
                "reference split must contain at least two rows for a real floor"
            )
        floor_left_indices = permutation[:floor_count]
        floor_right_indices = permutation[floor_count : 2 * floor_count]
    else:
        reference_count = config.reference_limit
        if available_count < 2 * reference_count:
            raise ValueError(
                "reference split is too small for two disjoint reference panels "
                f"of size {reference_count}"
            )
        reference_indices = permutation[:reference_count]
        floor_count = reference_count
        floor_left_indices = reference_indices
        floor_right_indices = permutation[
            reference_count : 2 * reference_count
        ]

    def select(tensor: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        return tensor.index_select(0, index_tensor)

    reference_event_time = reference_split.event_time_minutes
    floor_event_time = reference_event_time
    return _ReferencePanels(
        available_count=available_count,
        reference_count=len(reference_indices),
        floor_count=floor_count,
        reference_prefix_codes=select(
            reference_split.prefix_codes, reference_indices
        ),
        reference_gap_tokens=select(
            reference_split.gap_tokens, reference_indices
        ),
        reference_lengths=select(reference_split.lengths, reference_indices),
        reference_event_time_minutes=(
            None
            if reference_event_time is None
            else select(reference_event_time, reference_indices)
        ),
        reference_city_ids=select(reference_split.city_ids, reference_indices),
        floor_left_prefix_codes=select(
            reference_split.prefix_codes, floor_left_indices
        ),
        floor_left_gap_tokens=select(
            reference_split.gap_tokens, floor_left_indices
        ),
        floor_left_lengths=select(reference_split.lengths, floor_left_indices),
        floor_left_event_time_minutes=(
            None
            if floor_event_time is None
            else select(floor_event_time, floor_left_indices)
        ),
        floor_right_prefix_codes=select(
            reference_split.prefix_codes, floor_right_indices
        ),
        floor_right_gap_tokens=select(
            reference_split.gap_tokens, floor_right_indices
        ),
        floor_right_lengths=select(
            reference_split.lengths, floor_right_indices
        ),
        floor_right_event_time_minutes=(
            None
            if floor_event_time is None
            else select(floor_event_time, floor_right_indices)
        ),
        train_prefix_codes=dataset.train.prefix_codes,
        train_lengths=dataset.train.lengths,
        reference_context_features=(
            None
            if reference_split.context_features is None
            else select(reference_split.context_features, reference_indices)
        ),
    )


def _validate_sample_payload(
    payload: Mapping[str, object],
    *,
    checkpoint: OccurrencePlannerCheckpoint,
) -> dict[str, Any]:
    config = checkpoint.config

    prefix_codes = _require_tensor(payload, "prefix_codes", dtype=torch.long)
    gap_tokens = _require_tensor(payload, "gap_tokens", dtype=torch.long)
    lengths = _require_tensor(payload, "lengths", dtype=torch.long)
    model_log_probability = _require_tensor(
        payload, "model_log_probability", dtype=torch.float32
    )
    sampling_log_probability = _require_tensor(
        payload, "sampling_log_probability", dtype=torch.float32
    )
    length_was_sampled = _require_tensor(
        payload, "length_was_sampled", dtype=torch.bool
    )
    city_ids = _require_tensor(payload, "city_ids", dtype=torch.long)
    context_schema = payload["context_schema"]
    if context_schema != checkpoint.config.context_schema:
        raise ValueError(
            "planner samples context_schema differs from checkpoint"
        )
    context_features = payload["context_features"]
    if checkpoint.config.context_dim == 0:
        if context_features is not None:
            raise ValueError(
                "planner samples context must be absent when context_dim=0"
            )
    else:
        if not isinstance(context_features, torch.Tensor):
            raise ValueError(
                "planner samples context_features must be a tensor"
            )
        if (
            context_features.device.type != "cpu"
            or context_features.dtype != torch.float32
            or context_features.shape
            != (prefix_codes.shape[0], checkpoint.config.context_dim)
        ):
            raise ValueError(
                "planner samples context_features have the wrong contract"
            )
        if not bool(torch.isfinite(context_features).all()):
            raise ValueError("planner samples context_features must be finite")

    if prefix_codes.ndim != 3:
        raise ValueError(
            "planner samples prefix_codes must have shape [N, S, P]"
        )
    sample_count, steps, prefix_depth = prefix_codes.shape
    if sample_count < 1:
        raise ValueError("planner samples must contain at least one sample")
    if prefix_depth != config.prefix_depth or steps > config.max_occurrences:
        raise ValueError(
            "planner samples prefix_codes shape is incompatible with checkpoint"
        )
    if gap_tokens.shape != (sample_count, steps):
        raise ValueError(
            "planner samples gap_tokens shape must match prefix_codes"
        )
    if lengths.shape != (sample_count,):
        raise ValueError("planner samples lengths shape must be [N]")
    if model_log_probability.shape != (sample_count, 4):
        raise ValueError(
            "planner samples model_log_probability shape must be [N, 4]"
        )
    if sampling_log_probability.shape != (sample_count, 4):
        raise ValueError(
            "planner samples sampling_log_probability shape must be [N, 4]"
        )
    if length_was_sampled.shape != (sample_count,):
        raise ValueError(
            "planner samples length_was_sampled shape must be [N]"
        )
    if city_ids.shape != (sample_count,):
        raise ValueError("planner samples city_ids shape must be [N]")
    known_city = torch.isin(city_ids, checkpoint.catalog_city_ids)
    if not bool(known_city.all()):
        raise ValueError("planner samples contain unknown city_ids")

    if bool((lengths < 1).any()) or bool((lengths > steps).any()):
        raise ValueError(
            "planner samples lengths must be in [1, max_occurrences]"
        )
    for name, tensor in (
        ("model_log_probability", model_log_probability),
        ("sampling_log_probability", sampling_log_probability),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"planner samples {name} must be finite")

    positions = torch.arange(steps).unsqueeze(0)
    valid_positions = positions < lengths.unsqueeze(1)
    if bool((prefix_codes[~valid_positions] != 0).any()):
        raise ValueError("planner samples padded prefix_codes must be zero")
    if bool((gap_tokens[~valid_positions] != 0).any()):
        raise ValueError("planner samples padded gap_tokens must be zero")
    valid_prefixes = prefix_codes[valid_positions]
    if bool((valid_prefixes < 0).any()) or bool(
        (valid_prefixes >= config.codebook_size).any()
    ):
        raise ValueError(
            "planner samples valid prefix codes exceed checkpoint vocabulary"
        )
    valid_gaps = gap_tokens[valid_positions]
    if bool((valid_gaps < 0).any()) or bool(
        (valid_gaps >= config.gap_vocab).any()
    ):
        raise ValueError(
            "planner samples valid gap tokens exceed checkpoint vocabulary"
        )

    legal_prefixes = {
        tuple(int(value) for value in row)
        for row in checkpoint.legal_prefixes.tolist()
    }
    invalid_prefix = next(
        (
            tuple(int(value) for value in row)
            for row in valid_prefixes.tolist()
            if tuple(int(value) for value in row) not in legal_prefixes
        ),
        None,
    )
    if invalid_prefix is not None:
        raise ValueError(
            f"planner samples contain illegal prefix tuple: {invalid_prefix}"
        )

    city_positions = torch.searchsorted(checkpoint.catalog_city_ids, city_ids)
    row_catalog_mask = checkpoint.city_legal_prefix_mask[city_positions]
    flat_valid_rows = torch.arange(sample_count)[:, None].expand(
        sample_count, steps
    )[valid_positions]
    legality = checkpoint.model.legality
    valid_token_ids = legality.token_ids(valid_prefixes)
    if not bool(
        row_catalog_mask[flat_valid_rows]
        .gather(-1, valid_token_ids[:, None])
        .squeeze(-1)
        .all()
    ):
        raise ValueError(
            "planner samples contain a prefix illegal for its city"
        )

    event_time = payload["event_time_minutes"]
    event_time_monotonic_rate: float | None
    if config.temporal_state_enabled:
        if not isinstance(event_time, torch.Tensor):
            raise ValueError(
                "planner samples event_time_minutes is required by checkpoint"
            )
        if (
            event_time.device.type != "cpu"
            or event_time.dtype != torch.float32
            or event_time.shape != (sample_count, steps)
        ):
            raise ValueError(
                "planner samples event_time_minutes must be a CPU float32 tensor "
                "with shape [N, S]"
            )
        if not bool(torch.isfinite(event_time).all()):
            raise ValueError(
                "planner samples event_time_minutes must be finite"
            )
        if bool((event_time[~valid_positions] != 0).any()):
            raise ValueError(
                "planner samples padded event_time_minutes must be zero"
            )
        valid_times = event_time[valid_positions]
        observation_window = config.observation_window_minutes
        if observation_window is None:
            raise ValueError(
                "temporal checkpoint is missing observation_window_minutes"
            )
        if bool((valid_times < 0).any()) or bool(
            (valid_times > observation_window).any()
        ):
            raise ValueError(
                "planner samples event_time_minutes exceeds the checkpoint horizon"
            )
        monotonic_rows = []
        for row, length in enumerate(lengths.tolist()):
            row_times = event_time[row, : int(length)]
            monotonic_rows.append(
                int(length) < 2
                or bool((row_times[1:] >= row_times[:-1]).all())
            )
        event_time_monotonic_rate = sum(monotonic_rows) / sample_count
        if event_time_monotonic_rate != 1.0:
            raise ValueError(
                "planner samples event_time_minutes must be monotonic"
            )
    elif event_time is not None:
        raise ValueError(
            "planner samples event_time_minutes must be absent for a non-temporal checkpoint"
        )
    else:
        event_time_monotonic_rate = None

    seed = payload["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("planner samples seed must be an integer")

    temperatures = {
        key: _require_finite_positive(payload, key)
        for key in (
            "temperature",
            "length_temperature",
            "gap_temperature",
            "prefix_temperature",
            "route_temperature",
        )
    }
    lengths_float = lengths.to(torch.float64)
    return {
        "artifact_type": SAMPLES_ARTIFACT_TYPE,
        "sample_count": sample_count,
        "prefix_shape": list(prefix_codes.shape),
        "mean_length": float(lengths_float.mean().item()),
        "min_length": int(lengths.min().item()),
        "max_length": int(lengths.max().item()),
        "legal_prefix_rate": 1.0,
        "event_time_monotonic_rate": event_time_monotonic_rate,
        "finite_model_log_probability": True,
        "finite_sampling_log_probability": True,
        "checkpoint_epoch": checkpoint.epoch,
        "checkpoint_validation_metric": checkpoint.validation_metric,
        "seed": seed,
        **temperatures,
    }


def _router_diagnostics(
    checkpoint: OccurrencePlannerCheckpoint,
    panels: _ReferencePanels,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any] | None:
    model = checkpoint.model
    if model.mobility_router is None:
        return None
    names = list(model.mobility_router.expert_names)
    was_training = model.training
    original_device = next(model.parameters()).device

    retrieve_parts: list[torch.Tensor] = []
    responsibility_parts: list[torch.Tensor] = []
    mechanism_parts: list[torch.Tensor] = []
    history_available_parts: list[torch.Tensor] = []
    gap_parts: list[torch.Tensor] = []
    seen_before_parts: list[torch.Tensor] = []
    retrieval_target_probability_parts: list[torch.Tensor] = []
    distinct_parts: list[torch.Tensor] = []
    return_type_parts: dict[str, list[torch.Tensor]] = {
        "first": [],
        "explore": [],
        "immediate_return": [],
        "older_return": [],
    }
    frequency_parts: list[torch.Tensor] = []
    recency_steps_parts: list[torch.Tensor] = []
    valid_time_parts: list[torch.Tensor] = []
    retrieval_rank_parts: list[torch.Tensor] = []
    recency_minutes_parts: list[torch.Tensor] = []

    def move_slice(
        value: torch.Tensor | None, start: int, stop: int
    ) -> torch.Tensor | None:
        if value is None:
            return None
        return value[start:stop].to(device)

    model.to(device)
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, panels.reference_count, batch_size):
                stop = min(start + batch_size, panels.reference_count)
                prefix_codes = move_slice(
                    panels.reference_prefix_codes, start, stop
                )
                gap_tokens = move_slice(
                    panels.reference_gap_tokens, start, stop
                )
                lengths = move_slice(panels.reference_lengths, start, stop)
                assert prefix_codes is not None
                assert gap_tokens is not None
                assert lengths is not None
                batch = collate_occurrence_tensors(
                    prefix_codes,
                    gap_tokens,
                    lengths,
                    config=checkpoint.config,
                    legality=model.legality,
                    event_time_minutes=move_slice(
                        panels.reference_event_time_minutes, start, stop
                    ),
                    context_features=move_slice(
                        panels.reference_context_features, start, stop
                    ),
                    city_ids=move_slice(
                        panels.reference_city_ids, start, stop
                    ),
                )
                output = model(batch)
                if checkpoint.config.temporal_state_enabled:
                    assert batch.previous_event_time_minutes is not None
                    arrival_time_minutes = model._reconstruct_arrival_time(
                        batch.previous_event_time_minutes,
                        batch.target_gap_tokens,
                    )
                    conditioned = model.condition_on_gap_and_time(
                        output.hidden,
                        batch.target_gap_tokens,
                        arrival_time_minutes,
                    )
                else:
                    arrival_time_minutes = None
                    conditioned = model.condition_on_gap(
                        output.hidden,
                        batch.target_gap_tokens,
                    )
                memory_states = model._embed_retrieval_memory(
                    batch.previous_prefix_codes,
                    batch.previous_gap_tokens,
                    batch.previous_event_time_minutes,
                )
                routing = model.mobility_router(
                    conditioned,
                    memory_states,
                    batch.previous_prefix_codes,
                )
                if (
                    output.retrieve_probability is None
                    or output.retrieve_responsibility is None
                    or output.retrieval_mechanism_weights is None
                ):
                    raise RuntimeError(
                        "mobility router omitted required diagnostics"
                    )

                valid = batch.valid_mask
                retrieve_parts.append(
                    output.retrieve_probability.float()[valid].cpu()
                )
                responsibility_parts.append(
                    output.retrieve_responsibility.float()[valid].cpu()
                )
                mechanism_parts.append(
                    output.retrieval_mechanism_weights.float()[valid].cpu()
                )
                history_available_parts.append(
                    routing.history_mask.any(dim=-1)[valid].cpu()
                )
                gap_parts.append(batch.target_gap_tokens[valid].cpu())

                chunk_size, steps = batch.target_prefix_codes.shape[:2]
                indices = torch.arange(steps, device=device)
                memory_valid = (batch.previous_prefix_codes >= 0).all(dim=-1)
                causal = indices[None, :, None] >= indices[None, None, :]
                target_matches_all = (
                    batch.target_prefix_codes[:, :, None, :]
                    == batch.previous_prefix_codes[:, None, :, :]
                ).all(dim=-1)
                target_matches_all = (
                    target_matches_all & memory_valid[:, None, :] & causal
                )
                seen_before_matrix = target_matches_all.any(dim=-1)
                latest_match_index = (
                    torch.where(
                        target_matches_all,
                        indices[None, None, :],
                        torch.full(
                            (chunk_size, steps, steps),
                            -1,
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                    .max(dim=-1)
                    .values
                )

                immediate_matrix = torch.zeros_like(valid)
                if steps > 1:
                    immediate_matrix[:, 1:] = (
                        batch.target_prefix_codes[:, 1:]
                        == batch.target_prefix_codes[:, :-1]
                    ).all(dim=-1)
                older_return_matrix = seen_before_matrix & ~immediate_matrix
                first_matrix = torch.zeros_like(valid)
                first_matrix[:, 0] = valid[:, 0]
                explore_matrix = valid & ~seen_before_matrix & ~first_matrix

                target_frequency_matrix = target_matches_all.sum(dim=-1)
                query_index = indices[None, :].expand(chunk_size, -1)
                recency_steps_matrix = torch.where(
                    seen_before_matrix,
                    query_index - latest_match_index + 1,
                    torch.zeros_like(query_index),
                )
                if arrival_time_minutes is not None:
                    assert batch.previous_event_time_minutes is not None
                    safe_latest = latest_match_index.clamp_min(0)
                    latest_time = batch.previous_event_time_minutes.gather(
                        -1, safe_latest
                    )
                    recency_minutes_matrix = torch.where(
                        seen_before_matrix,
                        arrival_time_minutes - latest_time,
                        torch.zeros_like(arrival_time_minutes),
                    )
                    valid_time_parts.append(arrival_time_minutes[valid].cpu())
                    recency_minutes_parts.append(
                        recency_minutes_matrix[valid][
                            seen_before_matrix[valid]
                        ].cpu()
                    )

                target_latest_matches = (
                    batch.target_prefix_codes[:, :, None, :]
                    == batch.previous_prefix_codes[:, None, :, :]
                ).all(dim=-1) & routing.history_mask
                retrieval_target_log_probability = torch.logsumexp(
                    routing.retrieval_log_probability.masked_fill(
                        ~target_latest_matches,
                        float("-inf"),
                    ),
                    dim=-1,
                )
                has_retrieval_target = target_latest_matches.any(dim=-1)
                better_candidates = (
                    routing.retrieval_log_probability
                    > retrieval_target_log_probability[..., None]
                ) & routing.history_mask
                retrieval_rank_matrix = torch.where(
                    has_retrieval_target,
                    1 + better_candidates.sum(dim=-1),
                    torch.zeros_like(batch.target_gap_tokens),
                )

                seen_before_parts.append(seen_before_matrix[valid].cpu())
                retrieval_target_probability_parts.append(
                    torch.exp(retrieval_target_log_probability)[valid].cpu()
                )
                return_type_parts["first"].append(first_matrix[valid].cpu())
                return_type_parts["explore"].append(
                    explore_matrix[valid].cpu()
                )
                return_type_parts["immediate_return"].append(
                    immediate_matrix[valid].cpu()
                )
                return_type_parts["older_return"].append(
                    older_return_matrix[valid].cpu()
                )
                frequency_parts.append(target_frequency_matrix[valid].cpu())
                recency_steps_parts.append(recency_steps_matrix[valid].cpu())
                retrieval_rank_parts.append(retrieval_rank_matrix[valid].cpu())

                target_prefixes_cpu = batch.target_prefix_codes.cpu()
                distinct_values: list[int] = []
                for row, length in enumerate(batch.lengths.cpu().tolist()):
                    seen: set[tuple[int, ...]] = set()
                    for step in range(int(length)):
                        distinct_values.append(len(seen))
                        seen.add(
                            tuple(
                                int(value)
                                for value in target_prefixes_cpu[
                                    row, step
                                ].tolist()
                            )
                        )
                distinct_parts.append(
                    torch.tensor(distinct_values, dtype=torch.long)
                )
    finally:
        model.to(original_device)
        model.train(was_training)

    retrieve = torch.cat(retrieve_parts)
    responsibility = torch.cat(responsibility_parts)
    mechanisms = torch.cat(mechanism_parts)
    history_available = torch.cat(history_available_parts)
    gaps = torch.cat(gap_parts)
    seen_before_flat = torch.cat(seen_before_parts)
    retrieval_target_probability_flat = torch.cat(
        retrieval_target_probability_parts
    )
    distinct = torch.cat(distinct_parts)
    return_type_flat = {
        name: torch.cat(parts) for name, parts in return_type_parts.items()
    }
    frequency_flat = torch.cat(frequency_parts)
    recency_steps_flat = torch.cat(recency_steps_parts)
    retrieval_rank_flat = torch.cat(retrieval_rank_parts)

    def mechanism_means(mask: torch.Tensor) -> dict[str, float | None]:
        eligible = mask & history_available
        if not bool(eligible.any()):
            return {name: None for name in names}
        values = mechanisms[eligible]
        return {
            name: float(values[:, index].mean().item())
            for index, name in enumerate(names)
        }

    def routed_summary(mask: torch.Tensor) -> dict[str, Any]:
        if not bool(mask.any()):
            return {
                "event_count": 0,
                "mean_explore_probability": None,
                "mean_retrieve_probability": None,
                "mean_retrieve_responsibility": None,
                "mean_retrieval_target_probability": None,
                "mechanism_weights": {name: None for name in names},
            }
        seen_mask = mask & seen_before_flat
        return {
            "event_count": int(mask.sum().item()),
            "mean_explore_probability": float(
                (1.0 - retrieve[mask]).mean().item()
            ),
            "mean_retrieve_probability": float(retrieve[mask].mean().item()),
            "mean_retrieve_responsibility": float(
                responsibility[mask].mean().item()
            ),
            "mean_retrieval_target_probability": (
                None
                if not bool(seen_mask.any())
                else float(
                    retrieval_target_probability_flat[seen_mask].mean().item()
                )
            ),
            "mechanism_weights": mechanism_means(mask),
        }

    by_distinct: dict[str, Any] = {}
    for value in torch.unique(distinct, sorted=True).tolist():
        mask = distinct == int(value)
        by_distinct[str(int(value))] = routed_summary(mask)

    by_gap: dict[str, Any] = {}
    for token in torch.unique(gaps, sorted=True).tolist():
        mask = gaps == int(token)
        by_gap[str(int(token))] = routed_summary(mask)

    by_return_type = {
        name: routed_summary(mask) for name, mask in return_type_flat.items()
    }

    by_target_frequency: dict[str, Any] = {}
    for value in torch.unique(frequency_flat, sorted=True).tolist():
        mask = frequency_flat == int(value)
        by_target_frequency[str(int(value))] = routed_summary(mask)

    by_recency_steps: dict[str, Any] = {}
    for value in torch.unique(
        recency_steps_flat[recency_steps_flat > 0],
        sorted=True,
    ).tolist():
        mask = recency_steps_flat == int(value)
        by_recency_steps[str(int(value))] = routed_summary(mask)

    by_day_of_week: dict[str, Any] | None = None
    by_hour_of_day: dict[str, Any] | None = None
    if valid_time_parts:
        valid_times = torch.cat(valid_time_parts)
        week_minutes = 7.0 * 24.0 * 60.0
        day = torch.floor(
            torch.remainder(valid_times, week_minutes) / (24.0 * 60.0)
        ).to(torch.long)
        hour = torch.floor(
            torch.remainder(valid_times, 24.0 * 60.0) / 60.0
        ).to(torch.long)
        by_day_of_week = {
            str(int(value)): routed_summary(day == int(value))
            for value in torch.unique(day, sorted=True).tolist()
        }
        by_hour_of_day = {
            str(int(value)): routed_summary(hour == int(value))
            for value in torch.unique(hour, sorted=True).tolist()
        }

    seen_rank = retrieval_rank_flat[retrieval_rank_flat > 0]
    retrieval_rank: dict[str, Any]
    if seen_rank.numel() == 0:
        retrieval_rank = {
            "seen_target_count": 0,
            "mean_rank": None,
            "median_rank": None,
            "top1_rate": None,
            "top3_rate": None,
        }
    else:
        retrieval_rank = {
            "seen_target_count": int(seen_rank.numel()),
            "mean_rank": float(seen_rank.float().mean().item()),
            "median_rank": float(seen_rank.float().median().item()),
            "top1_rate": float((seen_rank <= 1).float().mean().item()),
            "top3_rate": float((seen_rank <= 3).float().mean().item()),
        }

    recency_minutes: dict[str, Any] | None = None
    if valid_time_parts:
        values = torch.cat(recency_minutes_parts)
        recency_minutes = (
            {
                "seen_target_count": 0,
                "mean_minutes": None,
                "median_minutes": None,
            }
            if values.numel() == 0
            else {
                "seen_target_count": int(values.numel()),
                "mean_minutes": float(values.float().mean().item()),
                "median_minutes": float(values.float().median().item()),
            }
        )

    all_events = torch.ones_like(retrieve, dtype=torch.bool)
    return {
        **routed_summary(all_events),
        "history_available_event_count": int(history_available.sum().item()),
        "seen_prefix_rate": float(seen_before_flat.float().mean().item()),
        "retrieval_rank": retrieval_rank,
        "retrieval_recency_minutes": recency_minutes,
        "by_return_type": by_return_type,
        "by_distinct_prefix_count": by_distinct,
        "by_target_prior_frequency": by_target_frequency,
        "by_target_recency_steps": by_recency_steps,
        "by_gap_token": by_gap,
        "by_day_of_week": by_day_of_week,
        "by_hour_of_day": by_hour_of_day,
    }


def evaluate_occurrence_planner_samples(
    config: OccurrencePlannerSamplesEvaluationConfig,
) -> dict[str, Any]:
    config.validate()
    diagnostic_device = _resolve_router_diagnostics_device(config.device)
    payload = _load_samples(config.samples)
    checkpoint = load_occurrence_planner_checkpoint(config.checkpoint)
    report = _validate_sample_payload(
        payload,
        checkpoint=checkpoint,
    )
    sample_prefix_codes = _require_tensor(
        payload, "prefix_codes", dtype=torch.long
    )
    sample_gap_tokens = _require_tensor(
        payload, "gap_tokens", dtype=torch.long
    )
    sample_lengths = _require_tensor(payload, "lengths", dtype=torch.long)
    panels = _load_reference_tensors(config, checkpoint)
    sample_event_time = payload["event_time_minutes"]
    if not isinstance(sample_event_time, torch.Tensor):
        sample_event_time = None
    generated_stats = _population_stats(
        sample_prefix_codes,
        sample_gap_tokens,
        sample_lengths,
        codebook_size=checkpoint.config.codebook_size,
    )
    reference_stats = _population_stats(
        panels.reference_prefix_codes,
        panels.reference_gap_tokens,
        panels.reference_lengths,
        codebook_size=checkpoint.config.codebook_size,
    )
    population_metrics = _population_metrics(
        generated_stats,
        reference_stats,
        train_prefix_codes=panels.train_prefix_codes,
        train_lengths=panels.train_lengths,
    )
    population_metrics["time_distributions"] = _time_distribution_metrics(
        sample_event_time,
        sample_gap_tokens,
        sample_lengths,
        panels.reference_event_time_minutes,
        panels.reference_gap_tokens,
        panels.reference_lengths,
        left_name="generated",
        right_name="reference",
    )
    floor_left_stats = _population_stats(
        panels.floor_left_prefix_codes,
        panels.floor_left_gap_tokens,
        panels.floor_left_lengths,
        codebook_size=checkpoint.config.codebook_size,
    )
    floor_right_stats = _population_stats(
        panels.floor_right_prefix_codes,
        panels.floor_right_gap_tokens,
        panels.floor_right_lengths,
        codebook_size=checkpoint.config.codebook_size,
    )
    real_floor = _population_metrics(
        floor_left_stats,
        floor_right_stats,
        generated_name="real_left",
        reference_name="real_right",
    )
    real_floor["time_distributions"] = _time_distribution_metrics(
        panels.floor_left_event_time_minutes,
        panels.floor_left_gap_tokens,
        panels.floor_left_lengths,
        panels.floor_right_event_time_minutes,
        panels.floor_right_gap_tokens,
        panels.floor_right_lengths,
        left_name="real_left",
        right_name="real_right",
    )
    report["population_metrics"] = population_metrics
    report["real_vs_real_floor"] = real_floor
    report["router_diagnostics"] = _router_diagnostics(
        checkpoint,
        panels,
        device=diagnostic_device,
        batch_size=config.batch_size,
    )
    report.update(
        {
            "samples": str(config.samples),
            "checkpoint": str(config.checkpoint),
            "reference": str(config.reference),
            "reference_split": config.reference_split,
            "reference_count": panels.reference_count,
            "reference_available_count": panels.available_count,
            "reference_limit": config.reference_limit,
            "evaluation_seed": config.evaluation_seed,
            "router_diagnostics_device": str(diagnostic_device),
            "router_diagnostics_batch_size": config.batch_size,
            "reference_sampling": (
                "full_split"
                if config.reference_limit is None
                else "seeded_without_replacement"
            ),
            "real_floor_panel_count": panels.floor_count,
            "output": str(config.output),
        }
    )
    _atomic_json(report, config.output)
    return report


__all__ = [
    "OccurrencePlannerSamplesEvaluationConfig",
    "evaluate_occurrence_planner_samples",
]
