from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch

from semoflow.metrics.divergence import jensen_shannon_divergence
from semoflow.planner.artifacts import (
    SAMPLES_ARTIFACT_TYPE,
    OccurrencePlannerCheckpoint,
    OccurrencePlannerSplit,
    load_occurrence_planner_checkpoint,
    load_occurrence_planner_dataset,
)
from semoflow.planner.data import collate_occurrence_tensors
from semoflow.planner.types import OccurrenceBatch
from semoflow.training.utils.runtime import parse_device_spec
from semoflow.utils.serialization import write_json_atomic

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


_PANEL_FIELDS = (
    "prefix_codes",
    "gap_tokens",
    "lengths",
    "event_time_minutes",
    "city_ids",
    "context_features",
)


@dataclass(frozen=True, slots=True)
class _Panel:
    """Padded occurrence plans: prefixes [N, S, P]; gaps and times [N, S]."""

    prefix_codes: torch.Tensor
    gap_tokens: torch.Tensor
    lengths: torch.Tensor
    event_time_minutes: torch.Tensor | None
    city_ids: torch.Tensor
    context_features: torch.Tensor | None

    @classmethod
    def of(cls, split: OccurrencePlannerSplit) -> _Panel:
        return cls(**{name: getattr(split, name) for name in _PANEL_FIELDS})

    @property
    def size(self) -> int:
        return int(self.lengths.shape[0])

    def map(self, transform: Callable[[torch.Tensor], torch.Tensor]) -> _Panel:
        values = (getattr(self, name) for name in _PANEL_FIELDS)
        return _Panel(
            *(None if value is None else transform(value) for value in values)
        )

    def chunk(self, start: int, size: int, *, device: torch.device) -> _Panel:
        return self.map(lambda tensor: tensor[start : start + size].to(device))

    def rows(self, indices: np.ndarray) -> _Panel:
        index = torch.as_tensor(indices, dtype=torch.long)
        return self.map(lambda tensor: tensor.index_select(0, index))


def _validate_context(
    payload: Mapping[str, object],
    config: Any,
    *,
    rows: int,
) -> torch.Tensor | None:
    if payload["context_schema"] != config.context_schema:
        raise ValueError(
            "planner samples context_schema differs from checkpoint"
        )
    context_features = payload["context_features"]
    if config.context_dim == 0:
        if context_features is not None:
            raise ValueError(
                "planner samples context must be absent when context_dim=0"
            )
        return None
    if not isinstance(context_features, torch.Tensor):
        raise ValueError("planner samples context_features must be a tensor")
    if (
        context_features.device.type != "cpu"
        or context_features.dtype != torch.float32
        or context_features.shape != (rows, config.context_dim)
    ):
        raise ValueError(
            "planner samples context_features have the wrong contract"
        )
    if not bool(torch.isfinite(context_features).all()):
        raise ValueError("planner samples context_features must be finite")
    return context_features


def _validate_codes(
    prefix_codes: torch.Tensor,
    gap_tokens: torch.Tensor,
    valid: torch.Tensor,
    config: Any,
) -> None:
    if bool((prefix_codes[~valid] != 0).any()):
        raise ValueError("planner samples padded prefix_codes must be zero")
    if bool((gap_tokens[~valid] != 0).any()):
        raise ValueError("planner samples padded gap_tokens must be zero")
    valid_prefixes = prefix_codes[valid]
    if bool((valid_prefixes < 0).any()) or bool(
        (valid_prefixes >= config.codebook_size).any()
    ):
        raise ValueError(
            "planner samples valid prefix codes exceed checkpoint vocabulary"
        )
    valid_gaps = gap_tokens[valid]
    if bool((valid_gaps < 0).any()) or bool(
        (valid_gaps >= config.gap_vocab).any()
    ):
        raise ValueError(
            "planner samples valid gap tokens exceed checkpoint vocabulary"
        )


def _validate_legality(
    prefix_codes: torch.Tensor,
    city_ids: torch.Tensor,
    valid: torch.Tensor,
    checkpoint: OccurrencePlannerCheckpoint,
) -> None:
    valid_prefixes = prefix_codes[valid]
    legal = set(map(tuple, checkpoint.legal_prefixes.tolist()))
    illegal = next(
        (
            prefix
            for prefix in map(tuple, valid_prefixes.tolist())
            if prefix not in legal
        ),
        None,
    )
    if illegal is not None:
        raise ValueError(
            f"planner samples contain illegal prefix tuple: {illegal}"
        )
    catalog_rows = torch.searchsorted(checkpoint.catalog_city_ids, city_ids)
    city_mask = checkpoint.city_legal_prefix_mask[catalog_rows]
    sample_rows = torch.arange(len(city_ids))[:, None].expand_as(valid)
    token_ids = checkpoint.model.legality.token_ids(valid_prefixes)
    if not bool(city_mask[sample_rows[valid], token_ids].all()):
        raise ValueError(
            "planner samples contain a prefix illegal for its city"
        )


def _validate_event_times(
    event_time: object,
    config: Any,
    valid: torch.Tensor,
) -> torch.Tensor | None:
    if not config.temporal_state_enabled:
        if event_time is not None:
            raise ValueError(
                "planner samples event_time_minutes must be absent for a non-temporal checkpoint"
            )
        return None
    if not isinstance(event_time, torch.Tensor):
        raise ValueError(
            "planner samples event_time_minutes is required by checkpoint"
        )
    if (
        event_time.device.type != "cpu"
        or event_time.dtype != torch.float32
        or event_time.shape != valid.shape
    ):
        raise ValueError(
            "planner samples event_time_minutes must be a CPU float32 tensor "
            "with shape [N, S]"
        )
    if not bool(torch.isfinite(event_time).all()):
        raise ValueError("planner samples event_time_minutes must be finite")
    if bool((event_time[~valid] != 0).any()):
        raise ValueError(
            "planner samples padded event_time_minutes must be zero"
        )
    valid_times = event_time[valid]
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
    ordered = (event_time[:, 1:] >= event_time[:, :-1]) | ~valid[:, 1:]
    if not bool(ordered.all()):
        raise ValueError(
            "planner samples event_time_minutes must be monotonic"
        )
    return event_time


def _validate_sample_payload(
    payload: Mapping[str, object],
    *,
    checkpoint: OccurrencePlannerCheckpoint,
) -> tuple[_Panel, dict[str, Any]]:
    """Check the samples against the checkpoint; return them and a summary."""
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
    context_features = _validate_context(
        payload, config, rows=prefix_codes.shape[0]
    )

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
    for name, tensor, shape, expected in (
        ("gap_tokens", gap_tokens, (sample_count, steps), "match prefix_codes"),
        ("lengths", lengths, (sample_count,), "be [N]"),
        ("model_log_probability", model_log_probability, (sample_count, 4), "be [N, 4]"),
        ("sampling_log_probability", sampling_log_probability, (sample_count, 4), "be [N, 4]"),
        ("length_was_sampled", length_was_sampled, (sample_count,), "be [N]"),
        ("city_ids", city_ids, (sample_count,), "be [N]"),
    ):  # fmt: skip
        if tensor.shape != shape:
            raise ValueError(f"planner samples {name} shape must {expected}")
    if not bool(torch.isin(city_ids, checkpoint.catalog_city_ids).all()):
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

    valid = torch.arange(steps).unsqueeze(0) < lengths.unsqueeze(1)
    _validate_codes(prefix_codes, gap_tokens, valid, config)
    _validate_legality(prefix_codes, city_ids, valid, checkpoint)
    event_time = _validate_event_times(
        payload["event_time_minutes"], config, valid
    )

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
    samples = _Panel(
        prefix_codes=prefix_codes,
        gap_tokens=gap_tokens,
        lengths=lengths,
        event_time_minutes=event_time,
        city_ids=city_ids,
        context_features=context_features,
    )
    report = {
        "artifact_type": SAMPLES_ARTIFACT_TYPE,
        "sample_count": sample_count,
        "prefix_shape": list(prefix_codes.shape),
        "mean_length": float(lengths.to(torch.float64).mean().item()),
        "min_length": int(lengths.min().item()),
        "max_length": int(lengths.max().item()),
        "legal_prefix_rate": 1.0,
        "event_time_monotonic_rate": None if event_time is None else 1.0,
        "finite_model_log_probability": True,
        "finite_sampling_log_probability": True,
        "checkpoint_epoch": checkpoint.epoch,
        "checkpoint_validation_metric": checkpoint.validation_metric,
        "seed": seed,
        **temperatures,
    }
    return samples, report


_Prefix = tuple[int, ...]
_Signature = tuple[_Prefix, ...]
_Transition = tuple[_Prefix, _Prefix]
_RETURN_TYPES = ("explore", "immediate_return", "older_return")
# Population summary entries taken from the full-depth prefix scale.
_FULL_SCALE_SUMMARY_KEYS = (
    "mean_repeat_mass",
    "std_repeat_mass",
    "p05_repeat_mass",
    "median_repeat_mass",
    "p95_repeat_mass",
    "mean_unique_prefixes",
    "unique_complete_plan_rate",
    "adjacent_repeat_share",
    "non_adjacent_prefix_return_rate",
    "non_adjacent_prefix_return_event_count",
    "codebook_entropy_bits",
)


def _signatures(panel: _Panel) -> list[_Signature]:
    """Each plan's valid prefixes as hashable tuples."""
    codes = panel.prefix_codes.numpy()
    return [
        tuple(map(tuple, codes[row, :length].tolist()))
        for row, length in enumerate(panel.lengths.tolist())
    ]


def _signature_digest(signature: _Signature) -> bytes:
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


def _quantile_summary(values: np.ndarray, name: str) -> dict[str, float]:
    return {
        f"mean_{name}": float(values.mean()),
        f"std_{name}": float(values.std()),
        f"p05_{name}": float(np.quantile(values, 0.05)),
        f"median_{name}": float(np.median(values)),
        f"p95_{name}": float(np.quantile(values, 0.95)),
    }


@dataclass(frozen=True, slots=True)
class _PrefixScaleStats:
    """Plan statistics with every prefix truncated to its first codes."""

    summary: dict[str, Any]
    prefix_counts: Counter[_Prefix]
    transition_counts: Counter[_Transition]
    signature_digests: tuple[bytes, ...]
    repeat_mass: np.ndarray


def _prefix_scale_stats(
    signatures: list[_Signature],
    *,
    depth: int,
    codebook_size: int,
    level_entropy_bits: list[float],
) -> _PrefixScaleStats:
    repeat_mass = np.empty(len(signatures), dtype=np.float64)
    unique_prefixes = np.empty(len(signatures), dtype=np.float64)
    prefix_counts: Counter[_Prefix] = Counter()
    transition_counts: Counter[_Transition] = Counter()
    signature_digests: list[bytes] = []
    adjacent_repeat_count = 0
    adjacent_pair_count = 0
    non_adjacent_return_count = 0
    event_count = 0
    for row, signature in enumerate(signatures):
        length = len(signature)
        unique = len(set(signature))
        prefix_counts.update(signature)
        transition_counts.update(pairwise(signature))
        signature_digests.append(_signature_digest(signature))
        repeat_mass[row] = (length - unique) / length
        unique_prefixes[row] = unique
        event_count += length
        adjacent_pair_count += max(length - 1, 0)
        adjacent_repeat_count += sum(
            left == right for left, right in pairwise(signature)
        )
        seen: set[_Prefix] = set()
        previous: _Prefix | None = None
        for value in signature:
            if value in seen and value != previous:
                non_adjacent_return_count += 1
            seen.add(value)
            previous = value

    summary = {
        "prefix_depth": depth,
        "event_count": event_count,
        "unique_prefix_count": len(prefix_counts),
        "prefix_utilization": len(prefix_counts) / (codebook_size**depth),
        **_quantile_summary(repeat_mass, "repeat_mass"),
        "mean_unique_prefixes": float(unique_prefixes.mean()),
        "adjacent_repeat_share": (
            adjacent_repeat_count / adjacent_pair_count
            if adjacent_pair_count
            else 0.0
        ),
        "non_adjacent_prefix_return_rate": (
            non_adjacent_return_count / event_count if event_count else 0.0
        ),
        "non_adjacent_prefix_return_event_count": non_adjacent_return_count,
        "codebook_entropy_bits": level_entropy_bits,
        "unique_complete_plan_rate": len(set(signature_digests))
        / len(signature_digests),
    }
    return _PrefixScaleStats(
        summary=summary,
        prefix_counts=prefix_counts,
        transition_counts=transition_counts,
        signature_digests=tuple(signature_digests),
        repeat_mass=repeat_mass,
    )


def _repeat_mass_by_length_bin(
    lengths: np.ndarray,
    repeat_mass: np.ndarray,
) -> list[dict[str, Any]]:
    bins = []
    for lower, upper in LENGTH_BINS:
        mask = (lengths >= lower) & (lengths < upper)
        bins.append(
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
    return bins


@dataclass(frozen=True, slots=True)
class _PopulationStats:
    summary: dict[str, Any]
    by_prefix_depth: dict[int, _PrefixScaleStats]
    transitions_by_return_type: dict[str, Counter[_Transition]]
    gap_counts: Counter[int]

    @property
    def full_scale(self) -> _PrefixScaleStats:
        return self.by_prefix_depth[max(self.by_prefix_depth)]


def _population_stats(
    panel: _Panel,
    *,
    codebook_size: int,
) -> _PopulationStats:
    signatures = _signatures(panel)
    prefix_depth = panel.prefix_codes.shape[-1]
    level_counts = [Counter[int]() for _ in range(prefix_depth)]
    transitions_by_return_type: dict[str, Counter[_Transition]] = {
        return_type: Counter() for return_type in _RETURN_TYPES
    }
    for signature in signatures:
        for level, counts in enumerate(level_counts):
            counts.update(prefix[level] for prefix in signature)
        seen = set(signature[:1])
        for previous, current in pairwise(signature):
            if current == previous:
                return_type = "immediate_return"
            elif current in seen:
                return_type = "older_return"
            else:
                return_type = "explore"
            transitions_by_return_type[return_type][previous, current] += 1
            seen.add(current)
    level_entropy_bits = [_entropy_bits(counts) for counts in level_counts]
    by_prefix_depth = {
        depth: _prefix_scale_stats(
            [
                tuple(prefix[:depth] for prefix in signature)
                for signature in signatures
            ],
            depth=depth,
            codebook_size=codebook_size,
            level_entropy_bits=level_entropy_bits[:depth],
        )
        for depth in range(1, prefix_depth + 1)
    }
    gap_counts: Counter[int] = Counter()
    for gaps, length in zip(
        panel.gap_tokens.tolist(), panel.lengths.tolist(), strict=True
    ):
        gap_counts.update(gaps[:length])

    full_scale = by_prefix_depth[prefix_depth]
    lengths = panel.lengths.numpy().astype(np.float64, copy=False)
    summary = {
        "trajectory_count": len(lengths),
        "occurrence_count": int(lengths.sum()),
        "mean_length": float(lengths.mean()),
        "p05_length": float(np.quantile(lengths, 0.05)),
        "median_length": float(np.median(lengths)),
        "p95_length": float(np.quantile(lengths, 0.95)),
        "repeat_mass_by_length_bin": _repeat_mass_by_length_bin(
            lengths, full_scale.repeat_mass
        ),
        **{key: full_scale.summary[key] for key in _FULL_SCALE_SUMMARY_KEYS},
    }
    return _PopulationStats(
        summary=summary,
        by_prefix_depth=by_prefix_depth,
        transitions_by_return_type=transitions_by_return_type,
        gap_counts=gap_counts,
    )


def _train_signature_overlap_metrics(
    generated_digests: tuple[bytes, ...],
    train: _Panel,
) -> dict[str, float]:
    generated = set(generated_digests)
    matched = generated.intersection(
        map(_signature_digest, _signatures(train))
    )
    return {
        "unique_train_signature_overlap": float(len(matched) / len(generated)),
        "sample_train_copy_rate": float(
            sum(digest in matched for digest in generated_digests)
            / len(generated_digests)
        ),
    }


def _scale_comparisons(
    left: _PrefixScaleStats,
    right: _PrefixScaleStats,
) -> dict[str, float]:
    comparisons = {
        "prefix_histogram_jsd": histogram_jsd(
            left.prefix_counts, right.prefix_counts
        ),
        "prefix_transition_histogram_jsd": histogram_jsd(
            left.transition_counts, right.transition_counts
        ),
    }
    for key in (
        "mean_repeat_mass",
        "adjacent_repeat_share",
        "non_adjacent_prefix_return_rate",
    ):
        comparisons[f"{key}_delta"] = float(
            left.summary[key] - right.summary[key]
        )
    return comparisons


def _population_metrics(
    left: _PopulationStats,
    right: _PopulationStats,
    *,
    names: tuple[str, str],
    train: _Panel | None = None,
) -> dict[str, Any]:
    left_name, right_name = names
    by_prefix_depth = {
        str(depth): {
            left_name: left.by_prefix_depth[depth].summary,
            right_name: right.by_prefix_depth[depth].summary,
            "comparisons": _scale_comparisons(
                left.by_prefix_depth[depth], right.by_prefix_depth[depth]
            ),
        }
        for depth in sorted(left.by_prefix_depth)
    }
    transition_by_return_type = {}
    for return_type in _RETURN_TYPES:
        left_counts = left.transitions_by_return_type[return_type]
        right_counts = right.transitions_by_return_type[return_type]
        transition_by_return_type[return_type] = {
            "generated_transition_count": left_counts.total(),
            "reference_transition_count": right_counts.total(),
            "transition_histogram_jsd": (
                None
                if not left_counts or not right_counts
                else histogram_jsd(left_counts, right_counts)
            ),
        }
    comparisons = {
        "mean_length_error_to_reference": float(
            abs(left.summary["mean_length"] - right.summary["mean_length"])
        ),
        "p95_length_error_to_reference": float(
            abs(left.summary["p95_length"] - right.summary["p95_length"])
        ),
        "prefix_histogram_jsd": histogram_jsd(
            left.full_scale.prefix_counts, right.full_scale.prefix_counts
        ),
        "prefix_transition_histogram_jsd": histogram_jsd(
            left.full_scale.transition_counts,
            right.full_scale.transition_counts,
        ),
        "gap_histogram_jsd": histogram_jsd(left.gap_counts, right.gap_counts),
    }
    if train is not None:
        comparisons.update(
            _train_signature_overlap_metrics(
                left.full_scale.signature_digests, train
            )
        )
    return {
        left_name: left.summary,
        right_name: right.summary,
        "comparisons": comparisons,
        "by_prefix_depth": by_prefix_depth,
        "transition_by_return_type": transition_by_return_type,
    }


_QUANTILES = (
    ("p05", 0.05),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p95", 0.95),
)


def _time_values(
    panel: _Panel,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Valid event times, per-step durations, and durations per gap token."""
    event_time = panel.event_time_minutes
    assert event_time is not None
    valid = torch.arange(event_time.shape[1])[None, :] < panel.lengths[:, None]
    durations = event_time.diff(
        dim=1, prepend=event_time.new_zeros(len(event_time), 1)
    )
    times = event_time[valid].numpy().astype(np.float64, copy=False)
    duration_values = durations[valid].numpy().astype(np.float64, copy=False)
    valid_gaps = panel.gap_tokens[valid].numpy()
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
            "quantiles_minutes": dict.fromkeys(name for name, _ in _QUANTILES),
        }
    return {
        "count": int(values.size),
        "mean_minutes": float(values.mean()),
        "std_minutes": float(values.std()),
        "quantiles_minutes": {
            name: float(np.quantile(values, quantile))
            for name, quantile in _QUANTILES
        },
    }


def _distribution_distances(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[float | None, float | None]:
    """Wasserstein-1 and Kolmogorov-Smirnov distances of two samples."""
    if left.size == 0 or right.size == 0:
        return None, None
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    support = np.unique(np.concatenate((left_sorted, right_sorted)))
    cdf_gap = np.abs(
        np.searchsorted(left_sorted, support, side="right") / left.size
        - np.searchsorted(right_sorted, support, side="right") / right.size
    )
    wasserstein = (
        0.0
        if support.size < 2
        else float(np.sum(cdf_gap[:-1] * np.diff(support)))
    )
    return wasserstein, float(np.max(cdf_gap))


def _continuous_distribution_pair(
    left: np.ndarray,
    right: np.ndarray,
    *,
    names: tuple[str, str],
) -> dict[str, Any]:
    left_summary = _continuous_summary(left)
    right_summary = _continuous_summary(right)
    quantile_deltas: dict[str, float | None] = {}
    for name, _ in _QUANTILES:
        left_value = left_summary["quantiles_minutes"][name]
        right_value = right_summary["quantiles_minutes"][name]
        quantile_deltas[name] = (
            None
            if left_value is None or right_value is None
            else float(left_value - right_value)
        )
    wasserstein, ks_distance = _distribution_distances(left, right)
    return {
        names[0]: left_summary,
        names[1]: right_summary,
        "wasserstein_1_minutes": wasserstein,
        "ks_distance": ks_distance,
        "mean_delta_minutes": (
            None
            if left.size == 0 or right.size == 0
            else float(left.mean() - right.mean())
        ),
        "quantile_delta_minutes": quantile_deltas,
    }


def _time_distribution_metrics(
    left: _Panel,
    right: _Panel,
    *,
    names: tuple[str, str],
) -> dict[str, Any] | None:
    if left.event_time_minutes is None or right.event_time_minutes is None:
        return None
    left_times, left_durations, left_by_gap = _time_values(left)
    right_times, right_durations, right_by_gap = _time_values(right)
    empty = np.empty(0, dtype=np.float64)
    return {
        "event_time_minutes": _continuous_distribution_pair(
            left_times, right_times, names=names
        ),
        "duration_minutes": _continuous_distribution_pair(
            left_durations, right_durations, names=names
        ),
        "duration_by_gap_token": {
            str(token): _continuous_distribution_pair(
                left_by_gap.get(token, empty),
                right_by_gap.get(token, empty),
                names=names,
            )
            for token in sorted(left_by_gap.keys() | right_by_gap.keys())
        },
    }


def _compare_panels(
    left: _Panel,
    right: _Panel,
    *,
    names: tuple[str, str],
    codebook_size: int,
    train: _Panel | None = None,
) -> dict[str, Any]:
    metrics = _population_metrics(
        _population_stats(left, codebook_size=codebook_size),
        _population_stats(right, codebook_size=codebook_size),
        names=names,
        train=train,
    )
    metrics["time_distributions"] = _time_distribution_metrics(
        left, right, names=names
    )
    return metrics


@dataclass(frozen=True, slots=True)
class _ReferencePanels:
    available_count: int
    reference: _Panel
    floor_left: _Panel
    floor_right: _Panel
    train: _Panel


def _load_reference_panels(
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
    if any(
        getattr(dataset.config, key) != getattr(checkpoint.config, key)
        for key in (
            "codebook_size",
            "gap_vocab",
            "prefix_depth",
            "max_occurrences",
            "context_dim",
            "temporal_state_enabled",
        )
    ):
        raise ValueError(
            "planner reference dataset configuration is incompatible with checkpoint"
        )

    split = _Panel.of(getattr(dataset, config.reference_split))
    available_count = split.size
    permutation = np.random.default_rng(config.evaluation_seed).permutation(
        available_count
    )
    if config.reference_limit is None:
        reference = split
        floor_count = available_count // 2
        if floor_count < 1:
            raise ValueError(
                "reference split must contain at least two rows for a real floor"
            )
    else:
        floor_count = config.reference_limit
        if available_count < 2 * floor_count:
            raise ValueError(
                "reference split is too small for two disjoint reference panels "
                f"of size {floor_count}"
            )
        reference = split.rows(permutation[:floor_count])
    return _ReferencePanels(
        available_count=available_count,
        reference=reference,
        floor_left=split.rows(permutation[:floor_count]),
        floor_right=split.rows(permutation[floor_count : 2 * floor_count]),
        train=_Panel.of(dataset.train),
    )


_ROUTER_RETURN_TYPES = ("first", *_RETURN_TYPES)
_DAY_MINUTES = 24.0 * 60.0
_WEEK_MINUTES = 7.0 * _DAY_MINUTES


@contextmanager
def _inference_on(
    model: torch.nn.Module, device: torch.device
) -> Generator[None]:
    was_training = model.training
    original_device = next(model.parameters()).device
    model.to(device)
    model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        model.to(original_device)
        model.train(was_training)


def _router_batch_events(
    model: Any,
    batch: OccurrenceBatch,
) -> dict[str, torch.Tensor]:
    """Router outputs and the target's visit history per valid occurrence."""
    output = model(batch)
    if (
        output.retrieve_probability is None
        or output.retrieve_responsibility is None
        or output.retrieval_mechanism_weights is None
    ):
        raise RuntimeError("mobility router omitted required diagnostics")
    arrival: torch.Tensor | None = None
    if model.config.temporal_state_enabled:
        assert batch.previous_event_time_minutes is not None
        arrival = model._reconstruct_arrival_time(
            batch.previous_event_time_minutes, batch.target_gap_tokens
        )
        conditioned = model.condition_on_gap_and_time(
            output.hidden, batch.target_gap_tokens, arrival
        )
    else:
        conditioned = model.condition_on_gap(
            output.hidden, batch.target_gap_tokens
        )
    routing = model.mobility_router(
        conditioned,
        model._embed_retrieval_memory(
            batch.previous_prefix_codes,
            batch.previous_gap_tokens,
            batch.previous_event_time_minutes,
        ),
        batch.previous_prefix_codes,
    )

    # previous[:, j] holds target[:, j - 1], so matching a target against
    # the causal memory finds its earlier visits.
    target = batch.target_prefix_codes
    previous = batch.previous_prefix_codes
    valid = batch.valid_mask
    indices = torch.arange(target.shape[1], device=target.device)
    matches = (target[:, :, None, :] == previous[:, None, :, :]).all(dim=-1)
    earlier = (
        matches
        & (previous >= 0).all(dim=-1)[:, None, :]
        & (indices[:, None] >= indices[None, :])
    )
    seen_before = earlier.any(dim=-1)
    latest = torch.where(earlier, indices, -1).max(dim=-1).values
    immediate = torch.zeros_like(valid)
    immediate[:, 1:] = (target[:, 1:] == target[:, :-1]).all(dim=-1)
    first = torch.zeros_like(valid)
    first[:, 0] = valid[:, 0]
    first_visits = (valid & ~seen_before).long()

    retrieval_targets = matches & routing.history_mask
    target_log_probability = torch.logsumexp(
        routing.retrieval_log_probability.masked_fill(
            ~retrieval_targets, float("-inf")
        ),
        dim=-1,
    )
    better = (
        routing.retrieval_log_probability > target_log_probability[..., None]
    ) & routing.history_mask

    per_step = {
        "retrieve": output.retrieve_probability.float(),
        "responsibility": output.retrieve_responsibility.float(),
        "mechanisms": output.retrieval_mechanism_weights.float(),
        "history_available": routing.history_mask.any(dim=-1),
        "gap": batch.target_gap_tokens,
        "seen_before": seen_before,
        "retrieval_target_probability": torch.exp(target_log_probability),
        "first": first,
        "explore": valid & ~seen_before & ~first,
        "immediate_return": immediate,
        "older_return": seen_before & ~immediate,
        "frequency": earlier.sum(dim=-1),
        "recency_steps": torch.where(seen_before, indices - latest + 1, 0),
        "retrieval_rank": torch.where(
            retrieval_targets.any(dim=-1), 1 + better.sum(dim=-1), 0
        ),
        "distinct": first_visits.cumsum(dim=1) - first_visits,
    }
    if arrival is not None:
        assert batch.previous_event_time_minutes is not None
        latest_time = batch.previous_event_time_minutes.gather(
            -1, latest.clamp_min(0)
        )
        per_step["time"] = arrival
        per_step["recency_minutes"] = torch.where(
            seen_before, arrival - latest_time, torch.zeros_like(arrival)
        )
    return {name: values[valid].cpu() for name, values in per_step.items()}


@dataclass(frozen=True, slots=True)
class _RouterEvents:
    """Router diagnostics of every valid reference occurrence."""

    expert_names: tuple[str, ...]
    values: dict[str, torch.Tensor]

    def summary(self, mask: torch.Tensor) -> dict[str, Any]:
        if not bool(mask.any()):
            return {
                "event_count": 0,
                "mean_explore_probability": None,
                "mean_retrieve_probability": None,
                "mean_retrieve_responsibility": None,
                "mean_retrieval_target_probability": None,
                "mechanism_weights": dict.fromkeys(self.expert_names),
            }
        retrieve = self.values["retrieve"][mask]
        target_probability = self.values["retrieval_target_probability"][
            mask & self.values["seen_before"]
        ]
        return {
            "event_count": int(mask.sum().item()),
            "mean_explore_probability": float((1.0 - retrieve).mean().item()),
            "mean_retrieve_probability": float(retrieve.mean().item()),
            "mean_retrieve_responsibility": float(
                self.values["responsibility"][mask].mean().item()
            ),
            "mean_retrieval_target_probability": (
                None
                if target_probability.numel() == 0
                else float(target_probability.mean().item())
            ),
            "mechanism_weights": self._mechanism_means(mask),
        }

    def _mechanism_means(self, mask: torch.Tensor) -> dict[str, float | None]:
        eligible = mask & self.values["history_available"]
        if not bool(eligible.any()):
            return dict.fromkeys(self.expert_names)
        weights = self.values["mechanisms"][eligible]
        return {
            name: float(weights[:, index].mean().item())
            for index, name in enumerate(self.expert_names)
        }

    def by_value(
        self,
        values: torch.Tensor,
        keys: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        keys = values if keys is None else keys
        return {
            str(int(key)): self.summary(values == int(key))
            for key in torch.unique(keys, sorted=True).tolist()
        }


def _retrieval_rank_summary(rank: torch.Tensor) -> dict[str, Any]:
    seen_rank = rank[rank > 0]
    if seen_rank.numel() == 0:
        return {
            "seen_target_count": 0,
            "mean_rank": None,
            "median_rank": None,
            "top1_rate": None,
            "top3_rate": None,
        }
    return {
        "seen_target_count": int(seen_rank.numel()),
        "mean_rank": float(seen_rank.float().mean().item()),
        "median_rank": float(seen_rank.float().median().item()),
        "top1_rate": float((seen_rank <= 1).float().mean().item()),
        "top3_rate": float((seen_rank <= 3).float().mean().item()),
    }


def _recency_minutes_summary(minutes: torch.Tensor) -> dict[str, Any]:
    if minutes.numel() == 0:
        return {
            "seen_target_count": 0,
            "mean_minutes": None,
            "median_minutes": None,
        }
    return {
        "seen_target_count": int(minutes.numel()),
        "mean_minutes": float(minutes.float().mean().item()),
        "median_minutes": float(minutes.float().median().item()),
    }


def _router_diagnostics(
    checkpoint: OccurrencePlannerCheckpoint,
    reference: _Panel,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any] | None:
    model = checkpoint.model
    if model.mobility_router is None:
        return None
    parts: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
    with _inference_on(model, device):
        for start in range(0, reference.size, batch_size):
            rows = reference.chunk(start, batch_size, device=device)
            batch = collate_occurrence_tensors(
                rows.prefix_codes,
                rows.gap_tokens,
                rows.lengths,
                config=checkpoint.config,
                legality=model.legality,
                event_time_minutes=rows.event_time_minutes,
                context_features=rows.context_features,
                city_ids=rows.city_ids,
            )
            for name, values in _router_batch_events(model, batch).items():
                parts[name].append(values)
    events = _RouterEvents(
        expert_names=tuple(model.mobility_router.expert_names),
        values={name: torch.cat(values) for name, values in parts.items()},
    )
    values = events.values
    recency_steps = values["recency_steps"]
    diagnostics = {
        **events.summary(
            torch.ones_like(values["retrieve"], dtype=torch.bool)
        ),
        "history_available_event_count": int(
            values["history_available"].sum().item()
        ),
        "seen_prefix_rate": float(values["seen_before"].float().mean().item()),
        "retrieval_rank": _retrieval_rank_summary(values["retrieval_rank"]),
        "retrieval_recency_minutes": None,
        "by_return_type": {
            name: events.summary(values[name]) for name in _ROUTER_RETURN_TYPES
        },
        "by_distinct_prefix_count": events.by_value(values["distinct"]),
        "by_target_prior_frequency": events.by_value(values["frequency"]),
        "by_target_recency_steps": events.by_value(
            recency_steps, keys=recency_steps[recency_steps > 0]
        ),
        "by_gap_token": events.by_value(values["gap"]),
        "by_day_of_week": None,
        "by_hour_of_day": None,
    }
    if "time" in values:
        times = values["time"]
        day = torch.floor(
            torch.remainder(times, _WEEK_MINUTES) / _DAY_MINUTES
        ).to(torch.long)
        hour = torch.floor(torch.remainder(times, _DAY_MINUTES) / 60.0).to(
            torch.long
        )
        diagnostics["by_day_of_week"] = events.by_value(day)
        diagnostics["by_hour_of_day"] = events.by_value(hour)
        diagnostics["retrieval_recency_minutes"] = _recency_minutes_summary(
            values["recency_minutes"][values["seen_before"]]
        )
    return diagnostics


def evaluate_occurrence_planner_samples(
    config: OccurrencePlannerSamplesEvaluationConfig,
) -> dict[str, Any]:
    config.validate()
    diagnostic_device = _resolve_router_diagnostics_device(config.device)
    payload = _load_samples(config.samples)
    checkpoint = load_occurrence_planner_checkpoint(config.checkpoint)
    samples, report = _validate_sample_payload(payload, checkpoint=checkpoint)
    panels = _load_reference_panels(config, checkpoint)
    codebook_size = checkpoint.config.codebook_size
    report["population_metrics"] = _compare_panels(
        samples,
        panels.reference,
        names=("generated", "reference"),
        codebook_size=codebook_size,
        train=panels.train,
    )
    report["real_vs_real_floor"] = _compare_panels(
        panels.floor_left,
        panels.floor_right,
        names=("real_left", "real_right"),
        codebook_size=codebook_size,
    )
    report["router_diagnostics"] = _router_diagnostics(
        checkpoint,
        panels.reference,
        device=diagnostic_device,
        batch_size=config.batch_size,
    )
    report.update(
        {
            "samples": str(config.samples),
            "checkpoint": str(config.checkpoint),
            "reference": str(config.reference),
            "reference_split": config.reference_split,
            "reference_count": panels.reference.size,
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
            "real_floor_panel_count": panels.floor_left.size,
            "output": str(config.output),
        }
    )
    write_json_atomic(config.output, report)
    return report


__all__ = [
    "OccurrencePlannerSamplesEvaluationConfig",
    "evaluate_occurrence_planner_samples",
]
