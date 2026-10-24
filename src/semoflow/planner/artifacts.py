from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast, overload

import torch

from semoflow.config.schemas.planner_config import (
    OccurrencePlannerTrainingOptions,
)
from semoflow.planner.config import (
    OccurrencePlannerConfig,
    validate_gap_duration_minutes,
)
from semoflow.planner.gap_codec import (
    bucketize_gap_durations,
    validate_gap_codec_contract,
)
from semoflow.planner.legality import LegalPrefixIndex
from semoflow.planner.model import OccurrenceARPlanner
from semoflow.planner.types import OccurrenceSequence

DATASET_ARTIFACT_TYPE = "occurrence_planner_dataset"

CHECKPOINT_ARTIFACT_TYPE = "occurrence_planner_checkpoint"

CONDITIONS_ARTIFACT_TYPE = "occurrence_planner_conditions"

SAMPLES_ARTIFACT_TYPE = "occurrence_planner_samples"


@dataclass(frozen=True, slots=True)
class OccurrencePlannerDataConfig:
    codebook_size: int
    gap_vocab: int
    prefix_depth: int
    max_occurrences: int
    context_dim: int = 0
    context_schema: tuple[str, ...] = ()
    city_conditioning: bool = False
    gap_codec_version: str | None = None
    gap_bucket_edges_minutes: tuple[float, ...] | None = None
    gap_duration_minutes: tuple[float, ...] | None = None
    observation_window_minutes: float | None = None

    def __post_init__(self) -> None:
        if (
            min(
                self.codebook_size,
                self.gap_vocab,
                self.prefix_depth,
                self.max_occurrences,
            )
            < 1
        ):
            raise ValueError("planner data vocabulary sizes must be positive")
        if self.context_dim < 0:
            raise ValueError("planner data context_dim must be non-negative")
        if len(self.context_schema) != self.context_dim:
            raise ValueError(
                "planner data context_schema must match context_dim"
            )
        if len(set(self.context_schema)) != len(self.context_schema):
            raise ValueError(
                "planner data context_schema names must be unique"
            )
        if not isinstance(self.city_conditioning, bool):
            raise ValueError("planner data city_conditioning must be boolean")
        validate_gap_duration_minutes(
            self.gap_duration_minutes,
            self.observation_window_minutes,
            gap_vocab=self.gap_vocab,
        )
        validate_gap_codec_contract(
            version=self.gap_codec_version,
            bucket_edges_minutes=self.gap_bucket_edges_minutes,
            representative_minutes=self.gap_duration_minutes,
        )

    @property
    def temporal_state_enabled(self) -> bool:
        return self.gap_duration_minutes is not None


class _OccurrenceSequenceView(Sequence[OccurrenceSequence]):
    __slots__ = (
        "_city_ids",
        "_event_time_minutes",
        "_gap_tokens",
        "_lengths",
        "_prefix_codes",
        "_size",
    )

    def __init__(
        self,
        prefix_codes: torch.Tensor,
        gap_tokens: torch.Tensor,
        lengths: torch.Tensor,
        event_time_minutes: torch.Tensor | None = None,
        city_ids: torch.Tensor | None = None,
    ) -> None:
        self._prefix_codes = prefix_codes
        self._gap_tokens = gap_tokens
        self._lengths = lengths
        self._event_time_minutes = event_time_minutes
        self._city_ids = city_ids
        self._size = int(lengths.shape[0])

    def __len__(self) -> int:
        return self._size

    @overload
    def __getitem__(self, index: int) -> OccurrenceSequence: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[OccurrenceSequence, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> OccurrenceSequence | tuple[OccurrenceSequence, ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(self._size)))
        if index < -self._size or index >= self._size:
            raise IndexError("occurrence sequence index out of range")
        if index < 0:
            index += self._size
        length = int(self._lengths[index].item())
        return OccurrenceSequence(
            prefix_codes=self._prefix_codes[index, :length],
            gap_tokens=self._gap_tokens[index, :length],
            event_time_minutes=(
                None
                if self._event_time_minutes is None
                else self._event_time_minutes[index, :length]
            ),
            city_id=(
                None
                if self._city_ids is None
                else int(self._city_ids[index].item())
            ),
        )

    def __iter__(self) -> Iterator[OccurrenceSequence]:
        for i in range(self._size):
            yield self[i]


@dataclass(frozen=True, slots=True)
class OccurrencePlannerSplit:
    sequences: Sequence[OccurrenceSequence]
    context_features: torch.Tensor | None
    prefix_codes: torch.Tensor
    gap_tokens: torch.Tensor
    lengths: torch.Tensor
    city_ids: torch.Tensor
    event_time_minutes: torch.Tensor | None = None

    @property
    def size(self) -> int:
        return int(self.lengths.shape[0])


@dataclass(frozen=True, slots=True)
class OccurrencePlannerDataset:
    config: OccurrencePlannerDataConfig
    legal_prefixes: torch.Tensor
    catalog_city_ids: torch.Tensor
    city_legal_prefix_mask: torch.Tensor
    train: OccurrencePlannerSplit
    validation: OccurrencePlannerSplit
    source_path: Path
    prefix_codebooks: torch.Tensor | None = None
    legal_prefix_centroids: torch.Tensor | None = None
    legal_prefix_centroid_available: torch.Tensor | None = None


def _validate_prefix_metadata(
    *,
    prefix_codebooks: object,
    centroids: object,
    centroid_available: object,
    codebook_size: int,
    prefix_depth: int,
    vocabulary: int,
    label: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    codebooks: torch.Tensor | None = None
    if prefix_codebooks is not None:
        if not isinstance(prefix_codebooks, torch.Tensor):
            raise ValueError(f"{label} prefix_codebooks must be a tensor")
        if (
            prefix_codebooks.device.type != "cpu"
            or prefix_codebooks.dtype != torch.float32
        ):
            raise ValueError(
                f"{label} prefix_codebooks must be a CPU float32 tensor"
            )
        if prefix_codebooks.ndim != 3 or prefix_codebooks.shape[:2] != (
            prefix_depth,
            codebook_size,
        ):
            raise ValueError(
                f"{label} prefix_codebooks must have shape "
                "[prefix_depth, codebook_size, latent_dim]"
            )
        if prefix_codebooks.shape[2] < 1:
            raise ValueError(f"{label} prefix_codebooks latent dim is empty")
        if not bool(torch.isfinite(prefix_codebooks).all()):
            raise ValueError(f"{label} prefix_codebooks must be finite")
        codebooks = prefix_codebooks.clone()

    if (centroids is None) != (centroid_available is None):
        raise ValueError(
            f"{label} prefix centroids and availability must be set together"
        )
    if centroids is None:
        return codebooks, None, None
    if not isinstance(centroids, torch.Tensor) or not isinstance(
        centroid_available, torch.Tensor
    ):
        raise ValueError(f"{label} prefix centroid metadata must be tensors")
    if (
        centroids.device.type != "cpu"
        or centroids.dtype != torch.float32
        or centroid_available.device.type != "cpu"
        or centroid_available.dtype != torch.bool
    ):
        raise ValueError(
            f"{label} prefix centroid metadata has an invalid dtype or device"
        )
    if centroids.shape != (vocabulary, 2) or centroid_available.shape != (
        vocabulary,
    ):
        raise ValueError(
            f"{label} prefix centroid metadata must align with legal prefixes"
        )
    if not bool(torch.isfinite(centroids).all()):
        raise ValueError(f"{label} prefix centroids must be finite")
    covered = centroids[centroid_available]
    if covered.numel() and (
        bool((covered[:, 0].abs() > 90.0).any())
        or bool((covered[:, 1].abs() > 180.0).any())
    ):
        raise ValueError(f"{label} prefix centroids are outside WGS84")
    return codebooks, centroids.clone(), centroid_available.clone()


def _publish_atomic(path: Path, write: Any, *, force: bool) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing artifact: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_torch_atomic(
    path: Path, payload: object, *, force: bool = False
) -> None:
    _publish_atomic(path, lambda tmp: torch.save(payload, tmp), force=force)


def _require_exact_keys(
    value: object, expected: set[str], *, label: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    mapping = cast("dict[str, Any]", value)
    if set(mapping) != expected:
        raise ValueError(
            f"{label} keys must be exactly {sorted(expected)}; "
            f"got {sorted(mapping)}"
        )
    return mapping


_SPLIT_KEYS = frozenset(
    {
        "prefix_codes",
        "gap_tokens",
        "lengths",
        "context_features",
        "event_time_minutes",
        "city_ids",
    }
)
_DATASET_KEYS = frozenset(
    {
        "artifact_type",
        "data_config",
        "legal_prefixes",
        "prefix_codebooks",
        "legal_prefix_centroids",
        "legal_prefix_centroid_available",
        "train",
        "validation",
    }
)
_CITY_LEGALITY_KEYS = frozenset({"catalog_city_ids", "city_legal_prefix_mask"})
_DATA_CONFIG_KEYS = frozenset(
    field.name for field in fields(OccurrencePlannerDataConfig)
)
_CONFIG_KEYS = frozenset(
    field.name for field in fields(OccurrencePlannerConfig)
)
_OPTIONS_KEYS = frozenset(
    field.name for field in fields(OccurrencePlannerTrainingOptions)
)


def _from_exact_mapping[T](
    build: Callable[..., T],
    value: object,
    keys: frozenset[str],
    *,
    label: str,
) -> T:
    """``build(**value)`` for a mapping with exactly ``keys``; invalid
    values surface as ValueError."""
    mapping = _require_exact_keys(value, set(keys), label=label)
    try:
        return build(**mapping)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc


def _load_context(
    context: object,
    *,
    rows: int,
    config: OccurrencePlannerDataConfig,
    label: str,
) -> torch.Tensor | None:
    if config.context_dim == 0:
        if context is not None:
            raise ValueError(
                f"{label} context must be absent when context_dim=0"
            )
        return None
    if not isinstance(context, torch.Tensor):
        raise ValueError(f"{label} context_features must be a tensor")
    if context.device.type != "cpu":
        raise ValueError(f"{label} context_features must be on CPU")
    if context.dtype != torch.float32:
        raise ValueError(f"{label} context_features must use torch.float32")
    if context.shape != (rows, config.context_dim):
        raise ValueError(f"{label} context_features have the wrong shape")
    if not bool(torch.isfinite(context).all()):
        raise ValueError(f"{label} context_features must be finite")
    return context.clone()


def _load_event_times(
    event_time: object,
    *,
    gaps: torch.Tensor,
    valid: torch.Tensor,
    config: OccurrencePlannerDataConfig,
    label: str,
) -> torch.Tensor | None:
    if not config.temporal_state_enabled:
        if event_time is not None:
            raise ValueError(
                f"{label} event_time_minutes supplied without a time channel"
            )
        return None
    if not isinstance(event_time, torch.Tensor):
        raise ValueError(
            f"{label} event_time_minutes is required by a time channel"
        )
    if event_time.device.type != "cpu":
        raise ValueError(f"{label} event_time_minutes must be on CPU")
    if event_time.dtype != torch.float32:
        raise ValueError(f"{label} event_time_minutes must use torch.float32")
    if event_time.shape != gaps.shape:
        raise ValueError(f"{label} event_time_minutes have the wrong shape")
    if not bool(torch.isfinite(event_time).all()):
        raise ValueError(f"{label} event_time_minutes must be finite")
    if bool((event_time < 0).any()):
        raise ValueError(f"{label} event_time_minutes must be non-negative")
    window = config.observation_window_minutes
    edges = config.gap_bucket_edges_minutes
    assert window is not None and edges is not None
    if bool((event_time[valid] > window + 1e-3).any()):
        raise ValueError(
            f"{label} event_time_minutes exceeds the observation window"
        )
    durations = event_time.clone()
    durations[:, 1:] -= event_time[:, :-1]
    if bool(((durations[:, 1:] < 0) & valid[:, 1:]).any()):
        raise ValueError(f"{label} event_time_minutes must be non-decreasing")
    expected_gaps = bucketize_gap_durations(durations, edges)
    if not torch.equal(gaps[valid], expected_gaps[valid]):
        raise ValueError(
            f"{label} gap_tokens disagree with event_time_minutes"
        )
    return event_time.clone()


def _load_split(
    value: object,
    *,
    label: str,
    config: OccurrencePlannerDataConfig,
    legality: LegalPrefixIndex,
) -> OccurrencePlannerSplit:
    split = _require_exact_keys(value, set(_SPLIT_KEYS), label=label)
    prefixes = split["prefix_codes"]
    gaps = split["gap_tokens"]
    lengths = split["lengths"]
    city_ids = split["city_ids"]
    if not (
        isinstance(prefixes, torch.Tensor)
        and isinstance(gaps, torch.Tensor)
        and isinstance(lengths, torch.Tensor)
    ):
        raise ValueError(f"{label} sequence fields must be tensors")
    sequence = (prefixes, gaps, lengths)
    if any(tensor.device.type != "cpu" for tensor in sequence):
        raise ValueError(f"{label} tensors must be on CPU")
    if any(tensor.dtype != torch.long for tensor in sequence):
        raise ValueError(f"{label} sequence tensors must use torch.long")
    if prefixes.ndim != 3:
        raise ValueError(f"{label}.prefix_codes must have shape [N, S, P]")
    rows, steps, depth = prefixes.shape
    if rows < 1:
        raise ValueError(f"{label} must be non-empty")
    if depth != config.prefix_depth:
        raise ValueError(f"{label} prefix depth differs from data_config")
    if steps < 1 or steps > config.max_occurrences:
        raise ValueError(f"{label} sequence width exceeds max_occurrences")
    if gaps.shape != (rows, steps) or lengths.shape != (rows,):
        raise ValueError(f"{label} gap/length tensors do not align")
    if bool((lengths < 1).any()) or bool((lengths > steps).any()):
        raise ValueError(f"{label} lengths are outside the padded width")
    if bool((prefixes < 0).any()) or bool(
        (prefixes >= config.codebook_size).any()
    ):
        raise ValueError(f"{label} prefix code is outside the codebook")
    if bool((gaps < 0).any()) or bool((gaps >= config.gap_vocab).any()):
        raise ValueError(f"{label} gap token is outside the vocabulary")
    valid = torch.arange(steps)[None, :] < lengths[:, None]
    if bool((prefixes[~valid] != 0).any()) or bool((gaps[~valid] != 0).any()):
        raise ValueError(f"{label} padded tokens must be zero")
    if not bool(legality.contains(prefixes[valid]).all()):
        raise ValueError(f"{label} contains an illegal prefix")
    if not isinstance(city_ids, torch.Tensor):
        raise ValueError(f"{label} city_ids must be a tensor")
    if (
        city_ids.device.type != "cpu"
        or city_ids.dtype != torch.long
        or city_ids.shape != (rows,)
    ):
        raise ValueError(
            f"{label} city_ids must be a CPU long tensor with shape [N]"
        )
    context = _load_context(
        split["context_features"], rows=rows, config=config, label=label
    )
    event_time = _load_event_times(
        split["event_time_minutes"],
        gaps=gaps,
        valid=valid,
        config=config,
        label=label,
    )
    prefixes, gaps, lengths, city_ids = (
        tensor.clone() for tensor in (prefixes, gaps, lengths, city_ids)
    )
    return OccurrencePlannerSplit(
        sequences=_OccurrenceSequenceView(
            prefixes, gaps, lengths, event_time, city_ids
        ),
        context_features=context,
        prefix_codes=prefixes,
        gap_tokens=gaps,
        lengths=lengths,
        event_time_minutes=event_time,
        city_ids=city_ids,
    )


def _load_data_config(value: object) -> OccurrencePlannerDataConfig:
    mapping = dict(cast("dict[str, Any]", value))
    mapping.setdefault("context_schema", ())
    mapping.setdefault("city_conditioning", False)
    return _from_exact_mapping(
        OccurrencePlannerDataConfig,
        mapping,
        _DATA_CONFIG_KEYS,
        label="planner dataset data_config",
    )


def _load_legal_prefixes(
    value: object, config: OccurrencePlannerDataConfig
) -> tuple[torch.Tensor, LegalPrefixIndex]:
    if not isinstance(value, torch.Tensor):
        raise ValueError("planner dataset legal_prefixes must be a tensor")
    if value.device.type != "cpu" or value.dtype != torch.long:
        raise ValueError(
            "planner dataset legal_prefixes must be a CPU long tensor"
        )
    legality = LegalPrefixIndex(
        value,
        codebook_size=config.codebook_size,
        prefix_depth=config.prefix_depth,
    )
    return value, legality


def _load_city_legality(
    payload: dict[str, Any], vocabulary: int
) -> tuple[torch.Tensor, torch.Tensor]:
    catalog_city_ids = payload["catalog_city_ids"]
    city_legal_prefix_mask = payload["city_legal_prefix_mask"]
    if not isinstance(catalog_city_ids, torch.Tensor) or not isinstance(
        city_legal_prefix_mask, torch.Tensor
    ):
        raise ValueError(
            "planner dataset city legality fields must be tensors"
        )
    if (
        catalog_city_ids.device.type != "cpu"
        or catalog_city_ids.dtype != torch.long
        or catalog_city_ids.ndim != 1
        or catalog_city_ids.numel() < 1
    ):
        raise ValueError(
            "catalog_city_ids must be a non-empty CPU long vector"
        )
    if (
        city_legal_prefix_mask.device.type != "cpu"
        or city_legal_prefix_mask.dtype != torch.bool
        or city_legal_prefix_mask.shape
        != (catalog_city_ids.numel(), vocabulary)
    ):
        raise ValueError(
            "city_legal_prefix_mask must have shape [cities, legal vocabulary]"
        )
    if not bool(city_legal_prefix_mask.any(dim=-1).all()):
        raise ValueError("each catalog city must permit at least one prefix")
    if torch.unique(catalog_city_ids).numel() != catalog_city_ids.numel():
        raise ValueError("catalog_city_ids must be unique")
    if not torch.equal(catalog_city_ids, torch.sort(catalog_city_ids).values):
        raise ValueError("catalog_city_ids must be sorted")
    return catalog_city_ids, city_legal_prefix_mask


def _check_split_cities(
    split: OccurrencePlannerSplit,
    *,
    label: str,
    catalog_city_ids: torch.Tensor,
    city_legal_prefix_mask: torch.Tensor,
    legality: LegalPrefixIndex,
) -> None:
    city_to_row = {
        int(city_id): row
        for row, city_id in enumerate(catalog_city_ids.tolist())
    }
    city_ids = split.city_ids.tolist()
    unknown = sorted(set(city_ids) - set(city_to_row))
    if unknown:
        raise ValueError(f"{label} contains unknown city_ids: {unknown}")
    steps = split.prefix_codes.shape[1]
    valid = torch.arange(steps)[None, :] < split.lengths[:, None]
    row_indices = torch.tensor(
        [city_to_row[value] for value in city_ids], dtype=torch.long
    )
    target_ids = legality.token_ids(split.prefix_codes[valid])
    event_rows = row_indices[:, None].expand(-1, steps)[valid]
    if not bool(city_legal_prefix_mask[event_rows, target_ids].all()):
        raise ValueError(f"{label} contains a prefix illegal for its city")


def load_occurrence_planner_dataset(path: Path) -> OccurrencePlannerDataset:
    path = Path(path)
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"could not load planner dataset: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("planner dataset must be a dictionary")
    raw = dict(raw)
    raw.pop("format_version", None)
    raw.setdefault("prefix_codebooks", None)
    raw.setdefault("legal_prefix_centroids", None)
    raw.setdefault("legal_prefix_centroid_available", None)
    legacy_single_city = "catalog_city_ids" not in raw
    expected_keys = set(_DATASET_KEYS)
    if not legacy_single_city:
        expected_keys |= _CITY_LEGALITY_KEYS
    payload = _require_exact_keys(raw, expected_keys, label="planner dataset")
    if payload["artifact_type"] != DATASET_ARTIFACT_TYPE:
        raise ValueError("planner dataset artifact_type is unsupported")
    config = _load_data_config(payload["data_config"])
    legal_prefixes, legality = _load_legal_prefixes(
        payload["legal_prefixes"], config
    )
    dataset_codebooks, dataset_centroids, dataset_centroid_available = (
        _validate_prefix_metadata(
            prefix_codebooks=payload["prefix_codebooks"],
            centroids=payload["legal_prefix_centroids"],
            centroid_available=payload["legal_prefix_centroid_available"],
            codebook_size=config.codebook_size,
            prefix_depth=config.prefix_depth,
            vocabulary=legality.vocabulary_size,
            label="planner dataset",
        )
    )
    if legacy_single_city:
        payload["catalog_city_ids"] = torch.tensor([0], dtype=torch.long)
        payload["city_legal_prefix_mask"] = torch.ones(
            (1, legality.vocabulary_size), dtype=torch.bool
        )
        for split_name in ("train", "validation"):
            split_payload = payload[split_name]
            if "city_ids" not in split_payload:
                split_payload["city_ids"] = torch.zeros(
                    split_payload["lengths"].shape[0], dtype=torch.long
                )
    catalog_city_ids, city_legal_prefix_mask = _load_city_legality(
        payload, legality.vocabulary_size
    )
    train, validation = (
        _load_split(
            payload[label], label=label, config=config, legality=legality
        )
        for label in ("train", "validation")
    )
    for label, split in (("train", train), ("validation", validation)):
        _check_split_cities(
            split,
            label=label,
            catalog_city_ids=catalog_city_ids,
            city_legal_prefix_mask=city_legal_prefix_mask,
            legality=legality,
        )

    return OccurrencePlannerDataset(
        config=config,
        legal_prefixes=legal_prefixes.clone(),
        prefix_codebooks=dataset_codebooks,
        legal_prefix_centroids=dataset_centroids,
        legal_prefix_centroid_available=dataset_centroid_available,
        catalog_city_ids=catalog_city_ids.clone(),
        city_legal_prefix_mask=city_legal_prefix_mask.clone(),
        train=train,
        validation=validation,
        source_path=path.resolve(),
    )


@dataclass(frozen=True, slots=True)
class OccurrencePlannerCheckpoint:
    config: OccurrencePlannerConfig
    legal_prefixes: torch.Tensor
    catalog_city_ids: torch.Tensor
    city_legal_prefix_mask: torch.Tensor
    model: OccurrenceARPlanner
    epoch: int
    validation_metric: float
    seed: int
    options: OccurrencePlannerTrainingOptions
    prefix_codebooks: torch.Tensor | None = None
    legal_prefix_centroids: torch.Tensor | None = None
    legal_prefix_centroid_available: torch.Tensor | None = None


def save_occurrence_planner_checkpoint(
    path: Path,
    *,
    config: OccurrencePlannerConfig,
    legal_prefixes: torch.Tensor,
    catalog_city_ids: torch.Tensor,
    city_legal_prefix_mask: torch.Tensor,
    model_state: dict[str, Any],
    epoch: int,
    validation_metric: float,
    seed: int,
    options: OccurrencePlannerTrainingOptions,
    prefix_codebooks: torch.Tensor | None = None,
    legal_prefix_centroids: torch.Tensor | None = None,
    legal_prefix_centroid_available: torch.Tensor | None = None,
    force: bool = False,
) -> None:
    payload = {
        "artifact_type": CHECKPOINT_ARTIFACT_TYPE,
        "config": asdict(config),
        "legal_prefixes": legal_prefixes.detach()
        .to("cpu", torch.long)
        .clone(),
        "catalog_city_ids": catalog_city_ids.detach()
        .to("cpu", torch.long)
        .clone(),
        "city_legal_prefix_mask": city_legal_prefix_mask.detach()
        .to("cpu", torch.bool)
        .clone(),
        "prefix_codebooks": (
            None
            if prefix_codebooks is None
            else prefix_codebooks.detach().to("cpu", torch.float32).clone()
        ),
        "legal_prefix_centroids": (
            None
            if legal_prefix_centroids is None
            else legal_prefix_centroids.detach()
            .to("cpu", torch.float32)
            .clone()
        ),
        "legal_prefix_centroid_available": (
            None
            if legal_prefix_centroid_available is None
            else legal_prefix_centroid_available.detach()
            .to("cpu", torch.bool)
            .clone()
        ),
        "model_state": model_state,
        "epoch": int(epoch),
        "validation_metric": float(validation_metric),
        "seed": int(seed),
        "options": asdict(options),
    }
    save_torch_atomic(path, payload, force=force)


def load_occurrence_planner_checkpoint(
    path: Path,
) -> OccurrencePlannerCheckpoint:
    path = Path(path)
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"could not load planner checkpoint: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("planner checkpoint must be a dictionary")
    raw = dict(raw)
    raw.pop("format_version", None)
    raw.pop("dataset_sha256", None)
    payload = _require_exact_keys(
        raw,
        {
            "artifact_type",
            "config",
            "legal_prefixes",
            "catalog_city_ids",
            "city_legal_prefix_mask",
            "prefix_codebooks",
            "legal_prefix_centroids",
            "legal_prefix_centroid_available",
            "model_state",
            "epoch",
            "validation_metric",
            "seed",
            "options",
        },
        label="planner checkpoint",
    )
    if payload["artifact_type"] != CHECKPOINT_ARTIFACT_TYPE:
        raise ValueError("planner checkpoint artifact_type is unsupported")
    config = _from_exact_mapping(
        OccurrencePlannerConfig,
        payload["config"],
        _CONFIG_KEYS,
        label="planner checkpoint config",
    )
    legal_prefixes = payload["legal_prefixes"]
    if not isinstance(legal_prefixes, torch.Tensor):
        raise ValueError("planner checkpoint legal_prefixes must be a tensor")
    if (
        legal_prefixes.device.type != "cpu"
        or legal_prefixes.dtype != torch.long
    ):
        raise ValueError(
            "planner checkpoint legal_prefixes must be a CPU long tensor"
        )
    model_state = payload["model_state"]
    if not isinstance(model_state, dict):
        raise ValueError("planner checkpoint model_state must be a mapping")
    epoch = payload["epoch"]
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise ValueError("planner checkpoint epoch must be a positive integer")
    validation_metric = payload["validation_metric"]
    if not isinstance(validation_metric, int | float) or not math.isfinite(
        float(validation_metric)
    ):
        raise ValueError(
            "planner checkpoint validation_metric must be a finite number"
        )
    seed = payload["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("planner checkpoint seed must be an integer")
    raw_options = payload["options"]
    if not isinstance(raw_options, dict):
        raise ValueError("planner checkpoint options must be a mapping")
    options = _from_exact_mapping(
        OccurrencePlannerTrainingOptions,
        {"recurrence_weight": 0.0, **raw_options},
        _OPTIONS_KEYS,
        label="planner checkpoint options",
    )
    catalog_city_ids = payload["catalog_city_ids"]
    city_legal_prefix_mask = payload["city_legal_prefix_mask"]
    if not isinstance(catalog_city_ids, torch.Tensor) or not isinstance(
        city_legal_prefix_mask, torch.Tensor
    ):
        raise ValueError(
            "planner checkpoint city legality fields must be tensors"
        )
    if (
        catalog_city_ids.device.type != "cpu"
        or catalog_city_ids.dtype != torch.long
        or catalog_city_ids.ndim != 1
        or catalog_city_ids.numel() < 1
    ):
        raise ValueError(
            "checkpoint catalog_city_ids must be a CPU long vector"
        )
    if (
        city_legal_prefix_mask.device.type != "cpu"
        or city_legal_prefix_mask.dtype != torch.bool
        or city_legal_prefix_mask.shape
        != (catalog_city_ids.numel(), legal_prefixes.shape[0])
    ):
        raise ValueError(
            "checkpoint city_legal_prefix_mask is incompatible or has the wrong shape or dtype"
        )
    checkpoint_codebooks, checkpoint_centroids, checkpoint_available = (
        _validate_prefix_metadata(
            prefix_codebooks=payload["prefix_codebooks"],
            centroids=payload["legal_prefix_centroids"],
            centroid_available=payload["legal_prefix_centroid_available"],
            codebook_size=config.codebook_size,
            prefix_depth=config.prefix_depth,
            vocabulary=int(legal_prefixes.shape[0]),
            label="planner checkpoint",
        )
    )
    try:
        model = OccurrenceARPlanner(
            config,
            legal_prefixes,
            catalog_city_ids=catalog_city_ids,
            city_legal_prefix_mask=city_legal_prefix_mask,
            prefix_codebooks=checkpoint_codebooks,
            legal_prefix_centroids=checkpoint_centroids,
            legal_prefix_centroid_available=checkpoint_available,
        )
        model.load_state_dict(model_state)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"planner checkpoint state is incompatible: {exc}"
        ) from exc
    model.eval()
    return OccurrencePlannerCheckpoint(
        config=config,
        legal_prefixes=legal_prefixes.clone(),
        catalog_city_ids=catalog_city_ids.clone(),
        city_legal_prefix_mask=city_legal_prefix_mask.clone(),
        model=model,
        epoch=epoch,
        validation_metric=float(validation_metric),
        seed=seed,
        options=options,
        prefix_codebooks=checkpoint_codebooks,
        legal_prefix_centroids=checkpoint_centroids,
        legal_prefix_centroid_available=checkpoint_available,
    )


@dataclass(frozen=True, slots=True)
class OccurrencePlannerConditions:
    lengths: torch.Tensor | None
    context_features: torch.Tensor | None
    city_ids: torch.Tensor | None


def load_occurrence_planner_conditions(
    path: Path,
    *,
    num_samples: int,
    context_dim: int,
    context_schema: tuple[str, ...],
    max_occurrences: int,
    catalog_city_ids: torch.Tensor,
) -> OccurrencePlannerConditions:
    path = Path(path)
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"could not load planner conditions: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("planner conditions must be a dictionary")
    raw = dict(raw)
    raw.pop("format_version", None)
    payload = _require_exact_keys(
        raw,
        {
            "artifact_type",
            "lengths",
            "context_features",
            "context_schema",
            "city_ids",
        },
        label="planner conditions",
    )
    if payload["artifact_type"] != CONDITIONS_ARTIFACT_TYPE:
        raise ValueError("planner conditions artifact_type is unsupported")
    raw_context_schema = payload["context_schema"]
    if not isinstance(raw_context_schema, tuple) or not all(
        isinstance(name, str) for name in raw_context_schema
    ):
        raise ValueError(
            "planner conditions context_schema must be a string tuple"
        )
    if raw_context_schema != context_schema:
        raise ValueError(
            "planner conditions context_schema differs from checkpoint"
        )
    lengths = payload["lengths"]
    if lengths is not None:
        if not isinstance(lengths, torch.Tensor):
            raise ValueError("planner conditions lengths must be a tensor")
        if (
            lengths.device.type != "cpu"
            or lengths.dtype != torch.long
            or lengths.shape != (num_samples,)
        ):
            raise ValueError(
                "planner conditions lengths must be a CPU long tensor "
                "with shape [num_samples]"
            )
        if bool((lengths < 1).any()) or bool(
            (lengths > max_occurrences).any()
        ):
            raise ValueError(
                "planner conditions length is outside the vocabulary"
            )
        lengths = lengths.clone()
    context_features = payload["context_features"]
    if context_dim == 0:
        if context_features is not None:
            raise ValueError(
                "planner conditions context must be absent when context_dim=0"
            )
    else:
        if not isinstance(context_features, torch.Tensor):
            raise ValueError(
                "planner conditions context_features are required by the checkpoint"
            )
        if (
            context_features.device.type != "cpu"
            or context_features.dtype != torch.float32
            or context_features.shape != (num_samples, context_dim)
        ):
            raise ValueError(
                "planner conditions context_features must be a CPU float32 "
                "tensor with shape [num_samples, context_dim]"
            )
        if not bool(torch.isfinite(context_features).all()):
            raise ValueError(
                "planner conditions context_features must be finite"
            )
        context_features = context_features.clone()
    city_ids = payload["city_ids"]
    if city_ids is None:
        if catalog_city_ids.numel() != 1:
            raise ValueError(
                "planner conditions city_ids are required for multi-city checkpoints"
            )
    else:
        if not isinstance(city_ids, torch.Tensor):
            raise ValueError("planner conditions city_ids must be a tensor")
        if (
            city_ids.device.type != "cpu"
            or city_ids.dtype != torch.long
            or city_ids.shape != (num_samples,)
        ):
            raise ValueError(
                "planner conditions city_ids must be a CPU long tensor "
                "with shape [num_samples]"
            )
        allowed = torch.isin(city_ids, catalog_city_ids)
        if not bool(allowed.all()):
            raise ValueError("planner conditions contain unknown city_ids")
        city_ids = city_ids.clone()
    return OccurrencePlannerConditions(
        lengths=lengths,
        context_features=context_features,
        city_ids=city_ids,
    )


def save_occurrence_planner_samples(
    path: Path,
    *,
    prefix_codes: torch.Tensor,
    gap_tokens: torch.Tensor,
    lengths: torch.Tensor,
    model_log_probability: torch.Tensor,
    sampling_log_probability: torch.Tensor,
    length_was_sampled: torch.Tensor,
    city_ids: torch.Tensor,
    context_features: torch.Tensor | None,
    context_schema: tuple[str, ...],
    seed: int,
    temperature: float,
    length_temperature: float,
    gap_temperature: float,
    prefix_temperature: float,
    route_temperature: float,
    event_time_minutes: torch.Tensor | None = None,
    force: bool = False,
) -> None:
    payload = {
        "artifact_type": SAMPLES_ARTIFACT_TYPE,
        "prefix_codes": prefix_codes.detach().to("cpu", torch.long).clone(),
        "gap_tokens": gap_tokens.detach().to("cpu", torch.long).clone(),
        "lengths": lengths.detach().to("cpu", torch.long).clone(),
        "model_log_probability": model_log_probability.detach()
        .to("cpu", torch.float32)
        .clone(),
        "sampling_log_probability": sampling_log_probability.detach()
        .to("cpu", torch.float32)
        .clone(),
        "length_was_sampled": length_was_sampled.detach()
        .to("cpu", torch.bool)
        .clone(),
        "city_ids": city_ids.detach().to("cpu", torch.long).clone(),
        "context_features": (
            None
            if context_features is None
            else context_features.detach().to("cpu", torch.float32).clone()
        ),
        "context_schema": context_schema,
        "event_time_minutes": (
            None
            if event_time_minutes is None
            else event_time_minutes.detach().to("cpu", torch.float32).clone()
        ),
        "seed": int(seed),
        "temperature": float(temperature),
        "length_temperature": float(length_temperature),
        "gap_temperature": float(gap_temperature),
        "prefix_temperature": float(prefix_temperature),
        "route_temperature": float(route_temperature),
    }
    save_torch_atomic(path, payload, force=force)


__all__ = [
    "CHECKPOINT_ARTIFACT_TYPE",
    "CONDITIONS_ARTIFACT_TYPE",
    "DATASET_ARTIFACT_TYPE",
    "SAMPLES_ARTIFACT_TYPE",
    "OccurrencePlannerCheckpoint",
    "OccurrencePlannerConditions",
    "OccurrencePlannerDataConfig",
    "OccurrencePlannerDataset",
    "OccurrencePlannerSplit",
    "load_occurrence_planner_checkpoint",
    "load_occurrence_planner_conditions",
    "load_occurrence_planner_dataset",
    "save_occurrence_planner_checkpoint",
    "save_occurrence_planner_samples",
    "save_torch_atomic",
]
