from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any

import torch

from semoflow.config.constants import MASK_ID, PAD_ID
from semoflow.metrics.trajectory.distribution import (
    activity_ngram_jsd,
    jsd,
    location_time_joint_jsd,
    marginal_distribution,
    od_pair_jsd,
    token_change_count_jsd,
    transition_distance,
    visit_count_jsd,
)
from semoflow.metrics.trajectory.geo import (
    centroid_geo_error,
    spatial_jsd,
    travel_distance_jsd,
)
from semoflow.metrics.trajectory.mobility_motifs import mobility_motif_metrics

logger = logging.getLogger(__name__)

_MAX_TRANSITION_VOCAB = 5000
_MAX_OD_VOCAB = 500
_ACTIVITY_FIELDS = ("top_cat", "naics4", "semantic_id")
_PRIMARY_FIELDS = (
    "hour",
    "weekday",
    "month",
    "top_cat",
    "naics4",
    "semantic_id",
    "semantic_id_token",
)


@dataclass
class TrajectoryMetrics:
    field_jsd: dict[str, float]
    field_transition_distance: dict[str, float]
    visit_count_jsd: float
    od_pair_jsd: dict[str, float]
    token_change_count_jsd: dict[str, float]
    # Kept for report-schema stability; nothing computes it any more.
    token_min_distance: float | None = None
    geo_error_km: float | None = None
    travel_dist_jsd: float | None = None
    geohash_spatial_jsd: float | None = None
    location_time_joint_jsd: dict[str, float] | None = None
    activity_ngram_jsd: dict[str, float] | None = None
    mobility_motif_jsd: float | None = None
    mobility_motifs: dict[str, Any] | None = None
    decode_validity: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def decode_validity_rates(
    tokens: dict[str, torch.Tensor],
    field_vocab: dict[str, int],
    *,
    pad_id: int = PAD_ID,
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for field, tensor in tokens.items():
        vocab_size = field_vocab.get(field)
        if vocab_size is None:
            continue
        invalid = (tensor < 0) | (tensor >= vocab_size) | (tensor == MASK_ID)
        scored = tensor != pad_id
        denom = int(scored.sum().item())
        rates[field] = (
            1.0
            if denom == 0
            else float(((~invalid) & scored).sum().item() / denom)
        )
    return rates


def _without_none(values: dict[str, float | None]) -> dict[str, float] | None:
    kept = {key: value for key, value in values.items() if value is not None}
    return kept or None


def _per_field_metrics(
    tokens_real: dict[str, torch.Tensor],
    tokens_gen: dict[str, torch.Tensor],
    field_vocab: dict[str, int],
    *,
    pad_id: int,
) -> dict[str, Any]:
    """Marginal, transition, OD-pair and change-count distances of every
    field on both sides, and the visit-count JSD of the first."""
    field_jsd: dict[str, float] = {}
    field_td: dict[str, float] = {}
    od_jsd: dict[str, float] = {}
    rc_jsd: dict[str, float] = {}
    shared = [
        field
        for field in field_vocab
        if field in tokens_real and field in tokens_gen
    ]
    for field in shared:
        real, gen, vocab_size = (
            tokens_real[field],
            tokens_gen[field],
            field_vocab[field],
        )
        field_jsd[field] = jsd(
            marginal_distribution(real, vocab_size, pad_id=pad_id),
            marginal_distribution(gen, vocab_size, pad_id=pad_id),
        )
        if vocab_size > _MAX_TRANSITION_VOCAB:
            logger.debug(
                "skipping transition distance for %s: vocab_size=%d > %d",
                field,
                vocab_size,
                _MAX_TRANSITION_VOCAB,
            )
        else:
            field_td[field] = transition_distance(
                real, gen, vocab_size, pad_id=pad_id
            )
        od_result = od_pair_jsd(
            real, gen, vocab_size, pad_id=pad_id, max_vocab=_MAX_OD_VOCAB
        )
        if od_result is not None:
            od_jsd[field] = od_result
        rc_jsd[field] = token_change_count_jsd(real, gen, pad_id=pad_id)
    return {
        "field_jsd": field_jsd,
        "field_transition_distance": field_td,
        "visit_count_jsd": (
            visit_count_jsd(
                tokens_real[shared[0]], tokens_gen[shared[0]], pad_id=pad_id
            )
            if shared
            else 0.0
        ),
        "od_pair_jsd": od_jsd,
        "token_change_count_jsd": rc_jsd,
    }


def _spatial_metrics(
    tokens_real: dict[str, torch.Tensor],
    tokens_gen: dict[str, torch.Tensor],
    placekey_latlon: dict[int, tuple[float, float]] | None,
    geohash_lookup: dict[int, str] | None,
    *,
    pad_id: int,
) -> dict[str, float]:
    """Travel, centroid and geohash distances of the placekey visits."""
    if "placekey_id" not in tokens_real or "placekey_id" not in tokens_gen:
        return {}
    real, gen = tokens_real["placekey_id"], tokens_gen["placekey_id"]
    metrics: dict[str, float] = {}
    if placekey_latlon is not None:
        metrics["travel_dist_jsd"] = travel_distance_jsd(
            real, gen, placekey_latlon, pad_id=pad_id
        )
        metrics["geo_error_km"] = centroid_geo_error(
            real, gen, placekey_latlon, pad_id=pad_id
        )
    if geohash_lookup is not None:
        metrics["geohash_spatial_jsd"] = spatial_jsd(
            real, gen, geohash_lookup, pad_id=pad_id
        )
    return metrics


def _activity_metrics(
    tokens_real: dict[str, torch.Tensor],
    tokens_gen: dict[str, torch.Tensor],
    field_vocab: dict[str, int],
    *,
    pad_id: int,
) -> dict[str, dict[str, float] | None]:
    """Location-hour joint and trigram distances of the activity fields."""
    fields = [
        field
        for field in _ACTIVITY_FIELDS
        if field in tokens_real
        and field in tokens_gen
        and field in field_vocab
    ]
    hour_vocab = field_vocab.get("hour", 0)
    location_time = None
    if "hour" in tokens_real and "hour" in tokens_gen and hour_vocab > 0:
        location_time = _without_none(
            {
                field: location_time_joint_jsd(
                    tokens_real[field],
                    tokens_real["hour"],
                    tokens_gen[field],
                    tokens_gen["hour"],
                    field_vocab[field],
                    hour_vocab,
                    pad_id=pad_id,
                )
                for field in fields
            }
        )
    return {
        "location_time_joint_jsd": location_time,
        "activity_ngram_jsd": _without_none(
            {
                field: activity_ngram_jsd(
                    tokens_real[field],
                    tokens_gen[field],
                    field_vocab[field],
                    n=3,
                    pad_id=pad_id,
                )
                for field in fields
            }
        ),
    }


def _motif_metrics(
    tokens_real: dict[str, torch.Tensor],
    tokens_gen: dict[str, torch.Tensor],
    day_keys_real: Any | None,
    day_keys_gen: Any | None,
    *,
    pad_id: int,
) -> dict[str, Any]:
    """Daily mobility motifs of the placekey visits; days default to the
    weekday tokens when no day keys are given."""
    if "placekey_id" not in tokens_real or "placekey_id" not in tokens_gen:
        return {}
    if (
        day_keys_real is None
        and day_keys_gen is None
        and "weekday" in tokens_real
        and "weekday" in tokens_gen
    ):
        day_keys_real = tokens_real["weekday"]
        day_keys_gen = tokens_gen["weekday"]
    motifs = mobility_motif_metrics(
        tokens_real["placekey_id"],
        tokens_gen["placekey_id"],
        day_keys_real=day_keys_real,
        day_keys_generated=day_keys_gen,
        pad_id=pad_id,
    )
    return {
        "mobility_motif_jsd": motifs.get("mobility_motif_jsd"),
        "mobility_motifs": motifs,
    }


def compute_trajectory_metrics(
    tokens_real: dict[str, torch.Tensor],
    tokens_gen: dict[str, torch.Tensor],
    field_vocab: dict[str, int],
    *,
    pad_id: int = PAD_ID,
    placekey_latlon: dict[int, tuple[float, float]] | None = None,
    geohash_lookup: dict[int, str] | None = None,
    motif_day_keys_real: Any | None = None,
    motif_day_keys_gen: Any | None = None,
    compute_device: torch.device | str | None = None,
) -> TrajectoryMetrics:
    if compute_device is not None:
        device = torch.device(compute_device)
        tokens_real = {f: v.to(device) for f, v in tokens_real.items()}
        tokens_gen = {f: v.to(device) for f, v in tokens_gen.items()}
    tokens_gen = {
        field: value.to(tokens_real[field].device)
        if field in tokens_real
        else value
        for field, value in tokens_gen.items()
    }
    return TrajectoryMetrics(
        **_per_field_metrics(
            tokens_real, tokens_gen, field_vocab, pad_id=pad_id
        ),
        **_spatial_metrics(
            tokens_real,
            tokens_gen,
            placekey_latlon,
            geohash_lookup,
            pad_id=pad_id,
        ),
        **_activity_metrics(
            tokens_real, tokens_gen, field_vocab, pad_id=pad_id
        ),
        **_motif_metrics(
            tokens_real,
            tokens_gen,
            motif_day_keys_real,
            motif_day_keys_gen,
            pad_id=pad_id,
        ),
        decode_validity=decode_validity_rates(
            tokens_gen, field_vocab, pad_id=pad_id
        ),
    )


def summarize_mobility_metrics(metrics: TrajectoryMetrics) -> dict[str, Any]:
    field_jsd = metrics.field_jsd
    selected = {
        field: float(field_jsd[field])
        for field in _PRIMARY_FIELDS
        if field in field_jsd
    } or {field: float(value) for field, value in field_jsd.items()}
    lt_joint = dict(metrics.location_time_joint_jsd or {})
    ngram = dict(metrics.activity_ngram_jsd or {})
    motif_jsd = metrics.mobility_motif_jsd
    score_terms = [
        *selected.values(),
        *(
            float(value)
            for value in (
                metrics.visit_count_jsd,
                metrics.travel_dist_jsd,
                metrics.geohash_spatial_jsd,
            )
            if value is not None
        ),
        *map(float, lt_joint.values()),
        *map(float, ngram.values()),
    ]
    if motif_jsd is not None:
        score_terms.append(float(motif_jsd))
    return {
        "primary_field_jsd": selected,
        "primary_field_jsd_mean": (
            float(sum(selected.values()) / len(selected)) if selected else None
        ),
        "visit_count_jsd": metrics.visit_count_jsd,
        "travel_dist_jsd": metrics.travel_dist_jsd,
        "geohash_spatial_jsd": metrics.geohash_spatial_jsd,
        "location_time_joint_jsd": lt_joint,
        "activity_ngram_jsd": ngram,
        "mobility_motif_jsd": motif_jsd,
        "mobility_motifs": metrics.mobility_motifs,
        "od_pair_jsd": dict(metrics.od_pair_jsd),
        "token_change_count_jsd": dict(metrics.token_change_count_jsd),
        "decode_validity": metrics.decode_validity,
        "mobility_score": (
            float(sum(score_terms) / len(score_terms)) if score_terms else None
        ),
        "score_direction": "lower_is_better",
    }
