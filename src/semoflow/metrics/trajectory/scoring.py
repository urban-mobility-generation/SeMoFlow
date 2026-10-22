from __future__ import annotations

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
    token_sequence_min_distance,
    transition_distance,
    visit_count_jsd,
)
from semoflow.metrics.trajectory.geo import (
    centroid_geo_error,
    spatial_jsd,
    travel_distance_jsd,
)
from semoflow.metrics.trajectory.mobility_motifs import (
    mobility_motif_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryMetrics:
    field_jsd: dict[str, float]
    field_transition_distance: dict[str, float]
    visit_count_jsd: float
    od_pair_jsd: dict[str, float]
    token_change_count_jsd: dict[str, float]
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
        return {
            "field_jsd": dict(self.field_jsd),
            "field_transition_distance": dict(self.field_transition_distance),
            "visit_count_jsd": self.visit_count_jsd,
            "od_pair_jsd": dict(self.od_pair_jsd),
            "token_change_count_jsd": dict(self.token_change_count_jsd),
            "token_min_distance": self.token_min_distance,
            "geo_error_km": self.geo_error_km,
            "travel_dist_jsd": self.travel_dist_jsd,
            "geohash_spatial_jsd": self.geohash_spatial_jsd,
            "location_time_joint_jsd": self.location_time_joint_jsd,
            "activity_ngram_jsd": self.activity_ngram_jsd,
            "mobility_motif_jsd": self.mobility_motif_jsd,
            "mobility_motifs": self.mobility_motifs,
            "decode_validity": self.decode_validity,
        }


def decode_validity_rates(
    tokens: dict[str, torch.Tensor],
    field_vocab: dict[str, int],
    *,
    pad_id: int = PAD_ID,
    invalid_ids: tuple[int, ...] = (MASK_ID,),
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for field, tensor in tokens.items():
        vocab_size = field_vocab.get(field)
        if vocab_size is None:
            continue
        invalid = (tensor < 0) | (tensor >= vocab_size)
        for token_id in invalid_ids:
            invalid |= tensor == token_id
        scored = tensor != pad_id
        denom = int(scored.sum().item())
        rates[field] = (
            1.0
            if denom == 0
            else float(((~invalid) & scored).sum().item() / denom)
        )
    return rates


def compute_trajectory_metrics(
    tokens_real: dict[str, torch.Tensor],
    tokens_gen: dict[str, torch.Tensor],
    field_vocab: dict[str, int],
    *,
    pad_id: int = PAD_ID,
    include_fields: set[str] | None = None,
    exclude_fields: set[str] | None = None,
    max_vocab_for_transition: int = 5000,
    max_vocab_for_od: int = 500,
    tokens_train: dict[str, torch.Tensor] | None = None,
    placekey_latlon: dict[int, tuple[float, float]] | None = None,
    geohash_lookup: dict[int, str] | None = None,
    motif_day_keys_real: Any | None = None,
    motif_day_keys_gen: Any | None = None,
    motif_frequency_threshold: float = 0.005,
    motif_location_field: str = "placekey_id",
    compute_device: torch.device | str | None = None,
) -> TrajectoryMetrics:
    if compute_device is not None:
        resolved_device = torch.device(compute_device)
        tokens_real = {
            field: value.to(resolved_device)
            for field, value in tokens_real.items()
        }
        tokens_gen = {
            field: value.to(resolved_device)
            for field, value in tokens_gen.items()
        }
        if tokens_train is not None:
            tokens_train = {
                field: value.to(resolved_device)
                for field, value in tokens_train.items()
            }
    tokens_gen = {
        field: value.to(tokens_real[field].device)
        if field in tokens_real and value.device != tokens_real[field].device
        else value
        for field, value in tokens_gen.items()
    }
    if tokens_train is not None:
        tokens_train = {
            field: value.to(tokens_real[field].device)
            if field in tokens_real
            and value.device != tokens_real[field].device
            else value
            for field, value in tokens_train.items()
        }

    field_jsd: dict[str, float] = {}
    field_td: dict[str, float] = {}
    od_jsd: dict[str, float] = {}
    rc_jsd: dict[str, float] = {}

    canonical_field: str | None = None

    for field, vocab_size in field_vocab.items():
        if include_fields is not None and field not in include_fields:
            continue
        if exclude_fields is not None and field in exclude_fields:
            continue
        if field not in tokens_real or field not in tokens_gen:
            continue

        if canonical_field is None:
            canonical_field = field

        p = marginal_distribution(
            tokens_real[field], vocab_size, pad_id=pad_id
        )
        q = marginal_distribution(tokens_gen[field], vocab_size, pad_id=pad_id)
        field_jsd[field] = jsd(p, q)

        if vocab_size > max_vocab_for_transition:
            logger.debug(
                "skipping transition distance for %s: vocab_size=%d > %d",
                field,
                vocab_size,
                max_vocab_for_transition,
            )
        else:
            field_td[field] = transition_distance(
                tokens_real[field],
                tokens_gen[field],
                vocab_size,
                pad_id=pad_id,
            )

        od_result = od_pair_jsd(
            tokens_real[field],
            tokens_gen[field],
            vocab_size,
            pad_id=pad_id,
            max_vocab=max_vocab_for_od,
        )
        if od_result is not None:
            od_jsd[field] = od_result

        rc_jsd[field] = token_change_count_jsd(
            tokens_real[field], tokens_gen[field], pad_id=pad_id
        )

    vc_jsd: float = 0.0
    if canonical_field is not None:
        vc_jsd = visit_count_jsd(
            tokens_real[canonical_field],
            tokens_gen[canonical_field],
            pad_id=pad_id,
        )

    min_dist: float | None = None
    if tokens_train is not None and canonical_field is not None:
        train_field = tokens_train.get(canonical_field)
        if train_field is not None:
            min_dist = token_sequence_min_distance(
                tokens_gen[canonical_field],
                train_field,
                pad_id=pad_id,
            )

    geo_err: float | None = None
    trav_jsd: float | None = None
    if (
        placekey_latlon is not None
        and "placekey_id" in tokens_real
        and "placekey_id" in tokens_gen
    ):
        trav_jsd = travel_distance_jsd(
            tokens_real["placekey_id"],
            tokens_gen["placekey_id"],
            placekey_latlon,
            pad_id=pad_id,
        )
        geo_err = centroid_geo_error(
            tokens_real["placekey_id"],
            tokens_gen["placekey_id"],
            placekey_latlon,
            pad_id=pad_id,
        )

    sp_jsd: float | None = None
    if (
        geohash_lookup is not None
        and "placekey_id" in tokens_real
        and "placekey_id" in tokens_gen
    ):
        sp_jsd = spatial_jsd(
            tokens_real["placekey_id"],
            tokens_gen["placekey_id"],
            geohash_lookup,
            pad_id=pad_id,
        )

    validity = decode_validity_rates(tokens_gen, field_vocab, pad_id=pad_id)

    lt_jsd: dict[str, float] | None = None
    _location_fields = ("top_cat", "naics4", "semantic_id")
    if "hour" in tokens_real and "hour" in tokens_gen:
        hour_vocab = field_vocab.get("hour", 0)
        if hour_vocab > 0:
            lt_results: dict[str, float] = {}
            for loc_field in _location_fields:
                if loc_field not in tokens_real or loc_field not in tokens_gen:
                    continue
                if loc_field not in field_vocab:
                    continue
                result = location_time_joint_jsd(
                    tokens_real[loc_field],
                    tokens_real["hour"],
                    tokens_gen[loc_field],
                    tokens_gen["hour"],
                    field_vocab[loc_field],
                    hour_vocab,
                    pad_id=pad_id,
                )
                if result is not None:
                    lt_results[loc_field] = result
            if lt_results:
                lt_jsd = lt_results

    ngram_jsd: dict[str, float] | None = None
    _ngram_fields = ("top_cat", "naics4", "semantic_id")
    ngram_results: dict[str, float] = {}
    for nf in _ngram_fields:
        if (
            nf not in tokens_real
            or nf not in tokens_gen
            or nf not in field_vocab
        ):
            continue
        result = activity_ngram_jsd(
            tokens_real[nf],
            tokens_gen[nf],
            field_vocab[nf],
            n=3,
            pad_id=pad_id,
        )
        if result is not None:
            ngram_results[nf] = result
    if ngram_results:
        ngram_jsd = ngram_results

    motifs: dict[str, Any] | None = None
    if (
        motif_location_field in tokens_real
        and motif_location_field in tokens_gen
    ):
        real_day_keys = motif_day_keys_real
        gen_day_keys = motif_day_keys_gen
        if (
            real_day_keys is None
            and gen_day_keys is None
            and "weekday" in tokens_real
            and "weekday" in tokens_gen
        ):
            real_day_keys = tokens_real["weekday"]
            gen_day_keys = tokens_gen["weekday"]
        motifs = mobility_motif_metrics(
            tokens_real[motif_location_field],
            tokens_gen[motif_location_field],
            day_keys_real=real_day_keys,
            day_keys_generated=gen_day_keys,
            frequency_threshold=motif_frequency_threshold,
            pad_id=pad_id,
        )

    return TrajectoryMetrics(
        field_jsd=field_jsd,
        field_transition_distance=field_td,
        visit_count_jsd=vc_jsd,
        od_pair_jsd=od_jsd,
        token_change_count_jsd=rc_jsd,
        token_min_distance=min_dist,
        geo_error_km=geo_err,
        travel_dist_jsd=trav_jsd,
        geohash_spatial_jsd=sp_jsd,
        location_time_joint_jsd=lt_jsd,
        activity_ngram_jsd=ngram_jsd,
        mobility_motif_jsd=(
            motifs.get("mobility_motif_jsd") if motifs is not None else None
        ),
        mobility_motifs=motifs,
        decode_validity=validity,
    )


def summarize_mobility_metrics(
    metrics: TrajectoryMetrics | dict[str, Any],
    *,
    primary_fields: tuple[str, ...] = (
        "hour",
        "weekday",
        "month",
        "top_cat",
        "naics4",
        "semantic_id",
        "semantic_id_token",
    ),
) -> dict[str, Any]:
    payload = (
        metrics.to_dict()
        if isinstance(metrics, TrajectoryMetrics)
        else metrics
    )
    field_jsd = dict(payload.get("field_jsd") or {})
    if not any(field in field_jsd for field in primary_fields):
        selected = {field: float(v) for field, v in field_jsd.items()}
    else:
        selected = {
            field: float(field_jsd[field])
            for field in primary_fields
            if field in field_jsd
        }
    od_pair = dict(payload.get("od_pair_jsd") or {})
    _change = dict(payload.get("token_change_count_jsd") or {})
    lt_joint = dict(payload.get("location_time_joint_jsd") or {})
    ngram = dict(payload.get("activity_ngram_jsd") or {})
    motifs = payload.get("mobility_motifs")
    motif_jsd = payload.get("mobility_motif_jsd")
    if motif_jsd is None and isinstance(motifs, dict):
        motif_jsd = motifs.get("mobility_motif_jsd")
    score_terms = list(selected.values())
    for key in ("visit_count_jsd", "travel_dist_jsd", "geohash_spatial_jsd"):
        value = payload.get(key)
        if value is not None:
            score_terms.append(float(value))
    for v in lt_joint.values():
        score_terms.append(float(v))
    for v in ngram.values():
        score_terms.append(float(v))
    if motif_jsd is not None:
        score_terms.append(float(motif_jsd))
    score = (
        None
        if not score_terms
        else float(sum(score_terms) / max(len(score_terms), 1))
    )
    return {
        "primary_field_jsd": selected,
        "primary_field_jsd_mean": (
            None
            if not selected
            else float(sum(selected.values()) / len(selected))
        ),
        "visit_count_jsd": payload.get("visit_count_jsd"),
        "travel_dist_jsd": payload.get("travel_dist_jsd"),
        "geohash_spatial_jsd": payload.get("geohash_spatial_jsd"),
        "location_time_joint_jsd": lt_joint,
        "activity_ngram_jsd": ngram,
        "mobility_motif_jsd": motif_jsd,
        "mobility_motifs": motifs,
        "od_pair_jsd": od_pair,
        "token_change_count_jsd": _change,
        "decode_validity": payload.get("decode_validity"),
        "mobility_score": score,
        "score_direction": "lower_is_better",
    }
