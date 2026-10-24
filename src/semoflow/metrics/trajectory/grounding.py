from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from semoflow.config.constants import PAD_ID
from semoflow.metrics.trajectory.distribution import (
    activity_ngram_jsd,
    distinct_ngram_ratio,
    ngram_counts_hashed,
    self_ngram_overlap,
    transition_distance,
    unique_location_count_jsd,
)
from semoflow.metrics.trajectory.geo import (
    radius_of_gyration_jsd,
    spatial_guardrail_uncertainty,
    spatial_jsd,
    travel_distance_jsd,
)
from semoflow.metrics.trajectory.identity_dynamics import (
    category_identity_dynamics_comparison,
    cluster_histogram_jsd_bootstrap,
    cluster_sparse_ngram_jsd_bootstrap,
    cluster_sparse_token_distribution_bootstrap,
    identity_dynamics_comparison,
)
from semoflow.metrics.trajectory.identity_histograms import (
    identity_histogram_contributions,
    rowwise_jsd_counts,
)
from semoflow.metrics.trajectory.mobility_motifs import mobility_motif_metrics
from semoflow.metrics.trajectory.poi import compute_poi_distribution_metrics
from semoflow.metrics.trajectory.scoring import compute_trajectory_metrics
from semoflow.tokenization import token_space
from semoflow.tokenization.decoding.grounding_index import pack_code_rows

_CANDIDATE_SELECTION_POLICIES = frozenset(
    {"first", "train_prior_argmax", "train_prior_sample", "a4_support"}
)
_CANDIDATE_SELECTION_SEED = 20260830
_CANDIDATE_WEIGHT_SMOOTHING = 1.0
# Columns of an encoded payload row.
_PLACEKEY, _TOP_CAT, _HAS_TOP_CAT, _NAICS4, _HAS_NAICS4 = range(5)
_HAS_LABELS = [_HAS_TOP_CAT, _HAS_NAICS4]


def _token_histogram_contributions(
    tokens: torch.Tensor, *, vocab_size: int, pad_id: int
) -> np.ndarray:
    values = tokens.long().cpu()
    out = np.zeros((values.size(0), vocab_size), dtype=np.int32)
    valid = (values != pad_id) & (values >= 0) & (values < vocab_size)
    rows = torch.arange(values.size(0)).unsqueeze(1).expand_as(values)
    np.add.at(out, (rows[valid].numpy(), values[valid].numpy()), 1)
    return out


def _vectorized_lookup(
    code_rows: np.ndarray,
    grounding_index: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Candidate counts and payload rows of token-space SID rows.

    Rows holding a special token (below ``REAL_TOKEN_OFFSET``) match nothing.
    """
    table_keys, table_counts, table_payloads = (
        grounding_index.packed_lookup_table()
    )
    n_rows = code_rows.shape[0]
    candidate_counts = np.zeros(n_rows, dtype=np.int32)
    payload_indices = np.full(n_rows, -1, dtype=np.int64)
    special = np.any(code_rows < token_space.REAL_TOKEN_OFFSET, axis=1)
    row_indices = np.flatnonzero(~special)
    if row_indices.size and table_keys.size:
        lookup_rows = code_rows[row_indices] - token_space.REAL_TOKEN_OFFSET
        if grounding_index.order_agnostic:
            lookup_rows = np.sort(lookup_rows, axis=1)
        packed = pack_code_rows(lookup_rows)
        positions = np.searchsorted(table_keys, packed, side="left")
        safe_positions = np.minimum(positions, table_keys.size - 1)
        found = (positions < table_keys.size) & (
            table_keys[safe_positions] == packed
        )
        candidate_counts[row_indices[found]] = table_counts[
            safe_positions[found]
        ]
        payload_indices[row_indices[found]] = safe_positions[found]
    return candidate_counts, payload_indices, special, table_payloads


def _label(value: Any, vocab: Mapping[str, int] | None) -> tuple[int, bool]:
    """A label's token id and whether it is known (unknown labels pad)."""
    if value is None or isinstance(value, bool):
        return PAD_ID, False
    if isinstance(value, str):
        encoded = None if vocab is None else vocab.get(value)
        return (PAD_ID, False) if encoded is None else (encoded, True)
    return int(value), True


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    return next(
        (payload[key] for key in keys if payload.get(key) is not None), None
    )


def _encode_payload(
    payload: Mapping[str, Any],
    *,
    category_string_vocab: Mapping[str, int] | None,
    naics_string_vocab: Mapping[str, int] | None,
) -> tuple[int, int, bool, int, bool]:
    placekey = payload.get("placekey_id")
    return (
        PAD_ID if placekey is None else int(placekey),
        *_label(
            _first_present(payload, "top_cat", "top_category"),
            category_string_vocab,
        ),
        *_label(
            _first_present(payload, "naics4", "naics_code"),
            naics_string_vocab,
        ),
    )


def _candidate_probabilities(
    group: Sequence[Mapping[str, Any]], weights: Mapping[int, float]
) -> np.ndarray:
    raw = np.array(
        [
            float(weights.get(int(payload["placekey_id"]), 0.0))
            if payload.get("placekey_id") is not None
            else 0.0
            for payload in group
        ],
        dtype=np.float64,
    )
    raw += _CANDIDATE_WEIGHT_SMOOTHING
    total = raw.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(raw.size, 1.0 / raw.size)
    return raw / total


def _a4_support_picks(
    encoded: np.ndarray,
    probabilities: np.ndarray,
    count: int,
    emitted: set[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Give each occurrence an unemitted candidate, most probable first."""
    unused = sorted(
        (
            index
            for index in range(len(encoded))
            if int(encoded[index, _PLACEKEY]) not in emitted
        ),
        key=lambda index: -probabilities[index],
    )
    picks = unused[:count]
    emitted.update(int(encoded[index, _PLACEKEY]) for index in picks)
    if len(picks) < count:
        picks.extend(
            int(value)
            for value in rng.choice(
                len(encoded), size=count - len(picks), p=probabilities
            )
        )
    return np.asarray(picks, dtype=np.int64)


def _sid_code_rows(
    sid_tokens: Mapping[str, torch.Tensor], key_columns: Sequence[str]
) -> tuple[torch.Size, np.ndarray]:
    missing = [field for field in key_columns if field not in sid_tokens]
    if missing:
        raise KeyError(f"missing SID token fields for grounding: {missing}")
    shape = sid_tokens[key_columns[0]].shape
    for field in key_columns[1:]:
        if sid_tokens[field].shape != shape:
            raise ValueError(
                f"SID field {field!r} shape {tuple(sid_tokens[field].shape)} "
                f"does not match {tuple(shape)}"
            )
    codes = np.stack(
        [
            sid_tokens[field].long().reshape(-1).cpu().numpy()
            for field in key_columns
        ],
        axis=1,
    ).astype(np.int64, copy=False)
    return shape, codes


def _preferred_payloads(
    preferred_placekey_ids: np.ndarray,
    active_codes: np.ndarray,
    grounding_index: Any,
) -> list[Mapping[str, Any]]:
    """Each active position's preferred POI payload.

    The preferred POI must be a candidate of the position's SID.
    """
    by_placekey: dict[int, tuple[tuple[int, ...], dict[str, Any]]] = {}
    for key, payloads in grounding_index.iter_key_payload_groups():
        for payload in payloads:
            placekey_id = payload.get("placekey_id")
            if placekey_id is not None:
                by_placekey[int(placekey_id)] = (
                    tuple(
                        int(value) + token_space.REAL_TOKEN_OFFSET
                        for value in key
                    ),
                    dict(payload),
                )
    payloads: list[Mapping[str, Any]] = []
    for offset, placekey_id in enumerate(preferred_placekey_ids.tolist()):
        selected = by_placekey.get(placekey_id)
        if selected is None:
            raise ValueError(
                f"preferred placekey_id {placekey_id} is not in the "
                "grounding index"
            )
        key, payload = selected
        row = tuple(int(value) for value in active_codes[offset])
        if key != row:
            raise ValueError(
                f"preferred placekey_id {placekey_id} does not belong "
                f"to SID row {row}"
            )
        payloads.append(payload)
    return payloads


def _unique_code_rows(
    codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct code rows, each row's index into them, and their counts."""
    if np.any(codes < 0) or np.any(codes > 255):
        return np.unique(
            codes, axis=0, return_inverse=True, return_counts=True
        )
    packed, inverse, counts = np.unique(
        pack_code_rows(codes), return_inverse=True, return_counts=True
    )
    unique = np.empty((packed.shape[0], codes.shape[1]), dtype=np.int64)
    for column in range(codes.shape[1]):
        unique[:, column] = (packed >> (8 * column)) & 255
    return unique, inverse, counts


def _grounding_stats(
    candidate_counts: np.ndarray, special: np.ndarray, occurrences: np.ndarray
) -> dict[str, float]:
    candidates = candidate_counts[~special].astype(np.int64)
    groundable = occurrences[~special].astype(np.int64)
    n_positions = int(occurrences.sum())
    n_valid = int(groundable[candidates > 0].sum())
    n_ambiguous = int(groundable[candidates > 1].sum())
    n_candidates = int((candidates * groundable)[candidates > 0].sum())
    return {
        "n_positions": float(n_positions),
        "valid_rate": n_valid / n_positions if n_positions else 0.0,
        "ambiguous_rate": n_ambiguous / n_valid if n_valid else 0.0,
        "mean_candidates_per_valid": (
            n_candidates / n_valid if n_valid else 0.0
        ),
    }


def ground_sid_tokens_to_poi_tokens(
    sid_tokens: dict[str, torch.Tensor],
    grounding_index: Any,
    *,
    category_string_vocab: dict[str, int] | None = None,
    naics_string_vocab: dict[str, int] | None = None,
    preferred_placekey_ids: torch.Tensor | None = None,
    candidate_selection: str = "first",
    candidate_weights: Mapping[int, float] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Ground SID tokens to placekey/top_cat/naics4 tokens.

    Each SID resolves to the grounding index's first candidate, or to a
    train-prior choice under ``candidate_selection``; ``a4_support`` and
    ``train_prior_sample`` choose per occurrence. ``preferred_placekey_ids``
    pins every position to a known POI of its SID instead.
    """
    if candidate_selection not in _CANDIDATE_SELECTION_POLICIES:
        raise ValueError(
            f"unknown candidate_selection {candidate_selection!r}; "
            f"expected one of {sorted(_CANDIDATE_SELECTION_POLICIES)}"
        )
    if candidate_selection != "first" and candidate_weights is None:
        raise ValueError(
            f"candidate_selection={candidate_selection!r} requires "
            "candidate_weights fitted on training data"
        )
    shape, codes = _sid_code_rows(sid_tokens, grounding_index.key_columns)
    active = np.flatnonzero(~(codes == PAD_ID).all(axis=1))
    active_codes = codes[active]
    preferred: list[Mapping[str, Any]] | None = None
    if preferred_placekey_ids is not None:
        if preferred_placekey_ids.shape != shape:
            raise ValueError(
                "preferred_placekey_ids shape must match SID tokens; got "
                f"{tuple(preferred_placekey_ids.shape)} and {tuple(shape)}"
            )
        preferred = _preferred_payloads(
            preferred_placekey_ids.long().reshape(-1).cpu().numpy()[active],
            active_codes,
            grounding_index,
        )
    unique_codes, inverse, counts = _unique_code_rows(active_codes)
    candidate_counts, payload_indices, special, first_payloads = (
        _vectorized_lookup(unique_codes, grounding_index)
    )
    encode = functools.partial(
        _encode_payload,
        category_string_vocab=category_string_vocab,
        naics_string_vocab=naics_string_vocab,
    )
    invalid_row = np.array([PAD_ID, PAD_ID, 0, PAD_ID, 0], dtype=np.int64)

    hit = np.flatnonzero(payload_indices >= 0)
    groups = (
        grounding_index.packed_lookup_groups()
        if candidate_selection != "first" and preferred is None
        else None
    )
    unique_encoded = np.tile(invalid_row, (len(unique_codes), 1))
    for row in hit:
        index = int(payload_indices[row])
        payload = first_payloads[index]
        if groups is not None and len(groups[index]) > 1:
            assert candidate_weights is not None
            group = groups[index]
            weights = _candidate_probabilities(group, candidate_weights)
            payload = group[int(np.argmax(weights))]
        unique_encoded[row] = encode(payload)

    encoded = np.tile(invalid_row, (len(codes), 1))
    encoded[active] = unique_encoded[inverse]
    has_labels = unique_encoded[:, _HAS_LABELS].any(axis=0)
    if (
        groups is not None
        and candidate_selection in ("a4_support", "train_prior_sample")
        and hit.size
    ):
        assert candidate_weights is not None
        rng = np.random.default_rng(_CANDIDATE_SELECTION_SEED)
        order = np.argsort(inverse, kind="stable")
        starts = np.searchsorted(inverse[order], hit, side="left")
        ends = np.searchsorted(inverse[order], hit, side="right")
        emitted: set[int] = set()
        for row, start, end in zip(hit, starts, ends, strict=True):
            group = groups[int(payload_indices[row])]
            if candidate_selection == "a4_support":
                group_encoded = np.array(
                    [encode(payload) for payload in group], dtype=np.int64
                )
                if len(group) < 2:
                    emitted.add(int(group_encoded[0, _PLACEKEY]))
                    continue
                picks = _a4_support_picks(
                    group_encoded,
                    _candidate_probabilities(group, candidate_weights),
                    int(end - start),
                    emitted,
                    rng,
                )
            else:
                if len(group) < 2:
                    continue
                probabilities = _candidate_probabilities(
                    group, candidate_weights
                )
                group_encoded = np.array(
                    [encode(payload) for payload in group], dtype=np.int64
                )
                picks = rng.choice(
                    len(group), size=int(end - start), p=probabilities
                )
            encoded[active[order[start:end]]] = group_encoded[picks]
            has_labels |= group_encoded[picks][:, _HAS_LABELS].any(axis=0)
    if preferred:
        preferred_encoded = np.array(
            [encode(payload) for payload in preferred], dtype=np.int64
        )
        encoded[active] = preferred_encoded
        has_labels |= preferred_encoded[:, _HAS_LABELS].any(axis=0)

    def _tokens(column: int) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(encoded[:, column])).view(
            *shape
        )

    tokens = {"placekey_id": _tokens(_PLACEKEY)}
    if has_labels[0]:
        tokens["top_cat"] = _tokens(_TOP_CAT)
    if has_labels[1]:
        tokens["naics4"] = _tokens(_NAICS4)
    return tokens, _grounding_stats(candidate_counts, special, counts)


def _field_vocab(
    real_poi: Mapping[str, torch.Tensor],
    gen_poi: Mapping[str, torch.Tensor],
    *,
    placekey_vocab_size: int | None,
    category_vocab_size: int | None,
) -> dict[str, int]:
    def observed(field: str) -> int:
        largest = max(
            [
                0,
                *(
                    int(tokens.max().item())
                    for tokens in (real_poi[field], gen_poi[field])
                    if tokens.numel()
                ),
            ]
        )
        return largest + 1

    vocab = {"placekey_id": placekey_vocab_size or observed("placekey_id")}
    if "top_cat" in real_poi and "top_cat" in gen_poi:
        vocab["top_cat"] = category_vocab_size or observed("top_cat")
    if "naics4" in real_poi and "naics4" in gen_poi:
        vocab["naics4"] = observed("naics4")
    return vocab


def _brief_poi_metrics(
    real_poi: Mapping[str, torch.Tensor],
    gen_poi: Mapping[str, torch.Tensor],
    metric_real_poi: dict[str, torch.Tensor],
    metric_gen_poi: dict[str, torch.Tensor],
    field_vocab: dict[str, int],
    *,
    placekey_latlon: dict[int, tuple[float, float]] | None,
    geohash_lookup: dict[int, str] | None,
    motif_day_keys_real: Any | None,
    motif_day_keys_gen: Any | None,
) -> dict[str, Any]:
    """The six headline comparison metrics plus POI coverage."""
    real_placekeys = metric_real_poi["placekey_id"]
    generated_placekeys = metric_gen_poi["placekey_id"]
    poi_distribution = compute_poi_distribution_metrics(
        metric_real_poi, metric_gen_poi, field_vocab, pad_id=PAD_ID
    )
    poi_values = (
        poi_distribution.to_dict() if poi_distribution is not None else {}
    )
    return_gaps = [
        identity_histogram_contributions(tokens["placekey_id"], pad_id=PAD_ID)[
            "return_gap_distribution_jsd"
        ].sum(axis=0, keepdims=True)
        for tokens in (real_poi, gen_poi)
    ]
    with_coordinates = placekey_latlon is not None
    return {
        "trajectory_metrics": {
            "geohash_spatial_jsd": (
                spatial_jsd(
                    real_placekeys,
                    generated_placekeys,
                    geohash_lookup,
                    pad_id=PAD_ID,
                )
                if geohash_lookup is not None
                else None
            ),
            "travel_dist_jsd": (
                travel_distance_jsd(
                    real_placekeys,
                    generated_placekeys,
                    placekey_latlon,
                    pad_id=PAD_ID,
                )
                if with_coordinates
                else None
            ),
            "mobility_motif_jsd": mobility_motif_metrics(
                real_placekeys,
                generated_placekeys,
                day_keys_real=motif_day_keys_real,
                day_keys_generated=motif_day_keys_gen,
                pad_id=PAD_ID,
            )["mobility_motif_jsd"],
        },
        "poi_distribution": {
            "poi_frequency_jsd": poi_values.get("poi_frequency_jsd"),
            "poi_coverage": poi_values.get("poi_coverage"),
        },
        "spatial_metrics": {
            "radius_of_gyration_jsd": (
                radius_of_gyration_jsd(
                    real_placekeys,
                    generated_placekeys,
                    placekey_latlon,
                    pad_id=PAD_ID,
                )
                if with_coordinates
                else None
            )
        },
        "identity_dynamics": {
            "poi": {
                "distribution_distances": {
                    "return_gap_distribution_jsd": {
                        "point_estimate": float(
                            rowwise_jsd_counts(*return_gaps)[0]
                        ),
                    }
                }
            }
        },
    }


def _category_metrics(
    real: torch.Tensor,
    generated: torch.Tensor,
    metric_real: torch.Tensor,
    metric_generated: torch.Tensor,
    vocab_size: int,
    *,
    cluster_ids: Sequence[str] | None,
    compute_device: torch.device | str | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "top_cat_transition_distance": transition_distance(
            real, generated, vocab_size, pad_id=PAD_ID
        )
    }
    uncertainty: dict[str, Any] = {
        "top_cat_unigram_jsd": cluster_histogram_jsd_bootstrap(
            _token_histogram_contributions(
                metric_real, vocab_size=vocab_size, pad_id=PAD_ID
            ),
            _token_histogram_contributions(
                metric_generated, vocab_size=vocab_size, pad_id=PAD_ID
            ),
            cluster_ids=cluster_ids,
            compute_device=compute_device,
        )
    }
    ngram = functools.partial(ngram_counts_hashed, pad_id=PAD_ID)
    for order in (2, 3):
        real_counts = ngram(metric_real, vocab_size, n=order)
        generated_counts = ngram(metric_generated, vocab_size, n=order)
        metrics[f"top_cat_ngram_jsd_n{order}"] = activity_ngram_jsd(
            metric_real,
            metric_generated,
            vocab_size,
            n=order,
            pad_id=PAD_ID,
            real_counts=real_counts,
            gen_counts=generated_counts,
        )
        for side, tokens, counts in (
            ("real", metric_real, real_counts),
            ("gen", metric_generated, generated_counts),
        ):
            metrics[f"top_cat_distinct_ratio_n{order}_{side}"] = (
                distinct_ngram_ratio(
                    tokens, vocab_size, n=order, pad_id=PAD_ID, counts=counts
                )
            )
        ngram_uncertainty = cluster_sparse_ngram_jsd_bootstrap(
            metric_real,
            metric_generated,
            vocab_size=vocab_size,
            n=order,
            pad_id=PAD_ID,
            cluster_ids=cluster_ids,
            compute_device=compute_device,
        )
        if ngram_uncertainty is not None:
            uncertainty[f"top_cat_ngram_jsd_n{order}"] = ngram_uncertainty
    for side, tokens in (("gen", metric_generated), ("real", metric_real)):
        metrics[f"top_cat_self_overlap_n3_{side}"] = self_ngram_overlap(
            tokens, vocab_size, n=3, pad_id=PAD_ID
        )
    metrics["uncertainty"] = uncertainty
    return metrics


def _spatial_metrics(
    real: torch.Tensor,
    generated: torch.Tensor,
    metric_real: torch.Tensor,
    metric_generated: torch.Tensor,
    placekey_latlon: dict[int, tuple[float, float]],
    *,
    geohash_lookup: dict[int, str] | None,
    cluster_ids: Sequence[str] | None,
    compute_device: torch.device | str | None,
) -> dict[str, Any]:
    return {
        "radius_of_gyration_jsd": radius_of_gyration_jsd(
            metric_real, metric_generated, placekey_latlon, pad_id=PAD_ID
        ),
        "unique_location_count_jsd": unique_location_count_jsd(
            metric_real, metric_generated, pad_id=PAD_ID
        ),
        "uncertainty": spatial_guardrail_uncertainty(
            real,
            generated,
            placekey_latlon,
            geohash_lookup=geohash_lookup,
            cluster_ids=cluster_ids,
            compute_device=compute_device,
            pad_id=PAD_ID,
        ),
    }


def compute_decoded_sid_poi_metrics(
    real_poi: dict[str, torch.Tensor],
    gen_poi: dict[str, torch.Tensor],
    *,
    real_stats: Mapping[str, float],
    gen_stats: Mapping[str, float],
    placekey_latlon: dict[int, tuple[float, float]] | None = None,
    geohash_lookup: dict[int, str] | None = None,
    placekey_vocab_size: int | None = None,
    category_vocab_size: int | None = None,
    cluster_ids: Sequence[str] | None = None,
    motif_day_keys_real: Any | None = None,
    motif_day_keys_gen: Any | None = None,
    compute_device: torch.device | str | None = None,
    brief_metrics: bool = False,
) -> dict[str, Any]:
    """Compare grounded real and generated POI tokens."""
    field_vocab = _field_vocab(
        real_poi,
        gen_poi,
        placekey_vocab_size=placekey_vocab_size,
        category_vocab_size=category_vocab_size,
    )
    metric_real_poi, metric_gen_poi = real_poi, gen_poi
    if compute_device is not None:
        device = torch.device(compute_device)
        metric_real_poi = {f: v.to(device) for f, v in real_poi.items()}
        metric_gen_poi = {f: v.to(device) for f, v in gen_poi.items()}
    if brief_metrics:
        return _brief_poi_metrics(
            real_poi,
            gen_poi,
            metric_real_poi,
            metric_gen_poi,
            field_vocab,
            placekey_latlon=placekey_latlon,
            geohash_lookup=geohash_lookup,
            motif_day_keys_real=motif_day_keys_real,
            motif_day_keys_gen=motif_day_keys_gen,
        )

    trajectory = compute_trajectory_metrics(
        metric_real_poi,
        metric_gen_poi,
        field_vocab,
        pad_id=PAD_ID,
        placekey_latlon=placekey_latlon,
        geohash_lookup=geohash_lookup,
        motif_day_keys_real=motif_day_keys_real,
        motif_day_keys_gen=motif_day_keys_gen,
    )
    poi_distribution = compute_poi_distribution_metrics(
        metric_real_poi, metric_gen_poi, field_vocab, pad_id=PAD_ID
    )
    has_categories = "top_cat" in field_vocab
    category_metrics = (
        _category_metrics(
            real_poi["top_cat"],
            gen_poi["top_cat"],
            metric_real_poi["top_cat"],
            metric_gen_poi["top_cat"],
            field_vocab["top_cat"],
            cluster_ids=cluster_ids,
            compute_device=compute_device,
        )
        if has_categories
        else None
    )
    spatial_metrics = (
        _spatial_metrics(
            real_poi["placekey_id"],
            gen_poi["placekey_id"],
            metric_real_poi["placekey_id"],
            metric_gen_poi["placekey_id"],
            placekey_latlon,
            geohash_lookup=geohash_lookup,
            cluster_ids=cluster_ids,
            compute_device=compute_device,
        )
        if placekey_latlon is not None
        else None
    )
    identity_dynamics = identity_dynamics_comparison(
        real_poi["placekey_id"],
        gen_poi["placekey_id"],
        pad_id=PAD_ID,
        kind="poi",
        cluster_ids=cluster_ids,
        compute_device=compute_device,
    )
    identity_dynamics["category"] = (
        category_identity_dynamics_comparison(
            real_poi["top_cat"],
            gen_poi["top_cat"],
            pad_id=PAD_ID,
            cluster_ids=cluster_ids,
            compute_device=compute_device,
        )
        if has_categories
        else None
    )
    poi_values = (
        poi_distribution.to_dict() if poi_distribution is not None else None
    )
    if poi_values is not None:
        poi_values["uncertainty"] = (
            cluster_sparse_token_distribution_bootstrap(
                real_poi["placekey_id"],
                gen_poi["placekey_id"],
                vocab_size=field_vocab["placekey_id"],
                pad_id=PAD_ID,
                cluster_ids=cluster_ids,
                compute_device=compute_device,
            )
        )
    return {
        "trajectory_metrics": trajectory.to_dict(),
        "poi_distribution": poi_values,
        "category_metrics": category_metrics,
        "spatial_metrics": spatial_metrics,
        "identity_dynamics": identity_dynamics,
        "grounding": {
            "invalid_token_id": PAD_ID,
            "valid_rate_real": real_stats["valid_rate"],
            "valid_rate_gen": gen_stats["valid_rate"],
            "ambiguous_rate_real": real_stats["ambiguous_rate"],
            "ambiguous_rate_gen": gen_stats["ambiguous_rate"],
            "mean_candidates_per_valid_real": real_stats[
                "mean_candidates_per_valid"
            ],
            "mean_candidates_per_valid_gen": gen_stats[
                "mean_candidates_per_valid"
            ],
            "n_positions": int(gen_stats["n_positions"]),
        },
    }
