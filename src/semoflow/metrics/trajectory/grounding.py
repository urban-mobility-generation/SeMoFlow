from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from semoflow.config.constants import PAD_ID
from semoflow.metrics.grounding_quality import (
    GroundingQualityTally,
)
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
from semoflow.metrics.trajectory.mobility_motifs import (
    mobility_motif_metrics,
)
from semoflow.metrics.trajectory.poi import (
    compute_poi_distribution_metrics,
)
from semoflow.metrics.trajectory.scoring import (
    compute_trajectory_metrics,
)
from semoflow.tokenization import token_space


def _pack_lookup_rows(rows: np.ndarray) -> np.ndarray:
    packed = np.zeros(rows.shape[0], dtype=np.uint64)
    for column in range(rows.shape[1]):
        packed |= rows[:, column].astype(np.uint64) << (8 * column)
    return packed


def _token_histogram_contributions(
    tokens: torch.Tensor, *, vocab_size: int, pad_id: int
) -> np.ndarray:
    values = tokens.long().cpu()
    out = np.zeros((values.size(0), vocab_size), dtype=np.int32)
    valid = (values != pad_id) & (values >= 0) & (values < vocab_size)
    rows = torch.arange(values.size(0)).unsqueeze(1).expand_as(values)
    np.add.at(out, (rows[valid].numpy(), values[valid].numpy()), 1)
    return out


_CANDIDATE_SELECTION_POLICIES = frozenset(
    {"first", "train_prior_argmax", "train_prior_sample", "a4_support"}
)


def _vectorized_flexible_lookup(
    code_rows: np.ndarray,
    grounding_index: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    table_keys, table_counts, table_payloads = (
        grounding_index.packed_lookup_table()
    )
    n_rows = code_rows.shape[0]
    candidate_counts = np.zeros(n_rows, dtype=np.int32)
    payload_indices = np.full(n_rows, -1, dtype=np.int64)
    raw_hits = np.zeros(n_rows, dtype=bool)

    def _probe(rows: np.ndarray, valid: np.ndarray) -> np.ndarray:
        hits = np.zeros(n_rows, dtype=bool)
        if not valid.any() or not table_keys.size:
            return hits
        row_indices = np.flatnonzero(valid)
        lookup_rows = rows[row_indices]
        if grounding_index.order_agnostic:
            lookup_rows = np.sort(lookup_rows, axis=1)
        packed = _pack_lookup_rows(lookup_rows)
        positions = np.searchsorted(table_keys, packed, side="left")
        safe_positions = np.minimum(positions, table_keys.size - 1)
        found = (positions < table_keys.size) & (
            table_keys[safe_positions] == packed
        )
        if found.any():
            found_rows = row_indices[found]
            found_positions = safe_positions[found]
            candidate_counts[found_rows] = table_counts[found_positions]
            payload_indices[found_rows] = found_positions
            hits[found_rows] = True
        return hits

    raw_valid = np.all(code_rows >= 0, axis=1)
    raw_hits = _probe(code_rows, raw_valid)
    shifted_valid = (~raw_hits) & np.all(
        code_rows >= token_space.REAL_TOKEN_OFFSET, axis=1
    )
    if shifted_valid.any():
        shifted_rows = code_rows - token_space.REAL_TOKEN_OFFSET
        _probe(shifted_rows, shifted_valid)

    special = (~raw_hits) & np.any(
        code_rows < token_space.REAL_TOKEN_OFFSET, axis=1
    )
    return candidate_counts, payload_indices, special, table_payloads


def ground_sid_tokens_to_poi_tokens(
    sid_tokens: dict[str, torch.Tensor],
    grounding_index: Any,
    *,
    invalid_token_id: int = PAD_ID,
    category_string_vocab: dict[str, int] | None = None,
    naics_string_vocab: dict[str, int] | None = None,
    preferred_placekey_ids: torch.Tensor | None = None,
    candidate_selection: str = "first",
    candidate_weights: Mapping[int, float] | None = None,
    candidate_weight_smoothing: float = 1.0,
    candidate_selection_seed: int = 20260830,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
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
    code_fields = [
        field for field in grounding_index.key_columns if field in sid_tokens
    ]
    if len(code_fields) != len(grounding_index.key_columns):
        missing = [
            field
            for field in grounding_index.key_columns
            if field not in sid_tokens
        ]
        raise KeyError(f"missing SID token fields for grounding: {missing}")

    shape = sid_tokens[code_fields[0]].shape
    for field in code_fields[1:]:
        if sid_tokens[field].shape != shape:
            raise ValueError(
                f"SID field {field!r} shape {tuple(sid_tokens[field].shape)} "
                f"does not match {tuple(shape)}"
            )

    n_flat = int(np.prod(shape))
    flat_placekey = np.full(n_flat, invalid_token_id, dtype=np.int64)
    flat_top_cat = np.full(n_flat, invalid_token_id, dtype=np.int64)
    flat_naics4 = np.full(n_flat, invalid_token_id, dtype=np.int64)
    has_top_cat = False
    has_naics4 = False
    tally = GroundingQualityTally()

    code_stack = np.stack(
        [
            sid_tokens[field].long().reshape(-1).cpu().numpy()
            for field in code_fields
        ],
        axis=1,
    ).astype(np.int64, copy=False)
    active_mask = ~(code_stack == invalid_token_id).all(axis=1)
    active_indices = np.nonzero(active_mask)[0]

    active_codes = code_stack[active_indices]
    preferred_payloads: (
        dict[int, tuple[tuple[int, ...], dict[str, Any]]] | None
    ) = None
    preferred_flat: np.ndarray | None = None
    if preferred_placekey_ids is not None:
        if preferred_placekey_ids.shape != shape:
            raise ValueError(
                "preferred_placekey_ids shape must match SID tokens; got "
                f"{tuple(preferred_placekey_ids.shape)} and {tuple(shape)}"
            )
        preferred_flat = (
            preferred_placekey_ids.long().reshape(-1).cpu().numpy()
        )
        preferred_payloads = {}
        for key, payloads in grounding_index.iter_key_payload_groups():
            for payload in payloads:
                placekey_id = payload.get("placekey_id")
                if placekey_id is not None:
                    preferred_payloads[int(placekey_id)] = (
                        tuple(int(value) for value in key),
                        dict(payload),
                    )
        for active_offset, flat_index in enumerate(active_indices):
            placekey_id = int(preferred_flat[flat_index])
            selected = preferred_payloads.get(placekey_id)
            if selected is None:
                raise ValueError(
                    f"preferred placekey_id {placekey_id} is not in the "
                    "grounding index"
                )
            selected_key, _payload = selected
            row = tuple(int(value) for value in active_codes[active_offset])
            shifted = tuple(
                value - token_space.REAL_TOKEN_OFFSET for value in row
            )
            if selected_key not in (row, shifted):
                raise ValueError(
                    f"preferred placekey_id {placekey_id} does not belong "
                    f"to SID row {row}"
                )
            active_codes[active_offset] = np.asarray(
                selected_key, dtype=np.int64
            )
    if np.any(active_codes < 0) or np.any(active_codes > 255):
        unique_codes, inverse, counts = np.unique(
            active_codes,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
    else:
        unique_packed, inverse, counts = np.unique(
            _pack_lookup_rows(active_codes),
            return_inverse=True,
            return_counts=True,
        )
        unique_codes = np.empty(
            (unique_packed.shape[0], active_codes.shape[1]), dtype=np.int64
        )
        for column in range(active_codes.shape[1]):
            unique_codes[:, column] = (unique_packed >> (8 * column)) & 255
    unique_placekey = np.full(
        unique_codes.shape[0], invalid_token_id, dtype=np.int64
    )
    unique_top_cat = np.full(
        unique_codes.shape[0], invalid_token_id, dtype=np.int64
    )
    unique_naics4 = np.full(
        unique_codes.shape[0], invalid_token_id, dtype=np.int64
    )
    unique_has_top_cat = np.zeros(unique_codes.shape[0], dtype=np.bool_)
    unique_has_naics4 = np.zeros(unique_codes.shape[0], dtype=np.bool_)

    candidate_counts, payload_indices, special, table_payloads = (
        _vectorized_flexible_lookup(unique_codes, grounding_index)
    )
    special_occurrences = int(counts[special].sum())
    if special_occurrences:
        tally.add_special(special_occurrences)
    count_values, count_inverse = np.unique(
        candidate_counts[~special], return_inverse=True
    )
    occurrence_by_count = np.bincount(
        count_inverse, weights=counts[~special]
    ).astype(np.int64, copy=False)
    for candidate_count, occurrences in zip(
        count_values, occurrence_by_count, strict=True
    ):
        tally.add_candidates(int(candidate_count), int(occurrences))

    def _encode_payload(
        payload: Mapping[str, Any],
    ) -> tuple[int, int, bool, int, bool]:
        pk_out = invalid_token_id
        cat_out, has_cat = invalid_token_id, False
        naics_out, has_naics = invalid_token_id, False
        pk = payload.get("placekey_id")
        if pk is not None:
            pk_out = int(pk)
        cat_val = payload.get("top_cat")
        if cat_val is None:
            cat_val = payload.get("top_category")
        if cat_val is not None:
            if isinstance(cat_val, str):
                if category_string_vocab is not None:
                    encoded = category_string_vocab.get(cat_val)
                    if encoded is not None:
                        cat_out, has_cat = encoded, True
            elif not isinstance(cat_val, bool):
                cat_out, has_cat = int(cat_val), True
        naics_val = payload.get("naics4")
        if naics_val is None:
            naics_val = payload.get("naics_code")
        if naics_val is not None:
            if isinstance(naics_val, str):
                if naics_string_vocab is not None:
                    encoded_naics = naics_string_vocab.get(naics_val)
                    if encoded_naics is not None:
                        naics_out, has_naics = encoded_naics, True
            elif not isinstance(naics_val, bool):
                naics_out, has_naics = int(naics_val), True
        return pk_out, cat_out, has_cat, naics_out, has_naics

    def _candidate_probabilities(
        group: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        assert candidate_weights is not None
        raw = np.array(
            [
                float(candidate_weights.get(int(payload["placekey_id"]), 0.0))
                if payload.get("placekey_id") is not None
                else 0.0
                for payload in group
            ],
            dtype=np.float64,
        )
        raw += candidate_weight_smoothing
        total = raw.sum()
        if not np.isfinite(total) or total <= 0.0:
            return np.full(raw.size, 1.0 / raw.size)
        return raw / total

    hit_unique = np.flatnonzero(payload_indices >= 0)
    use_policy = candidate_selection != "first" and preferred_payloads is None
    groups = grounding_index.packed_lookup_groups() if use_policy else None
    for u_idx in hit_unique:
        table_index = int(payload_indices[u_idx])
        payload = table_payloads[table_index]
        if use_policy and groups is not None:
            group = groups[table_index]
            if len(group) > 1:
                probabilities = _candidate_probabilities(group)
                payload = group[int(np.argmax(probabilities))]
        (
            unique_placekey[u_idx],
            unique_top_cat[u_idx],
            unique_has_top_cat[u_idx],
            unique_naics4[u_idx],
            unique_has_naics4[u_idx],
        ) = _encode_payload(payload)

    if active_indices.size:
        flat_placekey[active_indices] = unique_placekey[inverse]
        flat_top_cat[active_indices] = unique_top_cat[inverse]
        flat_naics4[active_indices] = unique_naics4[inverse]
        has_top_cat = bool(unique_has_top_cat.any())
        has_naics4 = bool(unique_has_naics4.any())

        if (
            candidate_selection == "a4_support"
            and groups is not None
            and hit_unique.size
        ):
            rng = np.random.default_rng(candidate_selection_seed)
            order = np.argsort(inverse, kind="stable")
            sorted_inverse = inverse[order]
            starts = np.searchsorted(sorted_inverse, hit_unique, side="left")
            ends = np.searchsorted(sorted_inverse, hit_unique, side="right")
            emitted: set[int] = set()
            for u_idx, start, end in zip(
                hit_unique, starts, ends, strict=True
            ):
                if end <= start:
                    continue
                group = groups[int(payload_indices[u_idx])]
                count = int(end - start)
                encoded = np.array(
                    [_encode_payload(payload) for payload in group],
                    dtype=np.int64,
                )
                if len(group) < 2:
                    emitted.add(int(encoded[0, 0]))
                    continue
                probabilities = _candidate_probabilities(group)
                unused = [
                    index
                    for index in range(len(group))
                    if int(encoded[index, 0]) not in emitted
                ]
                unused.sort(key=lambda index: -probabilities[index])
                picks: list[int] = []
                for index in unused[:count]:
                    picks.append(index)
                    emitted.add(int(encoded[index, 0]))
                if len(picks) < count:
                    picks.extend(
                        int(value)
                        for value in rng.choice(
                            len(group),
                            size=count - len(picks),
                            p=probabilities,
                        )
                    )
                pick_array = np.asarray(picks[:count], dtype=np.int64)
                targets = active_indices[order[start:end]]
                flat_placekey[targets] = encoded[pick_array, 0]
                flat_top_cat[targets] = encoded[pick_array, 1]
                flat_naics4[targets] = encoded[pick_array, 3]
                has_top_cat = has_top_cat or bool(encoded[pick_array, 2].any())
                has_naics4 = has_naics4 or bool(encoded[pick_array, 4].any())

        if (
            candidate_selection == "train_prior_sample"
            and groups is not None
            and hit_unique.size
        ):
            rng = np.random.default_rng(candidate_selection_seed)
            order = np.argsort(inverse, kind="stable")
            sorted_inverse = inverse[order]
            starts = np.searchsorted(sorted_inverse, hit_unique, side="left")
            ends = np.searchsorted(sorted_inverse, hit_unique, side="right")
            for u_idx, start, end in zip(
                hit_unique, starts, ends, strict=True
            ):
                if end <= start:
                    continue
                group = groups[int(payload_indices[u_idx])]
                if len(group) < 2:
                    continue
                probabilities = _candidate_probabilities(group)
                encoded = np.array(
                    [_encode_payload(payload) for payload in group],
                    dtype=np.int64,
                )
                sampled_picks = rng.choice(
                    len(group), size=int(end - start), p=probabilities
                )
                targets = active_indices[order[start:end]]
                flat_placekey[targets] = encoded[sampled_picks, 0]
                flat_top_cat[targets] = encoded[sampled_picks, 1]
                flat_naics4[targets] = encoded[sampled_picks, 3]
                has_top_cat = has_top_cat or bool(
                    encoded[sampled_picks, 2].any()
                )
                has_naics4 = has_naics4 or bool(
                    encoded[sampled_picks, 4].any()
                )

    if preferred_payloads is not None and preferred_flat is not None:
        for flat_index in active_indices:
            placekey_id = int(preferred_flat[flat_index])
            _selected_key, payload = preferred_payloads[placekey_id]
            flat_placekey[flat_index] = placekey_id
            flat_top_cat[flat_index] = invalid_token_id
            flat_naics4[flat_index] = invalid_token_id
            cat_val = payload.get("top_cat", payload.get("top_category"))
            if isinstance(cat_val, str) and category_string_vocab is not None:
                encoded_cat = category_string_vocab.get(cat_val)
                if encoded_cat is not None:
                    flat_top_cat[flat_index] = encoded_cat
                    has_top_cat = True
            elif cat_val is not None and not isinstance(cat_val, bool):
                flat_top_cat[flat_index] = int(cat_val)
                has_top_cat = True
            naics_val = payload.get("naics4", payload.get("naics_code"))
            if isinstance(naics_val, str) and naics_string_vocab is not None:
                encoded_naics = naics_string_vocab.get(naics_val)
                if encoded_naics is not None:
                    flat_naics4[flat_index] = encoded_naics
                    has_naics4 = True
            elif naics_val is not None and not isinstance(naics_val, bool):
                flat_naics4[flat_index] = int(naics_val)
                has_naics4 = True

    placekey_id = torch.from_numpy(flat_placekey).view(*shape)
    top_cat = torch.from_numpy(flat_top_cat).view(*shape)
    naics4 = torch.from_numpy(flat_naics4).view(*shape)

    tokens = {"placekey_id": placekey_id}
    if has_top_cat:
        tokens["top_cat"] = top_cat
    if has_naics4:
        tokens["naics4"] = naics4
    stats = {
        "n_positions": float(tally.n_positions),
        "valid_rate": tally.valid_rate,
        "ambiguous_rate": tally.ambiguous_rate,
        "mean_candidates_per_valid": tally.mean_candidates_per_valid,
    }
    return tokens, stats


def compute_decoded_sid_poi_metrics(
    tokens_real: dict[str, torch.Tensor],
    tokens_gen: dict[str, torch.Tensor],
    grounding_index: Any,
    *,
    placekey_latlon: dict[int, tuple[float, float]] | None = None,
    geohash_lookup: dict[int, str] | None = None,
    placekey_vocab_size: int | None = None,
    category_vocab_size: int | None = None,
    invalid_token_id: int = PAD_ID,
    real_poi: dict[str, torch.Tensor] | None = None,
    gen_poi: dict[str, torch.Tensor] | None = None,
    real_stats: dict[str, float] | None = None,
    gen_stats: dict[str, float] | None = None,
    cluster_ids: Sequence[str] | None = None,
    motif_day_keys_real: Any | None = None,
    motif_day_keys_gen: Any | None = None,
    motif_frequency_threshold: float = 0.005,
    compute_device: torch.device | str | None = None,
    brief_metrics: bool = False,
) -> dict[str, Any] | None:
    if not all(field in tokens_real for field in grounding_index.key_columns):
        return None
    if not all(field in tokens_gen for field in grounding_index.key_columns):
        return None

    if real_poi is None or gen_poi is None:
        category_string_vocab = grounding_index.build_category_string_vocab()
        naics_string_vocab = grounding_index.build_naics_string_vocab()
        real_poi, real_stats = ground_sid_tokens_to_poi_tokens(
            tokens_real,
            grounding_index,
            invalid_token_id=invalid_token_id,
            category_string_vocab=category_string_vocab,
            naics_string_vocab=naics_string_vocab,
        )
        gen_poi, gen_stats = ground_sid_tokens_to_poi_tokens(
            tokens_gen,
            grounding_index,
            invalid_token_id=invalid_token_id,
            category_string_vocab=category_string_vocab,
            naics_string_vocab=naics_string_vocab,
        )
    elif real_stats is None or gen_stats is None:
        raise ValueError(
            "real_stats and gen_stats are required when reusing pre-grounded "
            "POI tokens"
        )

    max_placekey = 0
    for tokens in (real_poi["placekey_id"], gen_poi["placekey_id"]):
        if tokens.numel():
            max_placekey = max(max_placekey, int(tokens.max().item()))
    field_vocab = {
        "placekey_id": placekey_vocab_size or max(max_placekey + 1, 1)
    }
    if "top_cat" in real_poi and "top_cat" in gen_poi:
        max_cat = 0
        for tokens in (real_poi["top_cat"], gen_poi["top_cat"]):
            if tokens.numel():
                max_cat = max(max_cat, int(tokens.max().item()))
        field_vocab["top_cat"] = category_vocab_size or max(max_cat + 1, 1)
    if "naics4" in real_poi and "naics4" in gen_poi:
        max_naics = 0
        for tokens in (real_poi["naics4"], gen_poi["naics4"]):
            if tokens.numel():
                max_naics = max(max_naics, int(tokens.max().item()))
        field_vocab["naics4"] = max(max_naics + 1, 1)

    metric_real_poi = real_poi
    metric_gen_poi = gen_poi
    if compute_device is not None:
        resolved_device = torch.device(compute_device)
        metric_real_poi = {
            field: value.to(resolved_device)
            for field, value in real_poi.items()
        }
        metric_gen_poi = {
            field: value.to(resolved_device)
            for field, value in gen_poi.items()
        }

    if brief_metrics:
        real_placekeys = metric_real_poi["placekey_id"]
        generated_placekeys = metric_gen_poi["placekey_id"]
        poi_distribution = compute_poi_distribution_metrics(
            metric_real_poi,
            metric_gen_poi,
            field_vocab,
            pad_id=invalid_token_id,
        )
        motif = mobility_motif_metrics(
            real_placekeys,
            generated_placekeys,
            frequency_threshold=motif_frequency_threshold,
            pad_id=invalid_token_id,
        )
        real_return_gaps = identity_histogram_contributions(
            real_poi["placekey_id"], pad_id=invalid_token_id
        )["return_gap_distribution_jsd"]
        generated_return_gaps = identity_histogram_contributions(
            gen_poi["placekey_id"], pad_id=invalid_token_id
        )["return_gap_distribution_jsd"]
        recurrence_jsd = float(
            rowwise_jsd_counts(
                real_return_gaps.sum(axis=0, keepdims=True),
                generated_return_gaps.sum(axis=0, keepdims=True),
            )[0]
        )
        poi_values = (
            poi_distribution.to_dict() if poi_distribution is not None else {}
        )
        return {
            "trajectory_metrics": {
                "geohash_spatial_jsd": (
                    spatial_jsd(
                        real_placekeys,
                        generated_placekeys,
                        geohash_lookup,
                        pad_id=invalid_token_id,
                    )
                    if geohash_lookup is not None
                    else None
                ),
                "travel_dist_jsd": (
                    travel_distance_jsd(
                        real_placekeys,
                        generated_placekeys,
                        placekey_latlon,
                        pad_id=invalid_token_id,
                    )
                    if placekey_latlon is not None
                    else None
                ),
                "mobility_motif_jsd": motif["mobility_motif_jsd"],
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
                        pad_id=invalid_token_id,
                    )
                    if placekey_latlon is not None
                    else None
                )
            },
            "identity_dynamics": {
                "poi": {
                    "distribution_distances": {
                        "return_gap_distribution_jsd": {
                            "point_estimate": recurrence_jsd,
                        }
                    }
                }
            },
        }

    traj = compute_trajectory_metrics(
        metric_real_poi,
        metric_gen_poi,
        field_vocab,
        pad_id=invalid_token_id,
        placekey_latlon=placekey_latlon,
        geohash_lookup=geohash_lookup,
        motif_day_keys_real=motif_day_keys_real,
        motif_day_keys_gen=motif_day_keys_gen,
        motif_frequency_threshold=motif_frequency_threshold,
    )
    poi_dist = compute_poi_distribution_metrics(
        metric_real_poi,
        metric_gen_poi,
        field_vocab,
        pad_id=invalid_token_id,
    )

    category_metrics: dict[str, Any] | None = None
    if "top_cat" in real_poi and "top_cat" in gen_poi:
        cat_vs = field_vocab.get("top_cat", 0)
        if cat_vs > 0:
            cm: dict[str, Any] = {}
            cm["top_cat_transition_distance"] = transition_distance(
                real_poi["top_cat"],
                gen_poi["top_cat"],
                cat_vs,
                pad_id=invalid_token_id,
            )
            real_toks = metric_real_poi["top_cat"]
            gen_toks = metric_gen_poi["top_cat"]
            real_counts = {
                order: ngram_counts_hashed(
                    real_toks, cat_vs, n=order, pad_id=invalid_token_id
                )
                for order in (2, 3)
            }
            gen_counts = {
                order: ngram_counts_hashed(
                    gen_toks, cat_vs, n=order, pad_id=invalid_token_id
                )
                for order in (2, 3)
            }
            for order in (2, 3):
                cm[f"top_cat_ngram_jsd_n{order}"] = activity_ngram_jsd(
                    real_toks,
                    gen_toks,
                    cat_vs,
                    n=order,
                    pad_id=invalid_token_id,
                    real_counts=real_counts[order],
                    gen_counts=gen_counts[order],
                )
                cm[f"top_cat_distinct_ratio_n{order}_real"] = (
                    distinct_ngram_ratio(
                        real_toks,
                        cat_vs,
                        n=order,
                        pad_id=invalid_token_id,
                        counts=real_counts[order],
                    )
                )
                cm[f"top_cat_distinct_ratio_n{order}_gen"] = (
                    distinct_ngram_ratio(
                        gen_toks,
                        cat_vs,
                        n=order,
                        pad_id=invalid_token_id,
                        counts=gen_counts[order],
                    )
                )
            cm["top_cat_self_overlap_n3_gen"] = self_ngram_overlap(
                gen_toks,
                cat_vs,
                n=3,
                pad_id=invalid_token_id,
            )
            cm["top_cat_self_overlap_n3_real"] = self_ngram_overlap(
                real_toks,
                cat_vs,
                n=3,
                pad_id=invalid_token_id,
            )
            category_uncertainty: dict[str, Any] = {
                "top_cat_unigram_jsd": cluster_histogram_jsd_bootstrap(
                    _token_histogram_contributions(
                        real_toks,
                        vocab_size=cat_vs,
                        pad_id=invalid_token_id,
                    ),
                    _token_histogram_contributions(
                        gen_toks,
                        vocab_size=cat_vs,
                        pad_id=invalid_token_id,
                    ),
                    cluster_ids=cluster_ids,
                    compute_device=compute_device,
                )
            }
            for order in (2, 3):
                ngram_uncertainty = cluster_sparse_ngram_jsd_bootstrap(
                    real_toks,
                    gen_toks,
                    vocab_size=cat_vs,
                    n=order,
                    pad_id=invalid_token_id,
                    cluster_ids=cluster_ids,
                    compute_device=compute_device,
                )
                if ngram_uncertainty is not None:
                    category_uncertainty[f"top_cat_ngram_jsd_n{order}"] = (
                        ngram_uncertainty
                    )
            cm["uncertainty"] = category_uncertainty
            category_metrics = cm

    spatial_metrics: dict[str, Any] | None = None
    if (
        placekey_latlon is not None
        and "placekey_id" in real_poi
        and "placekey_id" in gen_poi
    ):
        sm: dict[str, Any] = {}
        sm["radius_of_gyration_jsd"] = radius_of_gyration_jsd(
            metric_real_poi["placekey_id"],
            metric_gen_poi["placekey_id"],
            placekey_latlon,
            pad_id=invalid_token_id,
        )
        sm["unique_location_count_jsd"] = unique_location_count_jsd(
            metric_real_poi["placekey_id"],
            metric_gen_poi["placekey_id"],
            pad_id=invalid_token_id,
        )
        sm["uncertainty"] = spatial_guardrail_uncertainty(
            real_poi["placekey_id"],
            gen_poi["placekey_id"],
            placekey_latlon,
            geohash_lookup=geohash_lookup,
            cluster_ids=cluster_ids,
            compute_device=compute_device,
            pad_id=invalid_token_id,
        )
        spatial_metrics = sm

    identity_dynamics = None
    if "placekey_id" in real_poi and "placekey_id" in gen_poi:
        identity_dynamics = identity_dynamics_comparison(
            real_poi["placekey_id"],
            gen_poi["placekey_id"],
            pad_id=invalid_token_id,
            kind="poi",
            cluster_ids=cluster_ids,
            compute_device=compute_device,
        )
        identity_dynamics["category"] = (
            category_identity_dynamics_comparison(
                real_poi["top_cat"],
                gen_poi["top_cat"],
                pad_id=invalid_token_id,
                cluster_ids=cluster_ids,
                compute_device=compute_device,
            )
            if "top_cat" in real_poi and "top_cat" in gen_poi
            else None
        )

    poi_distribution = poi_dist.to_dict() if poi_dist is not None else None
    if poi_distribution is not None:
        poi_distribution["uncertainty"] = (
            cluster_sparse_token_distribution_bootstrap(
                real_poi["placekey_id"],
                gen_poi["placekey_id"],
                vocab_size=field_vocab["placekey_id"],
                pad_id=invalid_token_id,
                cluster_ids=cluster_ids,
                compute_device=compute_device,
            )
        )

    return {
        "trajectory_metrics": traj.to_dict(),
        "poi_distribution": poi_distribution,
        "category_metrics": category_metrics,
        "spatial_metrics": spatial_metrics,
        "identity_dynamics": identity_dynamics,
        "grounding": {
            "invalid_token_id": invalid_token_id,
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
