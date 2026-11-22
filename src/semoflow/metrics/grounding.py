from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from semoflow.config import PAD_ID
from semoflow.metrics.distribution import (
    jensen_shannon_divergence,
    token_marginal_jsd,
)
from semoflow.metrics.geo import (
    radius_of_gyration_jsd,
    spatial_jsd,
    travel_distance_jsd,
)
from semoflow.metrics.motifs import mobility_motif_metrics
from semoflow.tokenization.sids import pack_code_rows

_CANDIDATE_SELECTION_POLICIES = frozenset(
    {"first", "train_prior_argmax", "train_prior_sample"}
)
_CANDIDATE_SELECTION_SEED = 20260830
_CANDIDATE_WEIGHT_SMOOTHING = 1.0


def _vectorized_lookup(
    code_rows: np.ndarray, grounding_index: Any
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Each SID row's index into the packed payload table.

    Rows with a negative code or no catalog SID get ``-1``.
    """
    table_keys, table_payloads = grounding_index.packed_lookup_table()
    payload_indices = np.full(code_rows.shape[0], -1, dtype=np.int64)
    row_indices = np.flatnonzero(~np.any(code_rows < 0, axis=1))
    if row_indices.size:
        packed = pack_code_rows(code_rows[row_indices])
        positions = np.searchsorted(table_keys, packed, side="left")
        safe_positions = np.minimum(positions, table_keys.size - 1)
        found = (positions < table_keys.size) & (
            table_keys[safe_positions] == packed
        )
        payload_indices[row_indices[found]] = safe_positions[found]
    return payload_indices, table_payloads


def _placekey(payload: Mapping[str, Any]) -> int:
    placekey = payload.get("placekey_id")
    return PAD_ID if placekey is None else int(placekey)


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
    return raw / raw.sum()


def _preferred_payloads(
    preferred_placekey_ids: np.ndarray,
    active_codes: np.ndarray,
    grounding_index: Any,
) -> list[Mapping[str, Any]]:
    """Each active position's preferred POI payload.

    The preferred POI must be a candidate of the position's SID.
    """
    by_placekey: dict[int, tuple[tuple[int, ...], dict[str, Any]]] = {}
    for key, payloads in grounding_index.groups.items():
        for payload in payloads:
            placekey_id = payload.get("placekey_id")
            if placekey_id is not None:
                by_placekey[int(placekey_id)] = (
                    tuple(int(value) for value in key),
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


def ground_sid_codes(
    sid_codes: torch.Tensor,
    valid_mask: torch.Tensor,
    grounding_index: Any,
    *,
    preferred_placekey_ids: torch.Tensor | None = None,
    candidate_selection: str = "first",
    candidate_weights: Mapping[int, float] | None = None,
) -> torch.Tensor:
    """The placekey ID each valid ``[rows, events, depth]`` SID grounds to.

    Invalid or ungrounded positions get ``PAD_ID``.

    Each SID resolves to the grounding index's first candidate, or to a
    train-prior choice under ``candidate_selection``;
    ``train_prior_sample`` chooses per occurrence. ``preferred_placekey_ids``
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
    shape = valid_mask.shape
    codes = sid_codes.long().reshape(-1, sid_codes.size(-1)).cpu().numpy()
    active = np.flatnonzero(valid_mask.bool().reshape(-1).cpu().numpy())
    active_codes = codes[active]
    encoded = np.full(len(codes), PAD_ID, dtype=np.int64)
    if preferred_placekey_ids is not None:
        preferred = _preferred_payloads(
            preferred_placekey_ids.long().reshape(-1).cpu().numpy()[active],
            active_codes,
            grounding_index,
        )
        encoded[active] = [_placekey(payload) for payload in preferred]
        return torch.from_numpy(encoded).view(*shape)

    unique_codes, inverse, _counts = _unique_code_rows(active_codes)
    payload_indices, first_payloads = _vectorized_lookup(
        unique_codes, grounding_index
    )
    hit = np.flatnonzero(payload_indices >= 0)
    groups = (
        grounding_index.packed_lookup_groups()
        if candidate_selection != "first"
        else None
    )
    unique_encoded = np.full(len(unique_codes), PAD_ID, dtype=np.int64)
    for row in hit:
        index = int(payload_indices[row])
        payload = first_payloads[index]
        if groups is not None and len(groups[index]) > 1:
            assert candidate_weights is not None
            group = groups[index]
            weights = _candidate_probabilities(group, candidate_weights)
            payload = group[int(np.argmax(weights))]
        unique_encoded[row] = _placekey(payload)
    encoded[active] = unique_encoded[inverse]

    if groups is not None and candidate_selection == "train_prior_sample":
        assert candidate_weights is not None
        rng = np.random.default_rng(_CANDIDATE_SELECTION_SEED)
        order = np.argsort(inverse, kind="stable")
        starts = np.searchsorted(inverse[order], hit, side="left")
        ends = np.searchsorted(inverse[order], hit, side="right")
        for row, start, end in zip(hit, starts, ends, strict=True):
            group = groups[int(payload_indices[row])]
            if len(group) < 2:
                continue
            group_encoded = np.array(
                [_placekey(payload) for payload in group], dtype=np.int64
            )
            picks = rng.choice(
                len(group),
                size=int(end - start),
                p=_candidate_probabilities(group, candidate_weights),
            )
            encoded[active[order[start:end]]] = group_encoded[picks]
    return torch.from_numpy(encoded).view(*shape)


def _return_histograms(tokens: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Pooled return-gap (in visits) and per-POI visit-count histograms.

    Values at or beyond the trajectory width share the last bin.
    """
    width = tokens.size(1)
    gaps = np.zeros(width + 1, dtype=np.int64)
    multiplicity = np.zeros(width + 1, dtype=np.int64)
    for raw_row in tokens.cpu().tolist():
        row = [value for value in raw_row if value != PAD_ID]
        for count in Counter(row).values():
            multiplicity[min(count, width)] += 1
        previous: dict[int, int] = {}
        for position, value in enumerate(row):
            if value in previous:
                gaps[min(position - previous[value], width)] += 1
            previous[value] = position
    return gaps, multiplicity


def grounded_poi_metrics(
    real: torch.Tensor,
    generated: torch.Tensor,
    *,
    placekey_vocab_size: int,
    coordinates: dict[int, tuple[float, float]],
    geohashes: dict[int, str],
    day_keys_real: Any,
    day_keys_generated: Any,
    compute_device: torch.device | str | None = None,
) -> dict[str, float | None]:
    """The POI-level JSDs between grounded real and generated placekeys."""
    real_gaps, real_multiplicity = _return_histograms(real)
    generated_gaps, generated_multiplicity = _return_histograms(generated)
    if compute_device is not None:
        real = real.to(compute_device)
        generated = generated.to(compute_device)
    real_unique = torch.unique(real[real != PAD_ID])
    generated_unique = torch.unique(generated[generated != PAD_ID])
    return {
        "poi_frequency_jsd": token_marginal_jsd(
            real, generated, placekey_vocab_size, pad_id=PAD_ID
        ),
        "poi_coverage": float(
            torch.isin(real_unique, generated_unique).float().mean().item()
        ),
        "geohash_jsd": spatial_jsd(real, generated, geohashes, pad_id=PAD_ID),
        "travel_jsd": travel_distance_jsd(
            real, generated, coordinates, pad_id=PAD_ID
        ),
        "radius_jsd": radius_of_gyration_jsd(
            real, generated, coordinates, pad_id=PAD_ID
        ),
        "mobility_motif_jsd": mobility_motif_metrics(
            real,
            generated,
            day_keys_real=day_keys_real,
            day_keys_generated=day_keys_generated,
            pad_id=PAD_ID,
        )["mobility_motif_jsd"],
        "mobility_recurrence_pattern_jsd": jensen_shannon_divergence(
            real_gaps, generated_gaps
        ),
        "multiplicity_jsd": jensen_shannon_divergence(
            real_multiplicity, generated_multiplicity
        ),
    }


__all__ = ["ground_sid_codes", "grounded_poi_metrics"]
