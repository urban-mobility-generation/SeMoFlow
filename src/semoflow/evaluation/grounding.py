from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from semoflow.config import PAD_ID
from semoflow.evaluation.distribution import (
    jensen_shannon_divergence,
    token_marginal_jsd,
)
from semoflow.evaluation.geo import (
    radius_of_gyration_jsd,
    spatial_jsd,
    travel_distance_jsd,
)
from semoflow.evaluation.motifs import mobility_motif_jsd
from semoflow.tokenization.sids import pack_code_rows

_CANDIDATE_SELECTION_SEED = 20260830
_CANDIDATE_WEIGHT_SMOOTHING = 1.0


def _vectorized_lookup(
    code_rows: np.ndarray, grounding_index: Any
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    table_keys, table_payloads = grounding_index.packed_lookup_table()
    packed = pack_code_rows(code_rows)
    positions = np.searchsorted(table_keys, packed, side="left")
    safe_positions = np.minimum(positions, table_keys.size - 1)
    found = (positions < table_keys.size) & (
        table_keys[safe_positions] == packed
    )
    return np.where(found, safe_positions, -1), table_payloads


def _candidate_probabilities(
    group: Sequence[Mapping[str, Any]], weights: Mapping[int, float]
) -> np.ndarray:
    raw = np.array(
        [weights.get(int(payload["placekey_id"]), 0.0) for payload in group],
        dtype=np.float64,
    )
    raw += _CANDIDATE_WEIGHT_SMOOTHING
    return raw / raw.sum()


def _unique_code_rows(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    packed, inverse = np.unique(pack_code_rows(codes), return_inverse=True)
    unique = np.empty((packed.shape[0], codes.shape[1]), dtype=np.int64)
    for column in range(codes.shape[1]):
        unique[:, column] = (packed >> (8 * column)) & 255
    return unique, inverse


def ground_sid_codes(
    sid_codes: torch.Tensor,
    valid_mask: torch.Tensor,
    grounding_index: Any,
    *,
    preferred_placekey_ids: torch.Tensor | None = None,
    candidate_selection: str = "first",
    candidate_weights: Mapping[int, float] | None = None,
) -> torch.Tensor:
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
        encoded[active] = (
            preferred_placekey_ids.reshape(-1).cpu().numpy()[active]
        )
        return torch.from_numpy(encoded).view(*shape)

    unique_codes, inverse = _unique_code_rows(active_codes)
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
        unique_encoded[row] = int(payload["placekey_id"])
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
                [payload["placekey_id"] for payload in group], dtype=np.int64
            )
            picks = rng.choice(
                len(group),
                size=int(end - start),
                p=_candidate_probabilities(group, candidate_weights),
            )
            encoded[active[order[start:end]]] = group_encoded[picks]
    return torch.from_numpy(encoded).view(*shape)


def _return_histograms(tokens: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
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
        "mobility_motif_jsd": mobility_motif_jsd(
            real,
            generated,
            day_keys_real=day_keys_real,
            day_keys_generated=day_keys_generated,
        ),
        "mobility_recurrence_pattern_jsd": jensen_shannon_divergence(
            real_gaps, generated_gaps
        ),
        "multiplicity_jsd": jensen_shannon_divergence(
            real_multiplicity, generated_multiplicity
        ),
    }


__all__ = ["ground_sid_codes", "grounded_poi_metrics"]
