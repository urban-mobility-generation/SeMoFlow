from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import torch

from semoflow.config import PAD_ID
from semoflow.evaluation.distribution import jsd

_FREQUENT_MOTIF_SHARE = 0.005
type MotifSignature = tuple[int, tuple[tuple[int, int], ...]]


def _day_motif(locations: list[int]) -> MotifSignature:
    node_ids: dict[int, int] = {}
    edges: set[tuple[int, int]] = set()
    previous: int | None = None
    previous_location: int | None = None
    for location in locations:
        if location == previous_location:
            continue
        current = node_ids.setdefault(location, len(node_ids))
        if previous is not None:
            edges.add((previous, current))
        previous, previous_location = current, location
    return len(node_ids), tuple(sorted(edges))


def daily_mobility_motifs(
    locations: torch.Tensor, day_keys: torch.Tensor
) -> list[MotifSignature]:
    motifs: list[MotifSignature] = []
    for row, days in zip(locations.tolist(), day_keys.tolist(), strict=True):
        by_day: dict[int, list[int]] = defaultdict(list)
        for location, day in zip(row, days, strict=True):
            if location != PAD_ID:
                by_day[day].append(location)
        motifs.extend(_day_motif(visits) for visits in by_day.values())
    return motifs


def _signature_key(signature: MotifSignature) -> str:
    node_count, edges = signature
    edge_text = ",".join(f"{source}>{target}" for source, target in edges)
    return f"n{node_count}|e{edge_text}"


def mobility_motif_jsd(
    real: torch.Tensor,
    generated: torch.Tensor,
    *,
    day_keys_real: torch.Tensor,
    day_keys_generated: torch.Tensor,
) -> float:
    real_counts = Counter(daily_mobility_motifs(real, day_keys_real))
    generated_counts = Counter(
        daily_mobility_motifs(generated, day_keys_generated)
    )
    n_real = sum(real_counts.values())
    frequent = sorted(
        (
            motif
            for motif, count in real_counts.items()
            if count / n_real > _FREQUENT_MOTIF_SHARE
        ),
        key=_signature_key,
    )
    kept = set(frequent)

    def histogram(counts: Counter[MotifSignature]) -> np.ndarray:
        other = sum(
            count for motif, count in counts.items() if motif not in kept
        )
        return np.asarray(
            [counts[motif] for motif in frequent] + [other], dtype=np.float64
        )

    return jsd(histogram(real_counts), histogram(generated_counts))


__all__ = [
    "mobility_motif_jsd",
]
