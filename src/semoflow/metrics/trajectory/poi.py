from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import torch

from semoflow.config.constants import PAD_ID
from semoflow.metrics.trajectory.distribution import token_marginal_jsd


@dataclass
class POIDistributionMetrics:
    poi_unique_count_real: int
    poi_unique_count_gen: int
    poi_unique_ratio: float
    poi_coverage: float
    poi_frequency_jsd: float | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compute_poi_distribution_metrics(
    tokens_real: dict[str, torch.Tensor],
    tokens_gen: dict[str, torch.Tensor],
    field_vocab: dict[str, int],
    *,
    pad_id: int = PAD_ID,
) -> POIDistributionMetrics | None:
    if "placekey_id" not in tokens_real or "placekey_id" not in tokens_gen:
        return None
    real = tokens_real["placekey_id"].long().flatten()
    gen = tokens_gen["placekey_id"].long().flatten()
    real_valid = real[real != pad_id]
    gen_valid = gen[gen != pad_id]
    if real_valid.numel() == 0:
        return None

    real_unique = torch.unique(real_valid)
    gen_unique = torch.unique(gen_valid)
    vocab = field_vocab.get("placekey_id")
    return POIDistributionMetrics(
        poi_unique_count_real=real_unique.numel(),
        poi_unique_count_gen=gen_unique.numel(),
        poi_unique_ratio=gen_unique.numel() / real_unique.numel(),
        poi_coverage=float(
            torch.isin(real_unique, gen_unique).float().mean().item()
        ),
        poi_frequency_jsd=(
            None
            if vocab is None
            else token_marginal_jsd(
                tokens_real["placekey_id"],
                tokens_gen["placekey_id"],
                vocab,
                pad_id=pad_id,
            )
        ),
    )
