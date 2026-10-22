from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from semoflow.config.constants import PAD_ID
from semoflow.metrics.trajectory.distribution import (
    token_marginal_jsd,
)


@dataclass
class POIDistributionMetrics:
    poi_unique_count_real: int
    poi_unique_count_gen: int
    poi_unique_ratio: float
    poi_coverage: float
    poi_frequency_jsd: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "poi_unique_count_real": self.poi_unique_count_real,
            "poi_unique_count_gen": self.poi_unique_count_gen,
            "poi_unique_ratio": self.poi_unique_ratio,
            "poi_coverage": self.poi_coverage,
            "poi_frequency_jsd": self.poi_frequency_jsd,
        }


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
    n_real = int(real_unique.numel())
    n_gen = int(gen_unique.numel())
    coverage = (
        float(torch.isin(real_unique, gen_unique).float().mean().item())
        if n_real > 0
        else 0.0
    )

    freq_jsd: float | None = None
    vocab = field_vocab.get("placekey_id")
    if vocab is not None:
        freq_jsd = token_marginal_jsd(
            tokens_real["placekey_id"],
            tokens_gen["placekey_id"],
            vocab,
            pad_id=pad_id,
        )

    return POIDistributionMetrics(
        poi_unique_count_real=n_real,
        poi_unique_count_gen=n_gen,
        poi_unique_ratio=n_gen / n_real if n_real > 0 else 0.0,
        poi_coverage=coverage,
        poi_frequency_jsd=freq_jsd,
    )
