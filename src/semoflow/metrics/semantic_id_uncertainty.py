from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> list[float] | None:
    observed = np.asarray(values, dtype=np.float64)
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        return None
    if observed.size == 1 or resamples <= 0:
        value = float(observed.mean())
        return [value, value]
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    rng = np.random.RandomState(seed)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.randint(0, observed.size, size=observed.size)
        means[index] = observed[sampled].mean()
    alpha = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    ]


def bootstrap_difference_ci(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> list[float] | None:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    left_values = left_values[np.isfinite(left_values)]
    right_values = right_values[np.isfinite(right_values)]
    if left_values.size == 0 or right_values.size == 0:
        return None
    if resamples <= 0:
        return [
            float(left_values.mean() - right_values.mean()),
            float(left_values.mean() - right_values.mean()),
        ]
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    rng = np.random.RandomState(seed)
    differences = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        left_sample = rng.randint(0, left_values.size, size=left_values.size)
        right_sample = rng.randint(
            0, right_values.size, size=right_values.size
        )
        differences[index] = (
            left_values[left_sample].mean() - right_values[right_sample].mean()
        )
    alpha = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(differences, alpha)),
        float(np.quantile(differences, 1.0 - alpha)),
    ]


__all__ = ["bootstrap_difference_ci", "bootstrap_mean_ci"]
