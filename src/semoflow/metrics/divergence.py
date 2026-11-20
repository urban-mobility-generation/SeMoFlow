from __future__ import annotations

import math

import numpy as np

JSD_MAX_NATS = math.log(2.0)


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)

    left_total = float(left_array.sum())
    right_total = float(right_array.sum())
    if left_total <= 0.0 or right_total <= 0.0:
        return float("nan")

    p = left_array / left_total
    q = right_array / right_total
    midpoint = 0.5 * (p + q)
    positive_p = p > 0.0
    positive_q = q > 0.0
    kl_p = np.sum(p[positive_p] * np.log(p[positive_p] / midpoint[positive_p]))
    kl_q = np.sum(q[positive_q] * np.log(q[positive_q] / midpoint[positive_q]))
    value = float(0.5 * (kl_p + kl_q))
    return float(np.clip(value, 0.0, JSD_MAX_NATS))


__all__ = ["JSD_MAX_NATS", "jensen_shannon_divergence"]
