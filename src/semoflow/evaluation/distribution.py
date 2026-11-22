"""Jensen-Shannon divergences (natural log) over histograms."""

from __future__ import annotations

import math

import numpy as np
import torch

from semoflow.config import PAD_ID

JSD_MAX_NATS = math.log(2.0)


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """JSD of two unnormalized histograms in float64; NaN if one is empty."""
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


def jsd(p: torch.Tensor, q: torch.Tensor) -> float:
    """JSD of two unnormalized histogram tensors at their own precision;
    NaN if one is empty. Travel, radius, spatial and POI-frequency scores
    use it."""
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")
    p = p / p_sum
    q = q / q_sum
    m = (p + q) / 2
    positive_p = p > 0
    positive_q = q > 0
    kl_pm = (p[positive_p] * (p[positive_p] / m[positive_p]).log()).sum()
    kl_qm = (q[positive_q] * (q[positive_q] / m[positive_q]).log()).sum()
    value = 0.5 * kl_pm + 0.5 * kl_qm
    return float(torch.clamp(value, 0.0, JSD_MAX_NATS).item())


def token_marginal_jsd(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    vocab_size: int,
    *,
    pad_id: int = PAD_ID,
) -> float:
    """JSD of the token frequencies of two sequences, ignoring padding."""
    real = tokens_real[tokens_real != pad_id]
    gen = tokens_gen[tokens_gen != pad_id]
    p = torch.bincount(real, minlength=vocab_size).float()[:vocab_size]
    q = torch.bincount(gen, minlength=vocab_size).float()[:vocab_size]
    return jsd(p, q)


__all__ = [
    "JSD_MAX_NATS",
    "jensen_shannon_divergence",
    "jsd",
    "token_marginal_jsd",
]
