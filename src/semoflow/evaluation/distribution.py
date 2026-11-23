from __future__ import annotations

import math

import numpy as np
import torch

from semoflow.config import PAD_ID

JSD_MAX_NATS = math.log(2.0)


def _counts(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def jsd(
    left: np.ndarray | torch.Tensor, right: np.ndarray | torch.Tensor
) -> float:
    left_array, right_array = _counts(left), _counts(right)
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
    return float(np.clip(0.5 * (kl_p + kl_q), 0.0, JSD_MAX_NATS))


def token_marginal_jsd(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    vocab_size: int,
) -> float:
    real = tokens_real[tokens_real != PAD_ID]
    gen = tokens_gen[tokens_gen != PAD_ID]
    p = torch.bincount(real, minlength=vocab_size).float()[:vocab_size]
    q = torch.bincount(gen, minlength=vocab_size).float()[:vocab_size]
    return jsd(p, q)


__all__ = [
    "jsd",
    "token_marginal_jsd",
]
