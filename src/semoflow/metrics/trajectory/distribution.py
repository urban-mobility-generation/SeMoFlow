from __future__ import annotations

import math

import torch

from semoflow.config.constants import PAD_ID


def jsd(p: torch.Tensor, q: torch.Tensor) -> float:
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
    return float(torch.clamp(value, 0.0, math.log(2.0)).item())


def token_marginal_jsd(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    vocab_size: int,
    *,
    pad_id: int = PAD_ID,
) -> float:
    real = tokens_real[tokens_real != pad_id]
    gen = tokens_gen[tokens_gen != pad_id]
    p = torch.bincount(real, minlength=vocab_size).float()[:vocab_size]
    q = torch.bincount(gen, minlength=vocab_size).float()[:vocab_size]
    return jsd(p, q)
