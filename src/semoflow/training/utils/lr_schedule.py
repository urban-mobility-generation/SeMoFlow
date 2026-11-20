from __future__ import annotations

import math

from torch import optim

LR_SCHEDULES = ("constant", "cosine")


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    *,
    schedule: str,
    warmup_steps: int,
    total_updates: int,
    base_lr: float,
    min_lr: float,
) -> optim.lr_scheduler.LRScheduler | None:
    if schedule == "constant" and warmup_steps == 0:
        return None
    if schedule == "cosine" and warmup_steps >= total_updates:
        raise ValueError(
            "cosine warmup_steps must be smaller than total optimizer updates"
        )
    minimum = min_lr / base_lr if base_lr else 0.0

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return min(float(step + 1) / warmup_steps, 1.0)
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(total_updates - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return minimum + (1.0 - minimum) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


__all__ = ["LR_SCHEDULES", "build_lr_scheduler"]
