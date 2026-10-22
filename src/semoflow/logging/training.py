from __future__ import annotations

import logging
from collections.abc import Mapping

from semoflow.logging.primitives import fmt_value


def _progress(current: int, total: int) -> str:
    return f"{current:0{len(str(total))}d}/{total}"


def log_training_step(
    logger: logging.Logger,
    *,
    step: int,
    total_steps: int,
    epoch: int,
    total_epochs: int,
    metrics: Mapping[str, float],
    lr: float,
    samples_per_sec: float | None = None,
) -> None:
    parts = [
        f"Epoch {_progress(epoch, total_epochs)}",
        f"step {_progress(step, total_steps)}",
        *(f"{name} {fmt_value(value)}" for name, value in metrics.items()),
        f"lr {lr:.2e}",
    ]
    if samples_per_sec is not None:
        parts.append(f"{int(samples_per_sec):,} sps")
    logger.info("%s", " | ".join(parts))


def log_epoch_summary(
    logger: logging.Logger,
    *,
    epoch: int,
    total_epochs: int,
    epoch_elapsed_s: float,
    samples_per_sec: float,
    gpu_allocated_gb: float,
    gpu_peak_gb: float,
) -> None:
    logger.info(
        "Epoch %s | %.1fs | %s sps | GPU %.2fGB / peak %.2fGB",
        _progress(epoch, total_epochs),
        epoch_elapsed_s,
        f"{int(samples_per_sec):,}",
        gpu_allocated_gb,
        gpu_peak_gb,
    )


def log_ddp_summary(
    logger: logging.Logger,
    *,
    device_ids: list[int],
    port: int,
    batch_per_rank: int,
) -> None:
    logger.info(
        "DDP world_size=%d  devices=%s  port=%d  batch_per_rank=%d",
        len(device_ids),
        ", ".join(f"cuda:{device}" for device in device_ids),
        port,
        batch_per_rank,
    )


__all__ = ["log_ddp_summary", "log_epoch_summary", "log_training_step"]
