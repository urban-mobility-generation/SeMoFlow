"""Seeding, device, DDP, early-stopping and LR-schedule helpers shared by
every training stage."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import random
import socket
from collections.abc import Generator
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel

from semoflow.utils.log import configure_logging


def set_seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_device_spec(spec: str) -> tuple[torch.device, list[int]]:
    """A device, or ``[i,j,...]``: a GPU list whose first entry is primary."""
    stripped = spec.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        ids = [int(x) for x in stripped[1:-1].split(",") if x.strip()]
        if not ids:
            raise ValueError(f"Empty device list in spec: {spec!r}")
        return torch.device(f"cuda:{ids[0]}"), ids
    return torch.device(stripped), []


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def ddp_worker(
    rank: int,
    device_ids: list[int],
    port: int,
    log_level: str = "INFO",
    log_file: Path | None = None,
) -> Generator[torch.device, None, None]:
    """One spawned rank of a single-node NCCL group over ``device_ids``.

    Only rank 0 logs. Yields this rank's device.
    """
    if rank == 0:
        configure_logging(level=log_level, log_file=log_file)
    else:
        logging.disable(logging.CRITICAL)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    device = torch.device("cuda", device_ids[rank])
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=len(device_ids), device_id=device
    )
    try:
        yield device
    finally:
        dist.destroy_process_group()


def unwrap_ddp(model: nn.Module) -> nn.Module:
    m = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(m, "_orig_mod", m)


class EarlyStopper:
    """Track the best (lowest) validation value and stop after
    ``patience`` epochs without improvement; ``None`` never stops."""

    def __init__(self, *, patience: int | None) -> None:
        self.patience = patience
        self.best_value = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def step(self, value: float, *, epoch: int) -> bool:
        """Record one epoch's value; whether it is a new best."""
        if value < self.best_value:
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return (
            self.patience is not None
            and self.epochs_without_improvement >= self.patience
        )


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    *,
    schedule: str,
    warmup_steps: int,
    total_updates: int,
    base_lr: float,
    min_lr: float,
) -> optim.lr_scheduler.LRScheduler | None:
    """Linear warmup, then constant or cosine decay to ``min_lr``; ``None``
    for a constant rate without warmup."""
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


__all__ = [
    "EarlyStopper",
    "build_lr_scheduler",
    "ddp_worker",
    "find_free_port",
    "parse_device_spec",
    "set_seed_all",
    "unwrap_ddp",
]
