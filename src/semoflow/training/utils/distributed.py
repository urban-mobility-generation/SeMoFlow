from __future__ import annotations

import contextlib
import logging
import os
import socket
from collections.abc import Generator
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from semoflow.utils.log import configure_logging


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


__all__ = ["ddp_worker", "find_free_port", "unwrap_ddp"]
