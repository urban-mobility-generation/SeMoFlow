from __future__ import annotations

import contextlib
import datetime
import logging
import os
import socket
from collections.abc import Generator
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def setup_ddp(rank: int, world_size: int, device_id: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_kwargs: dict = {
        "backend": backend,
        "rank": rank,
        "world_size": world_size,
    }
    if torch.cuda.is_available():
        init_kwargs["device_id"] = torch.device(f"cuda:{device_id}")
    timeout_seconds = int(os.environ.get("SEMOFLOW_DDP_TIMEOUT_SECONDS", "0"))
    if timeout_seconds > 0:
        init_kwargs["timeout"] = datetime.timedelta(seconds=timeout_seconds)
    dist.init_process_group(**init_kwargs)
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)


def teardown_ddp() -> None:
    dist.destroy_process_group()


def init_worker(
    rank: int,
    device_ids: list[int],
    port: int,
    log_level: str,
    log_file: Path | None,
    *,
    log_all_ranks: bool = False,
) -> torch.device:
    from semoflow.logging.primitives import configure_logging

    configure_logging(
        level=log_level,
        log_file=log_file if (rank == 0 or log_all_ranks) else None,
    )
    if rank == 0:
        logger.info(
            "initialising  world_size=%d  devices=%s",
            len(device_ids),
            device_ids,
        )
    else:
        logger.debug(
            "rank=%d pid=%d device=cuda:%d  initialising",
            rank,
            os.getpid(),
            device_ids[rank],
        )
    setup_ddp(rank, len(device_ids), device_ids[rank], port)
    if rank == 0:
        logger.info("process group ready")
    else:
        logger.debug("rank=%d  process group ready", rank)
    return (
        torch.device(f"cuda:{device_ids[rank]}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


@contextlib.contextmanager
def ddp_worker(
    rank: int,
    device_ids: list[int],
    port: int,
    log_level: str = "INFO",
    log_file: Path | None = None,
    *,
    log_all_ranks: bool = False,
) -> Generator[torch.device, None, None]:
    device = init_worker(
        rank,
        device_ids,
        port,
        log_level,
        log_file,
        log_all_ranks=log_all_ranks,
    )
    try:
        yield device
    finally:
        teardown_ddp()


def unwrap_ddp(model: nn.Module) -> nn.Module:
    m = model.module if isinstance(model, DistributedDataParallel) else model
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


__all__ = [
    "ddp_worker",
    "find_free_port",
    "init_worker",
    "setup_ddp",
    "teardown_ddp",
    "unwrap_ddp",
]
