from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import os
import random
import socket
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch import optim
from torch.nn.parallel import DistributedDataParallel

_PACKAGE_ROOT = "semoflow."
_LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(subsystem)-22s | %(message)s"
_LOG_FORMAT_DEBUG = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_CONSOLE_HANDLER = "semoflow-console"
_FILE_HANDLER = "semoflow-file"


class _SubsystemFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.subsystem = record.name.removeprefix(_PACKAGE_ROOT)
        return True


def _handler(name: str) -> logging.Handler | None:
    return next(
        (h for h in logging.getLogger().handlers if h.name == name), None
    )


def configure_logging(
    *, level: str = "INFO", log_file: Path | None = None
) -> None:
    numeric_level = logging.getLevelNamesMapping().get(level.upper())
    if numeric_level is None:
        raise ValueError(
            f"Unknown log level {level!r}. Expected DEBUG/INFO/WARNING/ERROR."
        )
    root = logging.getLogger()
    root.setLevel(numeric_level)
    formatter = logging.Formatter(
        _LOG_FORMAT_DEBUG if numeric_level <= logging.DEBUG else _LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )

    console = _handler(_CONSOLE_HANDLER)
    if console is None:
        console = logging.StreamHandler()
        console.name = _CONSOLE_HANDLER
        console.addFilter(_SubsystemFilter())
        root.addHandler(console)
    console.setFormatter(formatter)

    path = None if log_file is None else log_file.expanduser().resolve()
    current = _handler(_FILE_HANDLER)
    if current is not None and path != get_configured_log_file():
        root.removeHandler(current)
        current.close()
        current = None
    if path is not None and current is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        current = logging.FileHandler(path, encoding="utf-8")
        current.name = _FILE_HANDLER
        current.addFilter(_SubsystemFilter())
        root.addHandler(current)
    if current is not None:
        current.setFormatter(formatter)


def get_configured_log_file() -> Path | None:
    handler = _handler(_FILE_HANDLER)
    if isinstance(handler, logging.FileHandler):
        return Path(handler.baseFilename)
    return None


def _fmt_value(value: object) -> str:
    if isinstance(value, float) and not (
        value == 0.0 or 1e-3 <= abs(value) < 1e4
    ):
        return f"{value:.3e}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


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
        *(f"{name} {_fmt_value(value)}" for name, value in metrics.items()),
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


def normalize_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return normalize_value(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(k): normalize_value(v)
            for k, v in cast(Mapping[Any, Any], value).items()
        }
    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]
    return value


def to_yaml_string(obj: object) -> str:
    return yaml.dump(
        normalize_value(obj),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )


def dump_json(payload: object, *, indent: int | None = 2) -> str:
    return json.dumps(
        normalize_value(payload),
        indent=indent,
        sort_keys=True,
        default=str,
        allow_nan=False,
    )


def write_json(path: str | Path, payload: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(dump_json(payload) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def set_seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_device_spec(spec: str) -> tuple[torch.device, list[int]]:
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
    def __init__(self, *, patience: int | None) -> None:
        self.patience = patience
        self.best_value = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def step(self, value: float, *, epoch: int) -> bool:
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


def save_torch_atomic(
    path: Path, payload: object, *, force: bool = False
) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing artifact: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_artifact(path: Path, artifact_type: str) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("artifact_type") != artifact_type:
        raise ValueError(f"{path} is not an {artifact_type} artifact")
    return payload


__all__ = [
    "EarlyStopper",
    "build_lr_scheduler",
    "configure_logging",
    "ddp_worker",
    "dump_json",
    "find_free_port",
    "get_configured_log_file",
    "load_artifact",
    "log_ddp_summary",
    "log_epoch_summary",
    "log_training_step",
    "normalize_value",
    "parse_device_spec",
    "save_torch_atomic",
    "set_seed_all",
    "to_yaml_string",
    "unwrap_ddp",
    "write_json",
]
