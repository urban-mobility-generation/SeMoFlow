from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from semoflow.training.utils.distributed import unwrap_ddp

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    extra: dict[str, Any] | None = None,
) -> None:
    raw_model = unwrap_ddp(model)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }
    if extra:
        payload.update(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
    allow_missing_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    ckpt: dict[str, Any] = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )
    raw_model = unwrap_ddp(model)
    state = ckpt["model_state_dict"]
    if allow_missing_prefixes:
        target_state = raw_model.state_dict()
        missing = set(target_state) - set(state)
        unexpected = set(state) - set(target_state)
        allowed_missing = {
            key
            for key in missing
            if any(key.startswith(prefix) for prefix in allow_missing_prefixes)
        }
        if missing and missing == allowed_missing and not unexpected:
            logger.warning(
                "warm-start checkpoint omits %d allowed model tensors under "
                "%s; retaining fresh initialization",
                len(missing),
                allow_missing_prefixes,
            )
            merged_state = dict(target_state)
            merged_state.update(state)
            state = merged_state
    raw_model.load_state_dict(state)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


__all__ = [
    "load_checkpoint",
    "save_checkpoint",
]
