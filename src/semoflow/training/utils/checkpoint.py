from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from semoflow.training.utils.distributed import unwrap_ddp


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
    device: torch.device | str = "cpu",
) -> None:
    """Restore the model weights written by :func:`save_checkpoint`."""
    checkpoint = torch.load(
        Path(path), map_location=device, weights_only=False
    )
    unwrap_ddp(model).load_state_dict(checkpoint["model_state_dict"])


__all__ = [
    "load_checkpoint",
    "save_checkpoint",
]
