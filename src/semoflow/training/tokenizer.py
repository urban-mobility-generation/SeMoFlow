from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
from torch import nn

__all__ = ["TensorDictDataset", "train_tokenizer"]

logger = logging.getLogger(__name__)

Batches = Iterable[dict[str, torch.Tensor]]


class TensorDictDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors
        self.length = next(iter(tensors.values())).shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: tensor[idx] for key, tensor in self.tensors.items()}


def _run_epoch(
    model: nn.Module,
    batches: Batches,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    totals: dict[str, torch.Tensor] = {}
    count = 0
    model.train(optimizer is not None)
    with torch.set_grad_enabled(optimizer is not None):
        for batch in batches:
            batch = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                out = model(batch, return_loss=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                out.loss.backward()
                optimizer.step()
            for key, value in out.losses.items():
                totals[key] = totals.get(key, 0) + value.detach()
            count += 1
    return {
        key: float(value.float().cpu().item() / max(count, 1))
        for key, value in totals.items()
    }


def _log(epoch: int, epochs: int, metrics: dict[str, float]) -> None:
    text = "  ".join(f"{key}={value:.6g}" for key, value in metrics.items())
    logger.info("[tokenizer] epoch %d/%d  %s", epoch, epochs, text)


def train_tokenizer(
    model: nn.Module,
    batches: Batches,
    *,
    epochs: int,
    lr: float,
    device: torch.device | str,
) -> list[dict[str, float]]:
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    _log(0, epochs, _run_epoch(model, batches, device, None))
    history = []
    for epoch in range(1, epochs + 1):
        metrics = _run_epoch(model, batches, device, optimizer)
        _log(epoch, epochs, metrics)
        history.append({**metrics, "epoch": float(epoch)})
    return history
