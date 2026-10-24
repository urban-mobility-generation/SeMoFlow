from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from semoflow.training.utils.mixed_precision import build_amp_context

__all__ = ["TensorDictDataset", "train_tokenizer"]

logger = logging.getLogger(__name__)


class TensorDictDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        if not tensors:
            raise ValueError("tensors must be non-empty")
        lengths = {tensor.shape[0] for tensor in tensors.values()}
        if len(lengths) != 1:
            raise ValueError(f"all tensors must share length, got {lengths}")
        self.tensors = tensors
        self.length = next(iter(lengths))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: tensor[idx] for key, tensor in self.tensors.items()}


def _batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return {
        key: value.to(device, non_blocking=non_blocking)
        for key, value in batch.items()
    }


def _materialize_metrics(
    totals: dict[str, torch.Tensor], n_batches: int
) -> dict[str, float]:
    denominator = max(n_batches, 1)
    return {
        key: float(value.detach().float().cpu().item() / denominator)
        for key, value in totals.items()
    }


def _loss_metrics(
    model: nn.Module,
    dataloader: DataLoader[dict[str, torch.Tensor]]
    | Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    totals: dict[str, torch.Tensor] = {}
    n_batches = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = _batch_to_device(batch, device)
            out = model(batch, return_loss=True)
            if out.loss is None:
                raise RuntimeError(
                    "tokenizer returned no loss while return_loss=True"
                )
            loss_map = out.losses or {"loss": out.loss}
            for key, value in loss_map.items():
                detached = value.detach()
                if key in totals:
                    totals[key] += detached
                else:
                    totals[key] = detached.clone()
            n_batches += 1
    if was_training:
        model.train()
    return _materialize_metrics(totals, n_batches)


def _log_epoch_metrics(
    *,
    epoch: int,
    epochs: int,
    metrics: dict[str, float],
) -> None:
    metric_text = "  ".join(
        f"{key}={value:.6g}" for key, value in metrics.items()
    )
    logger.info("[tokenizer] epoch %d/%d  %s", epoch, epochs, metric_text)


def train_tokenizer(
    model: nn.Module,
    dataloader: DataLoader[dict[str, torch.Tensor]]
    | Iterable[dict[str, torch.Tensor]],
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    mixed_precision: str = "bf16",
    checkpoint_dir: str | Path | None = None,
) -> list[dict[str, float]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    resolved_device = torch.device(device)
    model.to(resolved_device)
    amp_ctx, scaler = build_amp_context(mixed_precision, resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    initial_metrics = _loss_metrics(model, dataloader, resolved_device)
    _log_epoch_metrics(epoch=0, epochs=epochs, metrics=initial_metrics)
    for epoch in range(epochs):
        model.train()
        totals: dict[str, torch.Tensor] = {}
        n_batches = 0
        for batch in dataloader:
            batch = _batch_to_device(batch, resolved_device)
            with amp_ctx:
                out = model(batch, return_loss=True)
            if out.loss is None:
                raise RuntimeError(
                    "tokenizer returned no loss while return_loss=True"
                )
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(out.loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out.loss.backward()
                optimizer.step()

            loss_map = out.losses or {"loss": out.loss}
            for key, value in loss_map.items():
                detached = value.detach()
                if key in totals:
                    totals[key] += detached
                else:
                    totals[key] = detached.clone()
            n_batches += 1

        epoch_metrics = _materialize_metrics(totals, n_batches)
        epoch_metrics["epoch"] = float(epoch + 1)
        history.append(epoch_metrics)
        _log_epoch_metrics(
            epoch=epoch + 1,
            epochs=epochs,
            metrics={
                key: value
                for key, value in epoch_metrics.items()
                if key != "epoch"
            },
        )

        current_loss = float(epoch_metrics.get("loss", float("inf")))
        if ckpt_dir is not None and current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": best_epoch,
                    "best_epoch": best_epoch,
                    "best_metric": "loss",
                    "best_loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": epoch_metrics,
                },
                ckpt_dir / "best.pt",
            )

    if ckpt_dir is not None:
        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": history[-1],
            },
            ckpt_dir / "last.pt",
        )

    return history
