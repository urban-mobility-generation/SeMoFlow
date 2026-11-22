from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from semoflow.config.artifacts import RunLayout
from semoflow.config.schema import PlannerConfig
from semoflow.planner.artifacts import (
    OccurrencePlannerDataset,
    OccurrencePlannerSplit,
    load_occurrence_planner_dataset,
    save_occurrence_planner_checkpoint,
)
from semoflow.planner.config import OccurrencePlannerConfig
from semoflow.planner.data import (
    OccurrencePlannerTensorDataset,
    build_occurrence_tensor_dataloader,
)
from semoflow.planner.loss import occurrence_planner_loss
from semoflow.planner.model import OccurrenceARPlanner
from semoflow.training.utils.distributed import find_free_port
from semoflow.training.utils.early_stopping import EarlyStopper
from semoflow.training.utils.lr_schedule import build_lr_scheduler
from semoflow.training.utils.runtime import parse_device_spec, set_seed_all
from semoflow.utils.log import (
    configure_logging,
    get_configured_log_file,
    log_epoch_summary,
    log_training_step,
)
from semoflow.utils.serialization import write_json

logger = logging.getLogger(__name__)

_OUTPUT_NAMES = ("best.pt", "last.pt", "metrics.json")
_METRIC_NAMES = (
    "nll",
    "length_nll",
    "gap_nll",
    "prefix_nll",
    "nll_per_event",
    "gap_nll_per_event",
    "prefix_nll_per_event",
)


@dataclass(frozen=True, slots=True)
class _Rank:
    rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


class _NLL:
    """Sequence and event totals of the planner's NLL terms."""

    def __init__(self) -> None:
        self.sums = [0.0] * 6

    def add(self, loss: Any, batch: Any) -> None:
        sequences = int(batch.lengths.shape[0])
        self.sums[0] += sequences
        self.sums[1] += int(batch.valid_mask.sum().item())
        for index, term in enumerate(
            (loss.total, loss.length, loss.gap, loss.prefix), start=2
        ):
            self.sums[index] += float(term.detach()) * sequences

    def all_reduce(self, rank: _Rank) -> None:
        if rank.distributed:
            totals = torch.tensor(
                self.sums, dtype=torch.float64, device=rank.device
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            self.sums = totals.tolist()

    def metrics(self) -> dict[str, float]:
        sequences, events, total, length, gap, prefix = self.sums
        values = (
            total / sequences,
            length / sequences,
            gap / sequences,
            prefix / sequences,
            total / events,
            gap / events,
            prefix / events,
        )
        return dict(zip(_METRIC_NAMES, values, strict=True))


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def _split_dataset(
    split: OccurrencePlannerSplit,
) -> OccurrencePlannerTensorDataset:
    return OccurrencePlannerTensorDataset(
        split.prefix_codes, split.gap_tokens, split.lengths, split.city_ids
    )


def _evaluate(
    model: OccurrenceARPlanner, loader: DataLoader, rank: _Rank
) -> dict[str, float]:
    """NLL on ``loader``, computed on the primary rank and shared."""
    values = torch.zeros(
        len(_METRIC_NAMES), dtype=torch.float64, device=rank.device
    )
    if rank.primary:
        nll = _NLL()
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(rank.device, non_blocking=True)
                nll.add(occurrence_planner_loss(model(batch), batch), batch)
        model.train()
        values[:] = torch.tensor(
            list(nll.metrics().values()), dtype=torch.float64
        )
    if rank.distributed:
        dist.broadcast(values, src=0)
    return dict(zip(_METRIC_NAMES, values.tolist(), strict=True))


def _planner_config(
    dataset: OccurrencePlannerDataset, settings: PlannerConfig
) -> OccurrencePlannerConfig:
    data, architecture = dataset.config, settings.architecture
    return OccurrencePlannerConfig(
        codebook_size=data.codebook_size,
        gap_vocab=data.gap_vocab,
        prefix_depth=data.prefix_depth,
        max_occurrences=data.max_occurrences,
        city_conditioning=data.city_conditioning,
        model_dim=architecture.model_dim,
        n_heads=architecture.n_heads,
        n_layers=architecture.n_layers,
        dropout=architecture.dropout,
        gap_codec_version=data.gap_codec_version,
        use_distance_prior=architecture.use_distance_prior,
        gap_bucket_edges_minutes=data.gap_bucket_edges_minutes,
        gap_duration_minutes=data.gap_duration_minutes,
        observation_window_minutes=data.observation_window_minutes,
    )


def _catalog_tensors(
    config: OccurrencePlannerConfig, dataset: OccurrencePlannerDataset
) -> dict[str, torch.Tensor | None]:
    if config.use_distance_prior and dataset.legal_prefix_centroids is None:
        raise ValueError(
            "architecture.use_distance_prior requires a planner dataset "
            "whose places carry coordinates"
        )
    centroids = config.use_distance_prior
    return {
        "legal_prefixes": dataset.legal_prefixes,
        "catalog_city_ids": dataset.catalog_city_ids,
        "city_legal_prefix_mask": dataset.city_legal_prefix_mask,
        "legal_prefix_centroids": (
            dataset.legal_prefix_centroids if centroids else None
        ),
        "legal_prefix_centroid_available": (
            dataset.legal_prefix_centroid_available if centroids else None
        ),
    }


def check_planner_settings(
    settings: PlannerConfig, layout: RunLayout, *, force: bool = False
) -> None:
    """Fail fast on unusable GPUs or outputs that would be overwritten."""
    _, device_ids = parse_device_spec(settings.training.device)
    if device_ids:
        if len(set(device_ids)) != len(device_ids) or min(device_ids) < 0:
            raise ValueError("device IDs must be unique non-negative integers")
        if not torch.cuda.is_available():
            raise ValueError("a GPU device list requires CUDA")
        if max(device_ids) >= torch.cuda.device_count():
            raise ValueError(
                f"device IDs {device_ids} exceed the "
                f"{torch.cuda.device_count()} visible CUDA devices"
            )
    existing = [
        str(path)
        for name in _OUTPUT_NAMES
        if (path := layout.planner_dir / name).exists()
    ]
    if existing and not force:
        raise FileExistsError(
            f"refusing to overwrite {', '.join(existing)}; pass --force"
        )


def _log_step(
    loss: Any,
    *,
    step: int,
    total_steps: int,
    epoch: int,
    total_epochs: int,
    lr: float,
    samples_per_sec: float,
) -> None:
    log_training_step(
        logger,
        step=step,
        total_steps=total_steps,
        epoch=epoch,
        total_epochs=total_epochs,
        metrics={
            "loss": float(loss.total.detach()),
            "prefix_loss": float(loss.prefix.detach()),
            "gap_loss": float(loss.gap.detach()),
            "length_loss": float(loss.length.detach()),
        },
        lr=lr,
        samples_per_sec=samples_per_sec,
    )


def _should_log(step: int, steps_per_epoch: int, total_steps: int) -> bool:
    interval = max(1, min(50, steps_per_epoch // 5))
    return (
        step == 1
        or step % interval == 0
        or step % steps_per_epoch == 0
        or step == total_steps
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_occurrence_planner(
    settings: PlannerConfig,
    layout: RunLayout,
    *,
    force: bool = False,
    rank: int = 0,
    device_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Train on one device, or as one rank of ``device_ids``."""
    options, seed = settings.training, settings.seed
    world_size = len(device_ids) if device_ids else 1
    if world_size > 1:
        assert device_ids is not None
        device = torch.device("cuda", device_ids[rank])
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=world_size, device_id=device
        )
    else:
        device = parse_device_spec(options.device)[0]
    ctx = _Rank(rank, world_size, device)

    dataset = load_occurrence_planner_dataset(layout.planner_dataset)
    if ctx.primary:
        logger.info(
            "loaded planner dataset  path=%s  train=%d  validation=%d  "
            "codebook_size=%d  gap_vocab=%d",
            layout.planner_dataset,
            dataset.train.size,
            dataset.validation.size,
            dataset.config.codebook_size,
            dataset.config.gap_vocab,
        )
    config = _planner_config(dataset, settings)
    check_planner_settings(settings, layout, force=force)
    best_path, last_path, metrics_path = (
        layout.planner_dir / name for name in _OUTPUT_NAMES
    )

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    set_seed_all(seed)
    catalog = _catalog_tensors(config, dataset)
    model = OccurrenceARPlanner(config, **catalog).to(device)
    if options.compile_model:
        model.compile(fullgraph=True)
    train_model: nn.Module = model
    if ctx.distributed:
        train_model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
        )
    set_seed_all(seed + rank)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    if ctx.primary:
        logger.info(
            "starting training  device=%s  world_size=%d  seed=%d  "
            "epochs=%d  batch_size=%d  learning_rate=%g  model_dim=%d  "
            "n_layers=%d  lr_scheduler=%s  warmup_steps=%d  min_lr=%g  "
            "compile=%s",
            device,
            world_size,
            seed,
            options.epochs,
            options.batch_size,
            options.learning_rate,
            config.model_dim,
            config.n_layers,
            options.lr_scheduler,
            options.warmup_steps,
            options.min_lr,
            options.compile_model,
        )

    build = partial(
        build_occurrence_tensor_dataloader,
        config=config,
        legality=model.legality,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_set = _split_dataset(dataset.train)
    sampler = (
        DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        if ctx.distributed
        else None
    )
    train_loader = build(
        train_set, shuffle=sampler is None, seed=seed, sampler=sampler
    )
    val_loader = build(_split_dataset(dataset.validation), shuffle=False)
    train_eval_loader = build(train_set, shuffle=False)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * options.epochs
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=options.lr_scheduler,
        warmup_steps=options.warmup_steps,
        total_updates=total_steps,
        base_lr=options.learning_rate,
        min_lr=options.min_lr,
    )
    save = partial(
        save_occurrence_planner_checkpoint,
        config=config,
        seed=seed,
        options=options,
        **catalog,
    )

    stopper = EarlyStopper(patience=options.early_stopping_patience or None)
    history: list[dict[str, float]] = []
    validation: dict[str, float] = {}
    step = 0
    epoch = 0
    for epoch in range(1, options.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_model.train()
        nll = _NLL()
        window_start, window_samples = time.perf_counter(), 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            loss = occurrence_planner_loss(train_model(batch), batch)
            nll.add(loss, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1
            window_samples += int(batch.lengths.shape[0])
            if ctx.primary and _should_log(step, steps_per_epoch, total_steps):
                _synchronize(device)
                now = time.perf_counter()
                _log_step(
                    loss,
                    step=step,
                    total_steps=total_steps,
                    epoch=epoch,
                    total_epochs=options.epochs,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    samples_per_sec=window_samples
                    / max(now - window_start, 1e-6),
                )
                window_start, window_samples = now, 0
        nll.all_reduce(ctx)
        _synchronize(device)
        train_elapsed = time.perf_counter() - started

        validation_started = time.perf_counter()
        validation = _evaluate(model, val_loader, ctx)
        _synchronize(device)
        validation_elapsed = time.perf_counter() - validation_started
        decision = stopper.step(validation["nll"], epoch=epoch)
        if ctx.primary:
            if decision.improved:
                save(
                    best_path,
                    model_state=model.state_dict(),
                    epoch=epoch,
                    validation_metric=stopper.best_value,
                    force=True,
                )
            train = nll.metrics()
            row = {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_elapsed_s": train_elapsed,
                "validation_elapsed_s": validation_elapsed,
                "train_samples_per_sec": nll.sums[0]
                / max(train_elapsed, 1e-6),
                **_prefixed("train", train),
                **_prefixed("validation", validation),
            }
            history.append(row)
            cuda = device.type == "cuda"
            log_epoch_summary(
                logger,
                epoch=epoch,
                total_epochs=options.epochs,
                epoch_elapsed_s=train_elapsed,
                samples_per_sec=row["train_samples_per_sec"],
                gpu_allocated_gb=torch.cuda.memory_allocated(device) / 1e9
                if cuda
                else 0.0,
                gpu_peak_gb=torch.cuda.max_memory_allocated(device) / 1e9
                if cuda
                else 0.0,
            )
            logger.info(
                "epoch %d/%d  lr=%.6g  train_nll=%.4f  validation_nll=%.4f  "
                "val[length=%.4f gap=%.4f prefix=%.4f]%s",
                epoch,
                options.epochs,
                row["learning_rate"],
                train["nll"],
                validation["nll"],
                validation["length_nll"],
                validation["gap_nll"],
                validation["prefix_nll"],
                "  (best)" if decision.improved else "",
            )
        if decision.should_stop:
            if ctx.primary:
                logger.info(
                    "early stopping after epoch %d: best epoch %d, "
                    "validation NLL %.4f",
                    epoch,
                    stopper.best_epoch,
                    stopper.best_value,
                )
            break

    final_train = _evaluate(model, train_eval_loader, ctx)
    summary: dict[str, Any] = {
        "best_epoch": stopper.best_epoch,
        "best_validation_nll": stopper.best_value,
        **_prefixed("final_train", final_train),
        **_prefixed("final_validation", validation),
    }
    if ctx.primary:
        save(
            last_path,
            model_state=model.state_dict(),
            epoch=epoch,
            validation_metric=validation["nll"],
            force=force,
        )
        write_json(metrics_path, {**summary, "epochs": history})
        logger.info(
            "training complete  best_epoch=%d  best_validation_nll=%.4f  "
            "final_train_nll=%.4f  output_dir=%s",
            stopper.best_epoch,
            stopper.best_value,
            final_train["nll"],
            layout.planner_dir,
        )
    if ctx.distributed:
        dist.barrier()
        dist.destroy_process_group()
    return summary


def _spawn_worker(
    rank: int,
    device_ids: list[int],
    port: int,
    settings: PlannerConfig,
    layout: RunLayout,
    log_level: str,
    log_file: Path | None,
    force: bool,
) -> None:
    if rank == 0:
        configure_logging(level=log_level, log_file=log_file)
    else:
        logging.disable(logging.CRITICAL)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    train_occurrence_planner(
        settings, layout, force=force, rank=rank, device_ids=device_ids
    )


def train_planner(
    settings: PlannerConfig,
    layout: RunLayout,
    *,
    force: bool = False,
    log_level: str = "INFO",
) -> dict[str, Any]:
    """Train in this process, or spawn one DDP rank per listed GPU."""
    _, device_ids = parse_device_spec(settings.training.device)
    if len(device_ids) < 2:
        return train_occurrence_planner(settings, layout, force=force)
    mp.spawn(
        _spawn_worker,
        args=(
            device_ids,
            find_free_port(),
            settings,
            layout,
            log_level,
            get_configured_log_file(),
            force,
        ),
        nprocs=len(device_ids),
        join=True,
    )
    metrics = json.loads((layout.planner_dir / "metrics.json").read_text())
    metrics.pop("epochs")
    return metrics


__all__ = [
    "check_planner_settings",
    "train_occurrence_planner",
    "train_planner",
]
