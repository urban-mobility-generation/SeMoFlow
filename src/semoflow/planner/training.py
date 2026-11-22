from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
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

from semoflow.config import PlannerConfig, RunLayout
from semoflow.planner.artifacts import (
    OccurrencePlannerDataset,
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
from semoflow.utils.log import (
    get_configured_log_file,
    log_epoch_summary,
    log_training_step,
)
from semoflow.utils.serialization import write_json
from semoflow.utils.training import (
    EarlyStopper,
    build_lr_scheduler,
    ddp_worker,
    find_free_port,
    parse_device_spec,
    set_seed_all,
)

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
    return OccurrencePlannerConfig(
        **asdict(dataset.config), **asdict(settings.architecture)
    )


def _catalog_tensors(
    config: OccurrencePlannerConfig, dataset: OccurrencePlannerDataset
) -> dict[str, torch.Tensor | None]:
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
    settings: PlannerConfig, layout: RunLayout
) -> dict[str, Any]:
    """Train on one device, or as one rank of an initialized process group."""
    options, seed = settings.training, settings.seed
    if dist.is_available() and dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        rank, world_size = 0, 1
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
    train_set = OccurrencePlannerTensorDataset(dataset.train)
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
    val_loader = build(
        OccurrencePlannerTensorDataset(dataset.validation), shuffle=False
    )
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

    stopper = EarlyStopper(patience=options.early_stopping_patience)
    history: list[dict[str, float]] = []
    validation: dict[str, float] = {}
    step = 0
    epoch = 0
    for epoch in range(1, options.epochs + 1):
        if device.type == "cuda":
            _synchronize(device)
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
                log_training_step(
                    logger,
                    step=step,
                    total_steps=total_steps,
                    epoch=epoch,
                    total_epochs=options.epochs,
                    metrics={
                        "loss": float(loss.total.detach()),
                        "prefix_loss": float(loss.prefix.detach()),
                        "gap_loss": float(loss.gap.detach()),
                        "length_loss": float(loss.length.detach()),
                    },
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
        improved = stopper.step(validation["nll"], epoch=epoch)
        if ctx.primary:
            if improved:
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
                "  (best)" if improved else "",
            )
        if stopper.should_stop:
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
            force=True,
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
    return summary


def _spawn_worker(
    rank: int,
    device_ids: list[int],
    port: int,
    settings: PlannerConfig,
    layout: RunLayout,
    log_level: str,
    log_file: Path | None,
) -> None:
    with ddp_worker(rank, device_ids, port, log_level, log_file):
        train_occurrence_planner(settings, layout)


def train_planner(
    settings: PlannerConfig, layout: RunLayout, *, log_level: str = "INFO"
) -> dict[str, Any]:
    """Train in this process, or spawn one DDP rank per listed GPU."""
    _, device_ids = parse_device_spec(settings.training.device)
    if len(device_ids) < 2:
        return train_occurrence_planner(settings, layout)
    mp.spawn(
        _spawn_worker,
        args=(
            device_ids,
            find_free_port(),
            settings,
            layout,
            log_level,
            get_configured_log_file(),
        ),
        nprocs=len(device_ids),
        join=True,
    )
    metrics = json.loads((layout.planner_dir / "metrics.json").read_text())
    metrics.pop("epochs")
    return metrics


__all__ = ["train_occurrence_planner", "train_planner"]
