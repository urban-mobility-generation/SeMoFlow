from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Sampler,
    SequentialSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler

from semoflow.config import PlannerConfig, RunLayout
from semoflow.data import DATASET_ARTIFACT_TYPE, reconstruct_event_time_minutes
from semoflow.planner.model import (
    OccurrenceARPlanner,
    OccurrenceBatch,
    OccurrencePlannerConfig,
    occurrence_planner_loss,
    save_occurrence_planner_checkpoint,
)
from semoflow.planner.prefix_decoder import LegalPrefixIndex
from semoflow.utils import (
    EarlyStopper,
    build_lr_scheduler,
    load_artifact,
    log_epoch_summary,
    log_training_step,
    run_on_devices,
    set_seed_all,
    write_json,
)


@dataclass(frozen=True, slots=True)
class OccurrencePlannerDataConfig:
    codebook_size: int
    gap_vocab: int
    prefix_depth: int
    max_occurrences: int
    city_conditioning: bool
    gap_bucket_edges_minutes: tuple[float, ...]
    gap_duration_minutes: tuple[float, ...]
    observation_window_minutes: float


@dataclass(frozen=True, slots=True)
class OccurrencePlannerSplit:
    prefix_codes: torch.Tensor
    gap_tokens: torch.Tensor
    lengths: torch.Tensor
    city_ids: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.lengths.shape[0])


@dataclass(frozen=True, slots=True)
class OccurrencePlannerDataset:
    config: OccurrencePlannerDataConfig
    legal_prefixes: torch.Tensor
    catalog_city_ids: torch.Tensor
    city_legal_prefix_mask: torch.Tensor
    train: OccurrencePlannerSplit
    validation: OccurrencePlannerSplit
    legal_prefix_centroids: torch.Tensor | None = None
    legal_prefix_centroid_available: torch.Tensor | None = None


def load_occurrence_planner_dataset(path: Path) -> OccurrencePlannerDataset:
    payload = load_artifact(path, DATASET_ARTIFACT_TYPE)
    return OccurrencePlannerDataset(
        config=OccurrencePlannerDataConfig(**payload["data_config"]),
        legal_prefixes=payload["legal_prefixes"],
        legal_prefix_centroids=payload["legal_prefix_centroids"],
        legal_prefix_centroid_available=payload[
            "legal_prefix_centroid_available"
        ],
        catalog_city_ids=payload["catalog_city_ids"],
        city_legal_prefix_mask=payload["city_legal_prefix_mask"],
        train=OccurrencePlannerSplit(**payload["train"]),
        validation=OccurrencePlannerSplit(**payload["validation"]),
    )


def _shift_right(
    values: torch.Tensor, valid_mask: torch.Tensor, fill: float
) -> torch.Tensor:
    shifted = torch.full_like(values, fill)
    previous_valid = valid_mask[:, 1:]
    if values.ndim == 3:
        previous_valid = previous_valid[..., None]
    shifted[:, 1:] = torch.where(previous_valid, values[:, :-1], fill)
    return shifted


def collate_occurrence_tensors(
    tensors: tuple[torch.Tensor, ...],
    *,
    config: OccurrencePlannerConfig,
    legality: LegalPrefixIndex,
) -> OccurrenceBatch:
    prefix_codes, gap_tokens, lengths, city_ids = tensors
    batch_size = prefix_codes.shape[0]
    device = prefix_codes.device
    max_len = int(lengths.max().item())

    sub_prefix = prefix_codes[:, :max_len]
    sub_gaps = gap_tokens[:, :max_len]
    valid_mask = (
        torch.arange(max_len, device=device)[None, :] < lengths[:, None]
    )
    default_prefix = (
        legality.codes[0]
        .to(device)
        .view(1, 1, -1)
        .expand(batch_size, max_len, -1)
    )
    target_prefixes = torch.where(
        valid_mask[..., None], sub_prefix, default_prefix
    )
    target_gaps = torch.where(valid_mask, sub_gaps, 0)
    event_times = reconstruct_event_time_minutes(
        target_gaps,
        valid_mask,
        representative_minutes=config.gap_duration_minutes,
        observation_window_minutes=config.observation_window_minutes,
    )
    return OccurrenceBatch(
        previous_prefix_codes=_shift_right(sub_prefix, valid_mask, -1),
        previous_gap_tokens=_shift_right(sub_gaps, valid_mask, -1),
        previous_event_time_minutes=_shift_right(event_times, valid_mask, 0.0),
        target_prefix_codes=target_prefixes,
        target_gap_tokens=target_gaps,
        valid_mask=valid_mask,
        lengths=lengths,
        city_ids=city_ids,
    )


class SeededShuffleSampler(Sampler[int]):
    def __init__(self, size: int, *, seed: int) -> None:
        self._size = int(size)
        self._random = random.Random(seed)

    def __iter__(self) -> Iterator[int]:
        order = list(range(self._size))
        self._random.shuffle(order)
        return iter(order)

    def __len__(self) -> int:
        return self._size


def build_occurrence_tensor_dataloader(
    split: OccurrencePlannerSplit,
    *,
    config: OccurrencePlannerConfig,
    legality: LegalPrefixIndex,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    dataset = TensorDataset(
        split.prefix_codes, split.gap_tokens, split.lengths, split.city_ids
    )
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=BatchSampler(
            SequentialSampler(dataset) if sampler is None else sampler,
            batch_size=batch_size,
            drop_last=False,
        ),
        collate_fn=partial(
            collate_occurrence_tensors, config=config, legality=legality
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
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
        **asdict(dataset.config),
        model_dim=settings.model_dim,
        n_heads=settings.n_heads,
        n_layers=settings.n_layers,
        dropout=settings.dropout,
        use_distance_prior=settings.use_distance_prior,
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


@dataclass
class _Run:
    ctx: _Rank
    settings: PlannerConfig
    config: OccurrencePlannerConfig
    model: OccurrenceARPlanner
    train_model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler | None
    sampler: Sampler[int]
    train_loader: DataLoader
    val_loader: DataLoader
    train_eval_loader: DataLoader
    save: Callable[..., None]

    @property
    def total_steps(self) -> int:
        return len(self.train_loader) * self.settings.epochs


def _setup(settings: PlannerConfig, layout: RunLayout, ctx: _Rank) -> _Run:
    seed = settings.seed
    device = ctx.device
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
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    set_seed_all(seed)
    catalog = _catalog_tensors(config, dataset)
    model = OccurrenceARPlanner(config, **catalog).to(device)
    if settings.compile:
        model.compile(fullgraph=True)
    train_model: nn.Module = model
    if ctx.distributed:
        train_model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
        )
    set_seed_all(seed + ctx.rank)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    if ctx.primary:
        logger.info(
            "starting training  device=%s  world_size=%d  seed=%d  "
            "epochs=%d  batch_size=%d  learning_rate=%g  model_dim=%d  "
            "n_layers=%d  lr_scheduler=%s  warmup_steps=%d  min_lr=%g  "
            "compile=%s",
            device,
            ctx.world_size,
            seed,
            settings.epochs,
            settings.batch_size,
            settings.learning_rate,
            config.model_dim,
            config.n_layers,
            settings.lr_scheduler,
            settings.warmup_steps,
            settings.min_lr,
            settings.compile,
        )
    build = partial(
        build_occurrence_tensor_dataloader,
        config=config,
        legality=model.legality,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
    )
    sampler = (
        DistributedSampler(
            range(dataset.train.size),
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        if ctx.distributed
        else SeededShuffleSampler(dataset.train.size, seed=seed)
    )
    train_loader = build(dataset.train, sampler=sampler)
    return _Run(
        ctx=ctx,
        settings=settings,
        config=config,
        model=model,
        train_model=train_model,
        optimizer=optimizer,
        scheduler=build_lr_scheduler(
            optimizer,
            schedule=settings.lr_scheduler,
            warmup_steps=settings.warmup_steps,
            total_updates=len(train_loader) * settings.epochs,
            base_lr=settings.learning_rate,
            min_lr=settings.min_lr,
        ),
        sampler=sampler,
        train_loader=train_loader,
        val_loader=build(dataset.validation),
        train_eval_loader=build(dataset.train),
        save=partial(
            save_occurrence_planner_checkpoint,
            config=config,
            settings=settings,
            **catalog,
        ),
    )


def _train_epoch(run: _Run, epoch: int, step: int) -> tuple[_NLL, int]:
    device, optimizer = run.ctx.device, run.optimizer
    steps_per_epoch = len(run.train_loader)
    if isinstance(run.sampler, DistributedSampler):
        run.sampler.set_epoch(epoch)
    run.train_model.train()
    nll = _NLL()
    window_start, window_samples = time.perf_counter(), 0
    for batch in run.train_loader:
        batch = batch.to(device, non_blocking=True)
        loss = occurrence_planner_loss(run.train_model(batch), batch)
        nll.add(loss, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        optimizer.step()
        if run.scheduler is not None:
            run.scheduler.step()
        step += 1
        window_samples += int(batch.lengths.shape[0])
        if run.ctx.primary and _should_log(
            step, steps_per_epoch, run.total_steps
        ):
            _synchronize(device)
            now = time.perf_counter()
            log_training_step(
                logger,
                step=step,
                total_steps=run.total_steps,
                epoch=epoch,
                total_epochs=run.settings.epochs,
                metrics={
                    "loss": float(loss.total.detach()),
                    "prefix_loss": float(loss.prefix.detach()),
                    "gap_loss": float(loss.gap.detach()),
                    "length_loss": float(loss.length.detach()),
                },
                lr=float(optimizer.param_groups[0]["lr"]),
                samples_per_sec=window_samples / max(now - window_start, 1e-6),
            )
            window_start, window_samples = now, 0
    nll.all_reduce(run.ctx)
    _synchronize(device)
    return nll, step


def _log_epoch(
    run: _Run,
    epoch: int,
    row: dict[str, float],
    train: dict[str, float],
    validation: dict[str, float],
    improved: bool,
) -> None:
    device = run.ctx.device
    cuda = device.type == "cuda"
    log_epoch_summary(
        logger,
        epoch=epoch,
        total_epochs=run.settings.epochs,
        epoch_elapsed_s=row["train_elapsed_s"],
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
        run.settings.epochs,
        row["learning_rate"],
        train["nll"],
        validation["nll"],
        validation["length_nll"],
        validation["gap_nll"],
        validation["prefix_nll"],
        "  (best)" if improved else "",
    )


def _train_on_device(
    device: torch.device,
    rank: int,
    world_size: int,
    settings: PlannerConfig,
    layout: RunLayout,
) -> dict[str, Any]:
    run = _setup(settings, layout, _Rank(rank, world_size, device))
    ctx, device = run.ctx, run.ctx.device
    best_path, last_path, metrics_path = (
        layout.planner_dir / name for name in _OUTPUT_NAMES
    )
    stopper = EarlyStopper(patience=run.settings.early_stopping_patience)
    history: list[dict[str, float]] = []
    validation: dict[str, float] = {}
    step = 0
    epoch = 0
    for epoch in range(1, run.settings.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        nll, step = _train_epoch(run, epoch, step)
        train_elapsed = time.perf_counter() - started
        validation_started = time.perf_counter()
        validation = _evaluate(run.model, run.val_loader, ctx)
        _synchronize(device)
        validation_elapsed = time.perf_counter() - validation_started
        improved = stopper.step(validation["nll"], epoch=epoch)
        if ctx.primary:
            if improved:
                run.save(
                    best_path,
                    model_state=run.model.state_dict(),
                    epoch=epoch,
                    validation_metric=stopper.best_value,
                    force=True,
                )
            train = nll.metrics()
            row = {
                "epoch": epoch,
                "learning_rate": float(run.optimizer.param_groups[0]["lr"]),
                "train_elapsed_s": train_elapsed,
                "validation_elapsed_s": validation_elapsed,
                "train_samples_per_sec": nll.sums[0]
                / max(train_elapsed, 1e-6),
                **_prefixed("train", train),
                **_prefixed("validation", validation),
            }
            history.append(row)
            _log_epoch(run, epoch, row, train, validation, improved)
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

    final_train = _evaluate(run.model, run.train_eval_loader, ctx)
    summary: dict[str, Any] = {
        "best_epoch": stopper.best_epoch,
        "best_validation_nll": stopper.best_value,
        **_prefixed("final_train", final_train),
        **_prefixed("final_validation", validation),
    }
    if ctx.primary:
        run.save(
            last_path,
            model_state=run.model.state_dict(),
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


def train_planner(
    settings: PlannerConfig, layout: RunLayout
) -> dict[str, Any]:
    result = run_on_devices(
        _train_on_device, settings.device, settings, layout
    )
    if result is not None:
        return result
    metrics = json.loads((layout.planner_dir / "metrics.json").read_text())
    metrics.pop("epochs")
    return metrics


__all__ = [
    "train_planner",
]
