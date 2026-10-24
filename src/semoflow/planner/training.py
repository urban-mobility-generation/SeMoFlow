from __future__ import annotations

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, replace
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

from semoflow.config.schemas.planner_config import (
    OccurrencePlannerArchitecture,
    OccurrencePlannerTrainingOptions,
    PlannerTrainingConfig,
)
from semoflow.logging.primitives import (
    configure_logging,
    get_configured_log_file,
)
from semoflow.logging.training import (
    log_epoch_summary,
    log_training_step,
)
from semoflow.planner.artifacts import (
    OccurrencePlannerDataConfig,
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
from semoflow.planner.legality import LegalPrefixIndex
from semoflow.planner.loss import occurrence_planner_loss
from semoflow.planner.model import OccurrenceARPlanner
from semoflow.planner.warm_start import (
    PlannerWarmStart,
    apply_planner_warm_start,
)
from semoflow.training.utils.distributed import find_free_port
from semoflow.training.utils.early_stopping import EarlyStopper
from semoflow.training.utils.lr_schedule import build_lr_scheduler
from semoflow.training.utils.runtime import (
    parse_device_spec,
    set_seed_all,
)
from semoflow.utils.serialization import write_json_atomic

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device_index: int
    device: torch.device
    owns_process_group: bool = False

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def _resolve_distributed_context(
    options: OccurrencePlannerTrainingOptions,
) -> _DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    primary_device, device_ids = parse_device_spec(options.device)
    if options.distributed:
        if world_size < 2:
            raise ValueError(
                "training.distributed requires torchrun with WORLD_SIZE >= 2"
            )
        if not torch.cuda.is_available():
            raise ValueError("distributed planner training requires CUDA")
        if primary_device.type != "cuda":
            raise ValueError(
                "training.distributed requires training.device cuda"
            )
        if device_ids and world_size != len(device_ids):
            raise ValueError(
                "the multi-GPU device list length must equal WORLD_SIZE"
            )
        device_count = torch.cuda.device_count()
        if device_ids:
            if not 0 <= local_rank < len(device_ids):
                raise ValueError(
                    f"LOCAL_RANK={local_rank} is outside the requested "
                    f"device list of length {len(device_ids)}"
                )
            device_index = device_ids[local_rank]
        else:
            if primary_device.index is not None:
                raise ValueError(
                    "distributed planner training with torchrun requires "
                    "training.device cuda; use training.device '[0,1,...]' "
                    "without training.distributed to select explicit GPUs"
                )
            device_index = local_rank
        if not 0 <= device_index < device_count:
            raise ValueError(
                f"device index {device_index} is outside the {device_count} "
                "visible CUDA devices"
            )
        torch.cuda.set_device(device_index)
        owns_process_group = not dist.is_initialized()
        if owns_process_group:
            dist.init_process_group(
                backend="nccl",
                device_id=torch.device("cuda", device_index),
            )
        elif dist.get_world_size() != world_size or dist.get_rank() != rank:
            raise ValueError(
                "initialized process group disagrees with RANK/WORLD_SIZE"
            )
        return _DistributedContext(
            enabled=True,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device_index=device_index,
            device=torch.device("cuda", device_index),
            owns_process_group=owns_process_group,
        )
    if world_size > 1:
        raise ValueError(
            "torchrun WORLD_SIZE > 1 requires training.distributed to avoid "
            "duplicate independent training processes"
        )
    return _DistributedContext(
        enabled=False,
        rank=0,
        world_size=1,
        local_rank=0,
        device_index=primary_device.index or 0,
        device=primary_device,
    )


def _all_reduce_sum(
    values: torch.Tensor, context: _DistributedContext
) -> None:
    if context.enabled:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)


def _broadcast_metrics(
    metrics: _NLLMetrics | None,
    context: _DistributedContext,
) -> _NLLMetrics:
    values = torch.zeros(9, dtype=torch.float64, device=context.device)
    if context.is_primary:
        if metrics is None:
            raise RuntimeError("primary rank has no metrics to broadcast")
        values[:] = torch.tensor(
            [
                metrics.total,
                metrics.length,
                metrics.gap,
                metrics.duration,
                metrics.prefix,
                metrics.total_per_event,
                metrics.gap_per_event,
                metrics.duration_per_event,
                metrics.prefix_per_event,
            ],
            dtype=torch.float64,
            device=context.device,
        )
    if context.enabled:
        dist.broadcast(values, src=0)
    return _NLLMetrics(*[float(value) for value in values.cpu()])


@dataclass(frozen=True, slots=True)
class OccurrencePlannerTrainResult:
    output_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    metrics_path: Path
    best_epoch: int
    best_validation_nll: float
    final_train_objective: float
    final_train_nll: float
    final_validation_nll: float


@dataclass(frozen=True, slots=True)
class _NLLMetrics:
    total: float
    length: float
    gap: float
    duration: float
    prefix: float
    total_per_event: float
    gap_per_event: float
    duration_per_event: float
    prefix_per_event: float

    def to_dict(self) -> dict[str, float]:
        return {
            "nll": self.total,
            "length_nll": self.length,
            "gap_nll": self.gap,
            "duration_nll": self.duration,
            "prefix_nll": self.prefix,
            "nll_per_event": self.total_per_event,
            "gap_nll_per_event": self.gap_per_event,
            "duration_nll_per_event": self.duration_per_event,
            "prefix_nll_per_event": self.prefix_per_event,
        }


@dataclass(slots=True)
class _NLLAccumulator:
    sequence_count: int = 0
    event_count: int = 0
    total_sum: float = 0.0
    length_sum: float = 0.0
    gap_sum: float = 0.0
    duration_sum: float = 0.0
    prefix_sum: float = 0.0

    def add(self, loss: Any, batch: Any) -> None:
        sequences = int(batch.lengths.shape[0])
        events = int(batch.valid_mask.sum().item())
        self.sequence_count += sequences
        self.event_count += events
        self.total_sum += float(loss.total.detach()) * sequences
        self.length_sum += float(loss.length.detach()) * sequences
        self.gap_sum += float(loss.gap.detach()) * sequences
        self.duration_sum += float(loss.duration.detach()) * sequences
        self.prefix_sum += float(loss.prefix.detach()) * sequences

    def finalize(self) -> _NLLMetrics:
        if self.sequence_count < 1 or self.event_count < 1:
            raise ValueError("cannot summarize an empty planner dataset")
        return _NLLMetrics(
            total=self.total_sum / self.sequence_count,
            length=self.length_sum / self.sequence_count,
            gap=self.gap_sum / self.sequence_count,
            duration=self.duration_sum / self.sequence_count,
            prefix=self.prefix_sum / self.sequence_count,
            total_per_event=self.total_sum / self.event_count,
            gap_per_event=self.gap_sum / self.event_count,
            duration_per_event=self.duration_sum / self.event_count,
            prefix_per_event=self.prefix_sum / self.event_count,
        )


def _prefixed_metrics(prefix: str, metrics: _NLLMetrics) -> dict[str, float]:
    return {
        f"{prefix}_{name}": value for name, value in metrics.to_dict().items()
    }


def split_tensor_dataset(
    split: OccurrencePlannerSplit,
) -> OccurrencePlannerTensorDataset:
    return OccurrencePlannerTensorDataset(
        split.prefix_codes,
        split.gap_tokens,
        split.lengths,
        event_time_minutes=split.event_time_minutes,
        context_features=split.context_features,
        city_ids=split.city_ids,
    )


def split_nll_metrics(
    model: OccurrenceARPlanner,
    split: OccurrencePlannerSplit,
    *,
    config: OccurrencePlannerConfig,
    legality: LegalPrefixIndex,
    batch_size: int,
    device: torch.device,
) -> _NLLMetrics:
    loader = build_occurrence_tensor_dataloader(
        split_tensor_dataset(split),
        config=config,
        legality=legality,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    return _loader_nll_metrics(model, loader, device=device)


def _loader_nll_metrics(
    model: OccurrenceARPlanner,
    loader: DataLoader,
    *,
    device: torch.device,
) -> _NLLMetrics:
    accumulator = _NLLAccumulator()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                output = model(batch)
                loss = occurrence_planner_loss(output, batch)
                accumulator.add(loss, batch)
    finally:
        model.train(was_training)
    return accumulator.finalize()


def _planner_config(
    data: OccurrencePlannerDataConfig,
    architecture: OccurrencePlannerArchitecture,
) -> OccurrencePlannerConfig:
    return OccurrencePlannerConfig(
        codebook_size=data.codebook_size,
        gap_vocab=data.gap_vocab,
        prefix_depth=data.prefix_depth,
        max_occurrences=data.max_occurrences,
        context_dim=data.context_dim,
        context_schema=data.context_schema,
        city_conditioning=data.city_conditioning,
        model_dim=architecture.model_dim,
        n_heads=architecture.n_heads,
        n_layers=architecture.n_layers,
        dropout=architecture.dropout,
        prefix_decoder=architecture.prefix_decoder,
        mobility_router=architecture.mobility_router,
        mobility_gate_prior=architecture.mobility_gate_prior,
        gap_codec_version=data.gap_codec_version,
        use_tokenizer_latent=architecture.use_tokenizer_latent,
        use_distance_prior=architecture.use_distance_prior,
        gap_bucket_edges_minutes=data.gap_bucket_edges_minutes,
        gap_duration_minutes=data.gap_duration_minutes,
        observation_window_minutes=data.observation_window_minutes,
    )


@dataclass(slots=True)
class _StepLogger:
    total_updates: int
    steps_per_epoch: int
    total_epochs: int
    device: torch.device
    enabled: bool
    global_step: int = 0
    _t0: float = 0.0
    _samples: int = 0

    def start_epoch(self) -> None:
        self._t0 = time.perf_counter()
        self._samples = 0

    def step(
        self, *, epoch: int, loss: Any, batch_size: int, lr: float
    ) -> None:
        self.global_step += 1
        self._samples += batch_size
        interval = max(1, min(50, self.steps_per_epoch // 5))
        step = self.global_step
        if not self.enabled or not (
            step == 1
            or step % interval == 0
            or step % self.steps_per_epoch == 0
            or step == self.total_updates
        ):
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        metrics = {
            "loss": float(loss.total.detach()),
            "prefix_loss": float(loss.prefix.detach()),
            "gap_loss": float(loss.gap.detach()),
            "length_loss": float(loss.length.detach()),
        }
        duration = float(loss.duration.detach())
        if duration != 0.0:
            metrics["duration_loss"] = duration
        log_training_step(
            logger,
            step=step,
            total_steps=self.total_updates,
            epoch=epoch,
            total_epochs=self.total_epochs,
            metrics=metrics,
            lr=lr,
            samples_per_sec=self._samples / max(now - self._t0, 1e-6),
        )
        self._t0 = now
        self._samples = 0


def _train_epoch(
    train_model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None,
    *,
    options: OccurrencePlannerTrainingOptions,
    device: torch.device,
    distributed: _DistributedContext,
    step_logger: _StepLogger,
    epoch: int,
) -> tuple[float, _NLLMetrics, int]:
    train_model.train()
    step_logger.start_epoch()
    objective_sum = 0.0
    sequence_count = 0
    nll = _NLLAccumulator()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        output = train_model(batch)
        loss = occurrence_planner_loss(
            output,
            batch,
            length_weight=options.length_weight,
            gap_weight=options.gap_weight,
            prefix_weight=options.prefix_weight,
            label_smoothing=options.label_smoothing,
            recurrence_weight=options.recurrence_weight,
        )
        with torch.no_grad():
            nll.add(occurrence_planner_loss(output, batch), batch)
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        batch_size = int(batch.lengths.shape[0])
        objective_sum += float(loss.total.detach()) * batch_size
        sequence_count += batch_size
        step_logger.step(
            epoch=epoch,
            loss=loss,
            batch_size=batch_size,
            lr=float(optimizer.param_groups[0]["lr"]),
        )

    totals = torch.tensor(
        [
            objective_sum,
            float(sequence_count),
            float(nll.sequence_count),
            float(nll.event_count),
            nll.total_sum,
            nll.length_sum,
            nll.gap_sum,
            nll.duration_sum,
            nll.prefix_sum,
        ],
        dtype=torch.float64,
        device=device,
    )
    _all_reduce_sum(totals, distributed)
    values = totals.tolist()
    reduced = _NLLAccumulator(
        sequence_count=int(values[2]),
        event_count=int(values[3]),
        total_sum=values[4],
        length_sum=values[5],
        gap_sum=values[6],
        duration_sum=values[7],
        prefix_sum=values[8],
    )
    processed = int(values[1])
    return values[0] / processed, reduced.finalize(), processed


def _primary_split_nll(
    model: OccurrenceARPlanner,
    loader: DataLoader,
    *,
    device: torch.device,
    distributed: _DistributedContext,
) -> _NLLMetrics:
    metrics = (
        _loader_nll_metrics(model, loader, device=device)
        if distributed.is_primary
        else None
    )
    if distributed.enabled:
        dist.barrier()
    return _broadcast_metrics(metrics, distributed)


_OUTPUT_NAMES = ("best.pt", "last.pt", "metrics.json", "warm_start.json")


def _check_gpu_list(device_ids: list[int]) -> None:
    if len(set(device_ids)) != len(device_ids) or any(
        device_id < 0 for device_id in device_ids
    ):
        raise ValueError("device IDs must be unique non-negative integers")
    if not torch.cuda.is_available():
        raise ValueError("a GPU device list requires CUDA")
    device_count = torch.cuda.device_count()
    if any(device_id >= device_count for device_id in device_ids):
        raise ValueError(
            f"requested device IDs {device_ids} exceed the "
            f"{device_count} visible CUDA devices"
        )


def check_planner_settings(
    settings: PlannerTrainingConfig, *, force: bool = False
) -> None:
    """Fail fast on unusable GPUs or outputs that would be overwritten."""
    _, device_ids = parse_device_spec(settings.training.device)
    if device_ids:
        _check_gpu_list(device_ids)
    output_dir = Path(settings.output_dir)
    if (
        settings.initialize_from is not None
        and Path(settings.initialize_from).resolve().parent
        == output_dir.resolve()
    ):
        raise ValueError(
            "the warm-start checkpoint must not live in the output directory"
        )
    existing = [
        output_dir / name
        for name in _OUTPUT_NAMES
        if (output_dir / name).exists()
    ]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite existing artifacts: "
            f"{', '.join(str(p) for p in existing)}"
        )


def train_planner(
    settings: PlannerTrainingConfig,
    *,
    force: bool = False,
    log_level: str = "INFO",
) -> OccurrencePlannerTrainResult:
    """Train in this process, or spawn one DDP rank per listed GPU."""
    _, device_ids = parse_device_spec(settings.training.device)
    if len(device_ids) > 1 and not settings.training.distributed:
        return train_occurrence_planner_on_devices(
            settings, device_ids, force=force, log_level=log_level
        )
    return train_occurrence_planner(settings, force=force)


def _check_dataset_supports(
    config: OccurrencePlannerConfig,
    options: OccurrencePlannerTrainingOptions,
    dataset: OccurrencePlannerDataset,
) -> None:
    if config.mobility_router != "none" and options.label_smoothing != 0.0:
        raise ValueError(
            "label_smoothing must be zero when mobility routing is enabled"
        )
    if config.use_tokenizer_latent and dataset.prefix_codebooks is None:
        raise ValueError(
            "architecture.use_tokenizer_latent requires a dataset built "
            "with --codebooks"
        )
    if config.use_distance_prior and dataset.legal_prefix_centroids is None:
        raise ValueError(
            "architecture.use_distance_prior requires a dataset built with "
            "--places coordinates"
        )


def _catalog_tensors(
    config: OccurrencePlannerConfig,
    dataset: OccurrencePlannerDataset,
) -> dict[str, torch.Tensor | None]:
    """Catalog tensors shared by the model and every checkpoint it writes."""
    return {
        "legal_prefixes": dataset.legal_prefixes,
        "catalog_city_ids": dataset.catalog_city_ids,
        "city_legal_prefix_mask": dataset.city_legal_prefix_mask,
        "prefix_codebooks": (
            dataset.prefix_codebooks if config.use_tokenizer_latent else None
        ),
        "legal_prefix_centroids": (
            dataset.legal_prefix_centroids
            if config.use_distance_prior
            else None
        ),
        "legal_prefix_centroid_available": (
            dataset.legal_prefix_centroid_available
            if config.use_distance_prior
            else None
        ),
    }


def _log_training_start(
    config: OccurrencePlannerConfig,
    options: OccurrencePlannerTrainingOptions,
    *,
    device: torch.device,
    seed: int,
    distributed: _DistributedContext,
) -> None:
    logger.info(
        "starting training  device=%s  seed=%d  epochs=%d  batch_size=%d  "
        "learning_rate=%g  model_dim=%d  n_layers=%d  prefix_decoder=%s  "
        "mobility_router=%s  lr_scheduler=%s  warmup_steps=%d  min_lr=%g  "
        "num_workers=%d  distributed=%s world_size=%d compile=%s backend=%s mode=%s "
        "fullgraph=%s float32_matmul_precision=%s",
        device,
        seed,
        options.epochs,
        options.batch_size,
        options.learning_rate,
        config.model_dim,
        config.n_layers,
        config.prefix_decoder,
        config.mobility_router,
        options.lr_scheduler,
        options.warmup_steps,
        options.min_lr,
        options.num_workers,
        distributed.enabled,
        distributed.world_size,
        options.compile_model,
        options.compile_backend,
        options.compile_mode,
        options.compile_model,
        torch.get_float32_matmul_precision(),
    )


def _build_loaders(
    dataset: OccurrencePlannerDataset,
    *,
    config: OccurrencePlannerConfig,
    legality: LegalPrefixIndex,
    options: OccurrencePlannerTrainingOptions,
    seed: int,
    distributed: _DistributedContext,
) -> tuple[DataLoader, DistributedSampler | None, DataLoader, DataLoader]:
    """Shuffled train loader (and its DDP sampler), then ordered
    validation and train evaluation loaders."""
    build = partial(
        build_occurrence_tensor_dataloader,
        config=config,
        legality=legality,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        pin_memory=distributed.device.type == "cuda",
    )
    train_ds = split_tensor_dataset(dataset.train)
    train_sampler = (
        DistributedSampler(
            train_ds,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        if distributed.enabled
        else None
    )
    train_loader = build(
        train_ds,
        shuffle=train_sampler is None,
        seed=seed,
        sampler=train_sampler,
    )
    val_loader, eval_train_loader = (
        build(split_tensor_dataset(split), shuffle=False)
        for split in (dataset.validation, dataset.train)
    )
    return train_loader, train_sampler, val_loader, eval_train_loader


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _log_epoch(
    row: dict[str, float],
    *,
    total_epochs: int,
    improved: bool,
    device: torch.device,
) -> None:
    cuda = device.type == "cuda"
    log_epoch_summary(
        logger,
        epoch=int(row["epoch"]),
        total_epochs=total_epochs,
        epoch_elapsed_s=row["train_elapsed_s"],
        samples_per_sec=row["train_samples_per_sec"],
        gpu_allocated_gb=(
            torch.cuda.memory_allocated(device) / 1e9 if cuda else 0.0
        ),
        gpu_peak_gb=(
            torch.cuda.max_memory_allocated(device) / 1e9 if cuda else 0.0
        ),
    )
    logger.info(
        "epoch %d/%d  lr=%.6g  train_objective=%.4f  train_nll=%.4f  "
        "validation_nll=%.4f  validation_time=%.2fs  "
        "val[length=%.4f gap=%.4f duration=%.4f prefix=%.4f]%s",
        row["epoch"],
        total_epochs,
        row["learning_rate"],
        row["train_objective"],
        row["train_nll"],
        row["validation_nll"],
        row["validation_elapsed_s"],
        row["validation_length_nll"],
        row["validation_gap_nll"],
        row["validation_duration_nll"],
        row["validation_prefix_nll"],
        "  (best)" if improved else "",
    )


def train_occurrence_planner(
    settings: PlannerTrainingConfig, *, force: bool = False
) -> OccurrencePlannerTrainResult:
    options, seed = settings.training, settings.seed
    warm_start = (
        None
        if settings.initialize_from is None
        else PlannerWarmStart(checkpoint=settings.initialize_from)
    )
    dataset = load_occurrence_planner_dataset(settings.dataset_path)
    if int(os.environ.get("RANK", "0")) == 0:
        logger.info(
            "loaded planner dataset  path=%s  train=%d  validation=%d  "
            "codebook_size=%d  gap_vocab=%d  temporal_state=%s",
            settings.dataset_path,
            dataset.train.size,
            dataset.validation.size,
            dataset.config.codebook_size,
            dataset.config.gap_vocab,
            dataset.config.temporal_state_enabled,
        )
    config = _planner_config(dataset.config, settings.architecture)
    _check_dataset_supports(config, options, dataset)
    check_planner_settings(settings, force=force)
    output_dir = Path(settings.output_dir)
    best_path, last_path, metrics_path, warm_start_path = (
        output_dir / name for name in _OUTPUT_NAMES
    )

    distributed = _resolve_distributed_context(options)
    device = distributed.device
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    set_seed_all(seed)
    legality = LegalPrefixIndex(
        dataset.legal_prefixes,
        codebook_size=config.codebook_size,
        prefix_depth=config.prefix_depth,
    )
    catalog = _catalog_tensors(config, dataset)
    model = OccurrenceARPlanner(config, **catalog).to(device)
    warm_start_report = (
        None
        if warm_start is None
        else apply_planner_warm_start(model, warm_start)
    )
    if warm_start_report is not None and distributed.is_primary:
        logger.info(
            "warm started from %s  source_cities=%s  inherited_cities=%s  "
            "zero_initialized_cities=%s  source_prefixes=%d  "
            "target_prefixes=%d  catalog_changed=%s",
            warm_start_report.checkpoint,
            list(warm_start_report.source_city_ids),
            list(warm_start_report.inherited_city_ids),
            list(warm_start_report.zero_initialized_city_ids),
            warm_start_report.source_legal_prefix_count,
            warm_start_report.target_legal_prefix_count,
            warm_start_report.catalog_changed,
        )
        write_json_atomic(
            warm_start_path, warm_start_report.to_dict(), force=force
        )
    if options.compile_model:
        model.compile(
            backend=options.compile_backend,
            mode=options.compile_mode,
            fullgraph=True,
        )
    train_model: nn.Module = model
    if distributed.enabled:
        train_model = DistributedDataParallel(
            model,
            device_ids=[distributed.device_index],
            output_device=distributed.device_index,
            broadcast_buffers=False,
        )
    set_seed_all(seed + distributed.rank)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    if distributed.is_primary:
        _log_training_start(
            config, options, device=device, seed=seed, distributed=distributed
        )

    train_loader, train_sampler, val_loader, eval_train_loader = (
        _build_loaders(
            dataset,
            config=config,
            legality=legality,
            options=options,
            seed=seed,
            distributed=distributed,
        )
    )
    steps_per_epoch = len(train_loader)
    total_updates = steps_per_epoch * options.epochs
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=options.lr_scheduler,
        warmup_steps=options.warmup_steps,
        total_updates=total_updates,
        base_lr=options.learning_rate,
        min_lr=options.min_lr,
    )
    step_logger = _StepLogger(
        total_updates=total_updates,
        steps_per_epoch=steps_per_epoch,
        total_epochs=options.epochs,
        device=device,
        enabled=distributed.is_primary,
    )
    save_checkpoint = partial(
        save_occurrence_planner_checkpoint,
        config=config,
        seed=seed,
        options=options,
        **catalog,
    )

    stopper = EarlyStopper(patience=options.early_stopping_patience or None)
    history: list[dict[str, float]] = []
    train_objective = float("nan")
    validation_metrics: _NLLMetrics | None = None
    for epoch in range(1, options.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start_time = time.perf_counter()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_objective, train_nll_metrics, processed = _train_epoch(
            train_model,
            train_loader,
            optimizer,
            scheduler,
            options=options,
            device=device,
            distributed=distributed,
            step_logger=step_logger,
            epoch=epoch,
        )
        _synchronize(device)
        train_elapsed = time.perf_counter() - epoch_start_time

        validation_start = time.perf_counter()
        validation_metrics = _primary_split_nll(
            model, val_loader, device=device, distributed=distributed
        )
        _synchronize(device)
        validation_elapsed = time.perf_counter() - validation_start
        decision = stopper.step(validation_metrics.total, epoch=epoch)
        if decision.improved and distributed.is_primary:
            save_checkpoint(
                best_path,
                model_state=model.state_dict(),
                epoch=epoch,
                validation_metric=stopper.best_value,
                force=True,
            )
        if distributed.is_primary:
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train_objective": train_objective,
                    "train_elapsed_s": train_elapsed,
                    "validation_elapsed_s": validation_elapsed,
                    "train_samples_per_sec": processed
                    / max(train_elapsed, 1e-6),
                    **_prefixed_metrics("train", train_nll_metrics),
                    **_prefixed_metrics("validation", validation_metrics),
                }
            )
            _log_epoch(
                history[-1],
                total_epochs=options.epochs,
                improved=decision.improved,
                device=device,
            )
        if decision.should_stop:
            if distributed.is_primary:
                logger.info(
                    "early stopping after epoch %d: validation NLL has not "
                    "improved for %d epochs (best epoch %d, %.4f)",
                    epoch,
                    stopper.epochs_without_improvement,
                    stopper.best_epoch,
                    stopper.best_value,
                )
            break

    best_epoch = stopper.best_epoch
    best_validation_nll = stopper.best_value
    if validation_metrics is None or best_epoch < 1:
        raise RuntimeError("training completed without recording an epoch")

    final_train_metrics = _primary_split_nll(
        model, eval_train_loader, device=device, distributed=distributed
    )
    if distributed.is_primary:
        save_checkpoint(
            last_path,
            model_state=model.state_dict(),
            epoch=epoch,
            validation_metric=validation_metrics.total,
            force=force,
        )
        _write_metrics_json(
            metrics_path,
            best_epoch=best_epoch,
            best_validation_nll=best_validation_nll,
            final_train_objective=train_objective,
            final_train_metrics=final_train_metrics,
            final_validation_metrics=validation_metrics,
            history=history,
            warm_start=(
                None
                if warm_start_report is None
                else warm_start_report.to_dict()
            ),
            force=force,
        )
        logger.info(
            "training complete  best_epoch=%d  best_validation_nll=%.4f  "
            "final_train_nll=%.4f  best_checkpoint=%s  last_checkpoint=%s  metrics=%s",
            best_epoch,
            best_validation_nll,
            final_train_metrics.total,
            best_path,
            last_path,
            metrics_path,
        )
    if distributed.enabled:
        dist.barrier()
        if distributed.owns_process_group:
            dist.destroy_process_group()

    return OccurrencePlannerTrainResult(
        output_dir=output_dir,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        metrics_path=metrics_path,
        best_epoch=best_epoch,
        best_validation_nll=best_validation_nll,
        final_train_objective=train_objective,
        final_train_nll=final_train_metrics.total,
        final_validation_nll=validation_metrics.total,
    )


def _train_occurrence_planner_spawn_worker(
    rank: int,
    device_ids: list[int],
    port: int,
    settings: PlannerTrainingConfig,
    log_level: str,
    log_file: Path | None,
    force: bool,
) -> None:
    warnings.filterwarnings(
        "ignore",
        message=(
            r"`torch\._prims_common\.check` is deprecated and will be "
            r"removed in the future\..*"
        ),
        category=FutureWarning,
        module=r"torch\._inductor\.lowering",
    )
    if rank == 0:
        logging.disable(logging.NOTSET)
        configure_logging(level=log_level, log_file=log_file)
    else:
        logging.disable(logging.CRITICAL)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(len(device_ids))
    os.environ["LOCAL_RANK"] = str(rank)
    device_spec = f"[{','.join(str(device_id) for device_id in device_ids)}]"
    training = replace(settings.training, device=device_spec, distributed=True)
    train_occurrence_planner(replace(settings, training=training), force=force)


def train_occurrence_planner_on_devices(
    settings: PlannerTrainingConfig,
    device_ids: list[int],
    *,
    force: bool = False,
    log_level: str = "INFO",
) -> OccurrencePlannerTrainResult:
    if len(device_ids) < 2:
        raise ValueError(
            "multi-GPU planner training requires at least two GPUs"
        )
    _check_gpu_list(device_ids)

    port = find_free_port()
    mp.spawn(
        _train_occurrence_planner_spawn_worker,
        args=(
            device_ids,
            port,
            settings,
            log_level,
            get_configured_log_file(),
            force,
        ),
        nprocs=len(device_ids),
        join=True,
    )

    output_dir = Path(settings.output_dir)
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return OccurrencePlannerTrainResult(
        output_dir=output_dir,
        best_checkpoint=output_dir / "best.pt",
        last_checkpoint=output_dir / "last.pt",
        metrics_path=metrics_path,
        best_epoch=int(metrics["best_epoch"]),
        best_validation_nll=float(metrics["best_validation_nll"]),
        final_train_objective=float(metrics["final_train_objective"]),
        final_train_nll=float(metrics["final_train_nll"]),
        final_validation_nll=float(metrics["final_validation_nll"]),
    )


def _write_metrics_json(
    path: Path,
    *,
    best_epoch: int,
    best_validation_nll: float,
    final_train_objective: float,
    final_train_metrics: _NLLMetrics,
    final_validation_metrics: _NLLMetrics,
    history: list[dict[str, float]],
    warm_start: dict[str, object] | None,
    force: bool,
) -> None:
    payload: dict[str, object] = {
        "best_epoch": best_epoch,
        "best_validation_nll": best_validation_nll,
        "final_train_objective": final_train_objective,
        **_prefixed_metrics("final_train", final_train_metrics),
        **_prefixed_metrics("final_validation", final_validation_metrics),
        "epochs": history,
    }
    if warm_start is not None:
        payload["warm_start"] = warm_start
    write_json_atomic(path, payload, force=force)


__all__ = [
    "OccurrencePlannerTrainResult",
    "check_planner_settings",
    "train_occurrence_planner",
    "train_occurrence_planner_on_devices",
    "train_planner",
]
