from __future__ import annotations

import contextlib
import functools
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler

from semoflow.config import (
    RealizerConfig,
    RunLayout,
)
from semoflow.poi.sids import (
    load_codebooks,
    sid_code_columns,
)
from semoflow.realizer.bank import (
    bank_split_city_ids,
    bank_split_window,
    load_bank_payload,
)
from semoflow.realizer.model import (
    MODEL_TYPE,
    PrefixTimeSuffixLatentFlowRealizer,
)
from semoflow.realizer.objective import (
    CatalogCandidateBank,
    compute_tail_whitening_stats,
    prefix_time_suffix_latent_flow_loss,
)
from semoflow.realizer.time_tokens import N_TIME_TOKENS
from semoflow.utils import (
    EarlyStopper,
    build_lr_scheduler,
    set_seed_all,
    unwrap_ddp,
    write_json,
)

logger = logging.getLogger(__name__)
_CPU_THREADS = 4


class _BankSplit(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self, payload: dict[str, Any], split: str, city_vocab: int
    ) -> None:
        offset, count = bank_split_window(payload, split)
        rows = slice(offset, offset + count)
        self.codes = payload["sid_codes"][rows]
        self.times = payload["time_tokens"][rows]
        self.valid = payload["valid_mask"][rows].bool()
        self.city = (
            bank_split_city_ids(payload, split) if city_vocab > 0 else None
        )

    def __len__(self) -> int:
        return int(self.codes.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.__getitems__(torch.tensor([index]))

    def __getitems__(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        index = torch.as_tensor(indices, dtype=torch.long)
        valid = self.valid.index_select(0, index)
        width = int(valid.sum(dim=1).max().item())
        batch = {
            "sid_codes": self.codes.index_select(0, index)[:, :width],
            "time_tokens": self.times.index_select(0, index)[:, :width],
            "valid_mask": valid[:, :width],
        }
        if self.city is not None:
            batch["city_id"] = self.city.index_select(0, index)
        return batch


class _LengthBuckets(Sampler[torch.Tensor]):
    def __init__(
        self,
        lengths: torch.Tensor,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        rank: int,
        world_size: int,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self._epoch = 0
        lengths = lengths.long().cpu()
        self._buckets = [
            torch.where(lengths == length)[0]
            for length in torch.unique(lengths, sorted=True)
        ]

    def _batches(self) -> int:
        return sum(
            (indices.numel() + self.batch_size - 1) // self.batch_size
            for indices in self._buckets
        )

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1
        batches: list[torch.Tensor] = []
        for indices in self._buckets:
            if self.shuffle:
                indices = indices[
                    torch.randperm(indices.numel(), generator=generator)
                ]
            batches.extend(indices.split(self.batch_size))
        if self.shuffle:
            order = torch.randperm(len(batches), generator=generator)
            batches = [batches[int(i)] for i in order.tolist()]
            batches = batches[
                : len(batches) // self.world_size * self.world_size
            ]
        yield from batches[self.rank :: self.world_size]

    def __len__(self) -> int:
        batches = self._batches()
        if self.shuffle:
            return batches // self.world_size
        return (batches + self.world_size - self.rank - 1) // self.world_size


def _identity(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return batch


def _loader(
    split: _BankSplit,
    config: RealizerConfig,
    *,
    shuffle: bool,
    device: torch.device,
    rank: int,
    world_size: int,
) -> DataLoader:
    sampler = _LengthBuckets(
        split.valid.sum(dim=1),
        config.batch_size,
        shuffle=shuffle,
        seed=config.seed,
        rank=rank,
        world_size=world_size,
    )
    return DataLoader(
        split,
        batch_sampler=sampler,
        collate_fn=_identity,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )


def _catalog_codes(path: Path) -> torch.Tensor:
    fields = sid_code_columns(pl.read_parquet_schema(path))
    frame = pl.read_parquet(path, columns=list(fields))
    return torch.tensor(frame.to_numpy(), dtype=torch.long).contiguous()


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    objective: Any,
    *,
    device: torch.device,
    bf16: bool,
    distributed: bool,
    max_grad_norm: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16 and device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            with autocast:
                loss = objective(
                    model,
                    batch["sid_codes"].long(),
                    batch["time_tokens"].long(),
                    batch["valid_mask"],
                    city_id=batch.get("city_id"),
                )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            events = batch["valid_mask"].sum().to(torch.float64)
            totals[0] += loss.detach().to(torch.float64) * events
            totals[1] += events
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float((totals[0] / totals[1].clamp_min(1.0)).item())


def train_prefix_time_suffix_latent_flow_realizer(
    config: RealizerConfig, layout: RunLayout
) -> dict[str, Any]:
    set_seed_all(config.seed)
    torch.set_num_threads(_CPU_THREADS)
    device = torch.device(config.device)
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    is_main = rank == 0
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    tokenizer, output_dir = layout.tokenizer, layout.realizer_dir
    codebooks = load_codebooks(tokenizer.tokenizer_codebooks)
    payload = load_bank_payload(layout.bank)
    train, val = (
        _BankSplit(payload, split, config.city_vocab)
        for split in ("train", "val")
    )
    num_codebooks = int(train.codes.shape[-1])
    latent_dim = int(codebooks.shape[-1])
    loader = functools.partial(
        _loader, config=config, device=device, rank=rank, world_size=world_size
    )
    train_loader = loader(train, shuffle=True)
    val_loader = loader(val, shuffle=False)

    tail_mean, tail_whitener = (
        stat.to(device)
        for stat in compute_tail_whitening_stats(
            train.codes,
            train.valid,
            codebooks,
            prefix_depth=config.prefix_depth,
            chunk_size=config.tail_whitening_chunk_size,
            compute_device=device,
        )
    )
    candidate_bank = CatalogCandidateBank.from_codes(
        _catalog_codes(tokenizer.placekey_lookup),
        codebooks,
        prefix_depth=config.prefix_depth,
        observed_codes=train.codes[train.valid],
        count_device=device,
        max_candidates=config.catalog_candidate_count,
    ).to(device)
    model: torch.nn.Module = PrefixTimeSuffixLatentFlowRealizer(
        max_seq_len=int(train.codes.shape[1]),
        prefix_depth=config.prefix_depth,
        num_codebooks=num_codebooks,
        codebook_size=int(codebooks.shape[1]),
        time_vocab=N_TIME_TOKENS,
        latent_dim=latent_dim,
        model_dim=config.model_dim,
        condition_depth=config.condition_depth,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        prefix_plan_codebooks=codebooks[: config.prefix_depth],
        city_vocab=config.city_vocab,
    ).to(device)
    if config.compile and device.type == "cuda":
        model = torch.compile(model, dynamic=True)
    if distributed:
        model = DDP(model, device_ids=[device.index], broadcast_buffers=False)
        set_seed_all(config.seed + rank)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    total_updates = len(train_loader) * config.epochs
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=config.lr_scheduler,
        warmup_steps=config.warmup_steps,
        total_updates=total_updates,
        base_lr=config.learning_rate,
        min_lr=config.min_lr,
    )
    if is_main:
        logger.info(
            "runtime world_size=%d per_rank_batch=%d global_batch=%d "
            "bf16=%s compile=%s workers=%d total_updates=%d",
            world_size,
            config.batch_size,
            config.batch_size * world_size,
            config.bf16,
            config.compile,
            config.num_workers,
            total_updates,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_extra = {
        "model_type": MODEL_TYPE,
        "prefix_depth": config.prefix_depth,
        "num_codebooks": num_codebooks,
        "codebook_size": int(codebooks.shape[1]),
        "time_vocab": N_TIME_TOKENS,
        "latent_dim": latent_dim,
        "tail_mean": tail_mean.cpu(),
        "tail_whitener": tail_whitener.cpu(),
        "codebooks": codebooks,
        "config": asdict(config),
    }

    run_epoch = functools.partial(
        _run_epoch,
        objective=functools.partial(
            prefix_time_suffix_latent_flow_loss,
            codebooks=codebooks.to(device),
            tail_mean=tail_mean,
            tail_whitener=tail_whitener,
            candidate_bank=candidate_bank,
            catalog_loss_weight=config.catalog_loss_weight,
            catalog_temperature=config.catalog_temperature,
            endpoint_loss_weight=config.endpoint_loss_weight,
            population_loss_weight=config.population_loss_weight,
            catalog_prior_weight=config.catalog_prior_weight,
        ),
        device=device,
        bf16=config.bf16,
        distributed=distributed,
        max_grad_norm=config.max_grad_norm,
    )
    raw_model = unwrap_ddp(model)
    stopper = EarlyStopper(patience=config.early_stopping_patience)
    history: list[dict[str, float]] = []
    stopped_early = False
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer=optimizer, scheduler=scheduler
        )
        validation_model = model.module if isinstance(model, DDP) else model
        val_loss = run_epoch(validation_model, val_loader)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
            }
        )
        improved = stopper.step(val_loss, epoch=epoch)
        if is_main:
            logger.info(
                "latent suffix-flow epoch=%d train_loss=%.6f val_loss=%.6f "
                "lr=%.6g stale_validations=%d",
                epoch,
                train_loss,
                val_loss,
                learning_rate,
                stopper.epochs_without_improvement,
            )
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                **checkpoint_extra,
            }
            torch.save(checkpoint, output_dir / "last.pt")
            if improved:
                torch.save(checkpoint, output_dir / "best.pt")
        if stopper.should_stop:
            stopped_early = True
            if is_main:
                logger.info(
                    "latent suffix-flow early_stop epoch=%d best_epoch=%d "
                    "best_val_loss=%.6f",
                    epoch,
                    stopper.best_epoch,
                    stopper.best_value,
                )
            break

    if distributed:
        dist.barrier()
    result: dict[str, Any] = {
        "model_type": MODEL_TYPE,
        "config": asdict(config),
        "train_rows": len(train),
        "val_rows": len(val),
        "num_codebooks": num_codebooks,
        "latent_dim": latent_dim,
        "parameter_count": sum(p.numel() for p in raw_model.parameters()),
        "world_size": world_size,
        "global_batch_size": config.batch_size * world_size,
        "stopped_early": stopped_early,
        "epochs_completed": len(history),
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best_value,
        "history": history,
        "bank_path": str(layout.bank),
        "codebooks_path": str(tokenizer.tokenizer_codebooks),
        "catalog_path": str(tokenizer.placekey_lookup),
    }
    if is_main:
        write_json(output_dir / "metrics.json", result)
    return result


__all__ = ["train_prefix_time_suffix_latent_flow_realizer"]
