from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import polars as pl
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler

from semoflow.config.schemas.realizer_config import (
    PrefixTimeSuffixLatentFlowRealizerConfig,
)
from semoflow.realizer.bank import (
    bank_split_city_ids,
    load_bank_payload,
    resolve_bank_split_window,
)
from semoflow.realizer.data import (
    LengthBucketBatchSampler,
    ProposalCorruptionIndex,
    TimeAwareSIDDataset,
    build_proposal_corruption_index,
    collate_time_aware_rows,
    corrupt_proposal_prefixes_hierarchical,
)
from semoflow.realizer.model import (
    PrefixTimeSuffixLatentFlowRealizer,
    model_type,
)
from semoflow.realizer.objective import (
    CatalogCandidateBank,
    CatalogTailCandidates,
    LatentSuffixFlowObjectiveOutput,
    compute_factorized_tail_stats,
    compute_tail_stats,
    compute_tail_whitening_stats,
    prefix_time_suffix_latent_flow_loss,
)
from semoflow.realizer.spatial import (
    prefix_centroid_table,
    prefix_hop_distance_km,
)
from semoflow.tokenization.decoding.codebooks import (
    MAX_STANDALONE_CODEBOOK_ARTIFACT_BYTES,
    SIDCodebookBundle,
    codebook_bundle_metadata,
    extract_codebook_bundle,
)
from semoflow.tokenization.token_space import sid_code_columns, sid_code_names
from semoflow.training.utils.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from semoflow.training.utils.distributed import unwrap_ddp
from semoflow.training.utils.early_stopping import EarlyStopper
from semoflow.training.utils.lr_schedule import build_lr_scheduler
from semoflow.training.utils.mixed_precision import build_amp_context
from semoflow.training.utils.runtime import set_seed_all
from semoflow.utils.serialization import write_json

logger = logging.getLogger(__name__)


def _split_tensors(
    payload: dict[str, Any],
    path: Path,
    split: str,
    max_samples: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offset, cap = resolve_bank_split_window(payload, split, path=path)
    count = cap if max_samples is None else min(cap, max_samples)
    if count <= 0:
        raise ValueError(f"realizer bank split {split!r} is empty")
    end = offset + count
    return (
        payload["sid_codes"][offset:end],
        payload["time_tokens"][offset:end],
        payload["valid_mask"][offset:end],
    )


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _load_catalog_codes(path: Path, *, n_codebooks: int) -> torch.Tensor:
    fields = sid_code_columns(pl.read_parquet_schema(path))
    expected = sid_code_names(n_codebooks)
    if fields != expected:
        raise ValueError(
            "catalog SID columns must match the codebook bundle: "
            f"expected {expected}, got {fields}"
        )
    frame = pl.read_parquet(path, columns=list(fields))
    if frame.height == 0:
        raise ValueError(f"catalog is empty: {path}")
    return torch.tensor(frame.to_numpy(), dtype=torch.long).contiguous()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


class _CitySIDDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self,
        codes: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
        city: torch.Tensor | None,
        hop_distance: torch.Tensor | None = None,
    ) -> None:
        self._base = TimeAwareSIDDataset(codes, times, valid)
        if city is not None and int(city.shape[0]) != len(self._base):
            raise ValueError("city ids must align with the SID rows")
        if hop_distance is not None and hop_distance.shape != codes.shape[:2]:
            raise ValueError("hop_distance must align with the SID rows")
        self._city = city
        self._hop_distance = hop_distance

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        row = self._base[index]
        extra: tuple[torch.Tensor, ...] = ()
        if self._city is not None:
            extra = (*extra, self._city[index])
        if self._hop_distance is not None:
            extra = (*extra, self._hop_distance[index])
        return (*row, *extra)

    def __getitems__(self, indices: list[int] | torch.Tensor):
        index = torch.as_tensor(indices, dtype=torch.long)
        valid = self._base.valid_mask.index_select(0, index)
        max_len = max(1, int(valid.sum(dim=1).max().item()))
        batch = {
            "sid_codes": self._base.sid_codes.index_select(0, index)[
                :, :max_len
            ],
            "time_tokens": self._base.time_tokens.index_select(0, index)[
                :, :max_len
            ],
            "valid_mask": valid[:, :max_len],
        }
        if self._city is not None:
            batch["city_id"] = self._city.index_select(0, index)
        if self._hop_distance is not None:
            batch["hop_distance_km"] = self._hop_distance.index_select(
                0, index
            )[:, :max_len]
        return batch


class _DistributedLengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: torch.Tensor,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        rank: int,
        world_size: int,
        drop_uneven_batches: bool,
    ) -> None:
        if world_size < 2 or not 0 <= rank < world_size:
            raise ValueError("rank and world_size do not describe a DDP job")
        self._base = LengthBucketBatchSampler(
            lengths,
            batch_size,
            shuffle=shuffle,
            seed=seed,
            tensor_batches=True,
        )
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.drop_uneven_batches = bool(drop_uneven_batches)

    def __iter__(self):
        batches = list(self._base)
        if self.drop_uneven_batches:
            batches = batches[
                : len(batches) // self.world_size * self.world_size
            ]
        yield from batches[self.rank :: self.world_size]

    def __len__(self) -> int:
        batches = len(self._base)
        if self.drop_uneven_batches:
            return batches // self.world_size
        return (batches + self.world_size - self.rank - 1) // self.world_size


def _should_validate(epoch: int, *, epochs: int, val_every: int) -> bool:
    if epoch < 1 or epochs < 1 or epoch > epochs or val_every < 1:
        raise ValueError("validation cadence arguments are invalid")
    return epoch % val_every == 0 or epoch == epochs


def _distributed_context() -> tuple[bool, int, int]:
    if not dist.is_available() or not dist.is_initialized():
        return False, 0, 1
    return True, dist.get_rank(), dist.get_world_size()


def _configure_cuda_math(*, device: torch.device, tf32: bool) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def _collate_city_rows(
    rows: list[tuple[torch.Tensor, ...]] | dict[str, torch.Tensor],
    *,
    has_city: bool = False,
    has_hop_distance: bool = False,
) -> dict[str, torch.Tensor]:
    if isinstance(rows, dict):
        return rows
    batch = collate_time_aware_rows([row[:3] for row in rows])
    cursor = 3
    if has_city:
        batch["city_id"] = torch.stack([row[cursor] for row in rows]).long()
        cursor += 1
    if has_hop_distance:
        max_len = batch["sid_codes"].shape[1]
        batch["hop_distance_km"] = torch.stack(
            [row[cursor][:max_len] for row in rows]
        ).float()
        cursor += 1
    return batch


def _bank_city_ids(
    payload: dict[str, Any],
    bank_path: Path,
    split: str,
    valid: torch.BoolTensor,
    city_vocab: int,
) -> torch.LongTensor:
    city = bank_split_city_ids(
        payload, bank_path, split=split, valid_mask=valid
    )
    city = city.long().cpu()
    if int(city.min()) < 0 or int(city.max()) >= city_vocab:
        raise ValueError("bank city ids are outside city_vocab")
    return city


def _raise_nonfinite_loss(
    output: LatentSuffixFlowObjectiveOutput,
    *,
    phase: str,
    batch_index: int,
    codes: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    components = {
        "loss": output.loss,
        "flow": output.flow_loss,
        "endpoint": output.endpoint_loss,
        "sum_consistency": output.sum_consistency_loss,
        "catalog": output.catalog_loss,
        "population": output.population_loss,
    }
    logger.debug(
        "non-finite latent-flow loss phase=%s batch=%d rows=%d "
        "active_events=%d max_events_per_row=%d components=%s",
        phase,
        batch_index,
        codes.size(0),
        int(valid.sum()),
        int(valid.sum(dim=1).max()),
        {
            name: float(value.detach().float())
            for name, value in components.items()
        },
    )
    for name, value in {
        "timesteps": output.timesteps,
        "clean_tail": output.clean_tail,
        "noisy_tail": output.noisy_tail,
        "target_velocity": output.target_velocity,
        "predicted_velocity": output.predicted_velocity,
        "x0_pred": output.x0_pred,
    }.items():
        detached = value.detach()
        finite = torch.isfinite(detached)
        finite_values = detached[finite].float()
        logger.debug(
            "non-finite latent-flow tensor=%s shape=%s dtype=%s "
            "finite=%d nan=%d posinf=%d neginf=%d "
            "finite_min=%s finite_max=%s",
            name,
            tuple(detached.shape),
            detached.dtype,
            int(finite.sum()),
            int(torch.isnan(detached).sum()),
            int(torch.isposinf(detached).sum()),
            int(torch.isneginf(detached).sum()),
            float(finite_values.min()) if finite_values.numel() else None,
            float(finite_values.max()) if finite_values.numel() else None,
        )
    raise FloatingPointError(
        "non-finite latent-flow loss; see DEBUG diagnostics"
    )


def _raise_nonfinite_gradients(
    model: torch.nn.Module, *, batch_index: int, grad_norm: torch.Tensor
) -> None:
    logger.debug(
        "non-finite latent-flow gradient phase=train batch=%d grad_norm=%s",
        batch_index,
        float(grad_norm),
    )
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None or bool(torch.isfinite(gradient).all()):
            continue
        logger.debug(
            "non-finite gradient parameter=%s shape=%s nan=%d posinf=%d "
            "neginf=%d",
            name,
            tuple(gradient.shape),
            int(torch.isnan(gradient).sum()),
            int(torch.isposinf(gradient).sum()),
            int(torch.isneginf(gradient).sum()),
        )
    raise FloatingPointError(
        "non-finite latent-flow gradient; see DEBUG diagnostics"
    )


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    objective: Callable[..., LatentSuffixFlowObjectiveOutput],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    *,
    device: torch.device,
    max_grad_norm: float,
    autocast: AbstractContextManager[Any],
    scaler: torch.amp.GradScaler | None,
    reduce_across_ranks: bool,
    corruption_index: ProposalCorruptionIndex | None,
    proposal_corruption_rate: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    debug = logger.isEnabledFor(logging.DEBUG)
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_events = torch.zeros((), dtype=torch.float64, device=device)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader, start=1):
            codes = batch["sid_codes"].to(
                device, dtype=torch.long, non_blocking=True
            )
            times = batch["time_tokens"].to(
                device, dtype=torch.long, non_blocking=True
            )
            valid = batch["valid_mask"].to(device, non_blocking=True).bool()
            city = batch.get("city_id")
            if city is not None:
                city = city.to(device, non_blocking=True)
            hop = batch.get("hop_distance_km")
            if hop is not None:
                hop = hop.to(device, non_blocking=True)
            proposal_prefix = None
            if corruption_index is not None:
                proposal_prefix, _, _ = corrupt_proposal_prefixes_hierarchical(
                    codes,
                    valid,
                    corruption_index,
                    corruption_rate=proposal_corruption_rate,
                )
            with autocast:
                output = objective(
                    model,
                    codes,
                    times,
                    valid,
                    proposal_prefix_codes=proposal_prefix,
                    city_id=city,
                    hop_distance_km=hop,
                )
            if debug and not bool(torch.isfinite(output.loss)):
                _raise_nonfinite_loss(
                    output,
                    phase="train" if training else "validation",
                    batch_index=batch_index,
                    codes=codes,
                    valid=valid,
                )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(output.loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    output.loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                if debug and not bool(torch.isfinite(grad_norm)):
                    _raise_nonfinite_gradients(
                        model, batch_index=batch_index, grad_norm=grad_norm
                    )
                if scaler is not None:
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_stepped = scaler.get_scale() >= previous_scale
                else:
                    optimizer.step()
                    optimizer_stepped = True
                if scheduler is not None and optimizer_stepped:
                    scheduler.step()
            events = valid.sum().to(dtype=torch.float64)
            total_loss += output.loss.detach().to(dtype=torch.float64) * events
            total_events += events
    totals = torch.stack((total_loss, total_events))
    if reduce_across_ranks:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float((totals[0] / totals[1].clamp_min(1.0)).item())


@dataclass(frozen=True, slots=True)
class BankSplit:
    codes: torch.Tensor
    times: torch.Tensor
    valid: torch.Tensor
    city: torch.Tensor | None = None
    hop_distance: torch.Tensor | None = None

    def check_ranges(self, *, codebook_size: int, time_vocab: int) -> None:
        if self.codes.numel() and (
            self.codes.min() < 0 or self.codes.max() >= codebook_size
        ):
            raise ValueError("SID codes exceed codebook_size")
        active_times = self.times[self.valid]
        if active_times.numel() and (
            active_times.min() < 0 or active_times.max() >= time_vocab
        ):
            raise ValueError("time tokens exceed time_vocab")

    def dataset(self) -> _CitySIDDataset:
        return _CitySIDDataset(
            self.codes, self.times, self.valid, self.city, self.hop_distance
        )


def load_bank_splits(
    config: PrefixTimeSuffixLatentFlowRealizerConfig,
    *,
    codebook_size: int,
) -> tuple[BankSplit, BankSplit]:
    payload = load_bank_payload(config.bank_path)
    limits = {"train": config.train_max_samples, "val": config.val_max_samples}
    raw = {
        split: _split_tensors(payload, config.bank_path, split, limit)
        for split, limit in limits.items()
    }
    for split, (codes, times, valid) in raw.items():
        try:
            BankSplit(codes, times, valid).check_ranges(
                codebook_size=codebook_size,
                time_vocab=config.time_vocab,
            )
        except ValueError as error:
            name = "training" if split == "train" else "validation"
            raise ValueError(f"{name} {error}") from None

    centroids = None
    if config.spatial_conditioning:
        assert config.places_path is not None
        assert config.catalog_path is not None
        centroids = prefix_centroid_table(
            config.catalog_path,
            config.places_path,
            raw["train"][0].shape[-1],
            config.prefix_depth,
        )
    splits = []
    for split, (codes, times, valid) in raw.items():
        city = (
            _bank_city_ids(
                payload, config.bank_path, split, valid, config.city_vocab
            )
            if config.city_vocab > 0
            else None
        )
        hop = (
            prefix_hop_distance_km(
                codes[..., : config.prefix_depth], valid, centroids
            )
            if centroids is not None
            else None
        )
        splits.append(BankSplit(codes, times, valid, city, hop))
    return splits[0], splits[1]


def _load_initialization(
    config: PrefixTimeSuffixLatentFlowRealizerConfig, codebooks: torch.Tensor
) -> dict[str, Any] | None:
    if config.initialize_from is None:
        return None
    initialization = torch.load(
        config.initialize_from, map_location="cpu", weights_only=False
    )
    source_codebooks = initialization.get("codebooks")
    if not isinstance(source_codebooks, torch.Tensor) or not torch.equal(
        source_codebooks.float(), codebooks
    ):
        raise ValueError(
            "warm-start realizer checkpoint uses different codebooks"
        )
    if initialization.get("tail_normalization") != config.tail_normalization:
        raise ValueError(
            "warm-start realizer tail normalization does not match config"
        )
    if initialization.get("tail_representation") != config.tail_representation:
        raise ValueError(
            "warm-start realizer tail representation does not match config"
        )
    source_scope = initialization.get("latent_target_scope", "suffix")
    if source_scope != config.latent_target_scope:
        raise ValueError(
            "warm-start realizer latent target scope does not match config"
        )
    return initialization


@dataclass(frozen=True, slots=True)
class _TailStats:
    mean: torch.Tensor
    std: torch.Tensor
    whitener: torch.Tensor | None
    dewhitener: torch.Tensor | None

    def to(self, device: torch.device) -> _TailStats:
        return _TailStats(
            self.mean.to(device),
            self.std.to(device),
            None if self.whitener is None else self.whitener.to(device),
            None if self.dewhitener is None else self.dewhitener.to(device),
        )

    def cpu_payload(self) -> dict[str, torch.Tensor | None]:
        return {
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "whitener": (
                None if self.whitener is None else self.whitener.detach().cpu()
            ),
            "dewhitener": (
                None
                if self.dewhitener is None
                else self.dewhitener.detach().cpu()
            ),
        }


def _tail_stats(
    config: PrefixTimeSuffixLatentFlowRealizerConfig,
    initialization: dict[str, Any] | None,
    train: BankSplit,
    codebooks: torch.Tensor,
    *,
    target_level_start: int,
    device: torch.device,
) -> _TailStats:
    if initialization is not None:
        whitener = initialization.get("tail_whitener")
        dewhitener = initialization.get("tail_dewhitener")
        return _TailStats(
            initialization["tail_mean"].float(),
            initialization["tail_std"].float(),
            whitener.float() if isinstance(whitener, torch.Tensor) else None,
            (
                dewhitener.float()
                if isinstance(dewhitener, torch.Tensor)
                else None
            ),
        )
    valid = train.valid.bool()
    if config.tail_normalization == "whitened":
        mean, whitener, dewhitener = compute_tail_whitening_stats(
            train.codes,
            valid,
            codebooks,
            prefix_depth=target_level_start,
            shrinkage=config.tail_whitening_shrinkage,
            chunk_size=config.tail_whitening_chunk_size,
            compute_device=device,
        )
        return _TailStats(mean, torch.ones_like(mean), whitener, dewhitener)
    if config.tail_representation == "factorized":
        mean, std = compute_factorized_tail_stats(
            train.codes, valid, codebooks, prefix_depth=target_level_start
        )
    else:
        mean, std = compute_tail_stats(
            train.codes, valid, codebooks, prefix_depth=target_level_start
        )
    return _TailStats(mean, std, None, None)


def _build_loaders(
    config: PrefixTimeSuffixLatentFlowRealizerConfig,
    train: BankSplit,
    val: BankSplit,
    *,
    device: torch.device,
    distributed: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DataLoader]:
    loader_options: dict[str, Any] = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        loader_options["prefetch_factor"] = config.prefetch_factor
    loaders = []
    for split, shuffle in ((train, True), (val, False)):
        batch_sampler: Sampler[list[int]]
        if distributed:
            batch_sampler = _DistributedLengthBucketBatchSampler(
                split.valid.sum(dim=1),
                config.batch_size,
                shuffle=shuffle,
                seed=config.seed,
                rank=rank,
                world_size=world_size,
                drop_uneven_batches=shuffle,
            )
        else:
            batch_sampler = LengthBucketBatchSampler(
                split.valid.sum(dim=1),
                config.batch_size,
                shuffle=shuffle,
                seed=config.seed,
                tensor_batches=True,
            )
        loaders.append(
            DataLoader(
                split.dataset(),
                batch_sampler=batch_sampler,
                collate_fn=functools.partial(
                    _collate_city_rows,
                    has_city=split.city is not None,
                    has_hop_distance=split.hop_distance is not None,
                ),
                **loader_options,
            )
        )
    return loaders[0], loaders[1]


def _catalog_objectives(
    config: PrefixTimeSuffixLatentFlowRealizerConfig,
    train: BankSplit,
    codebooks: torch.Tensor,
    *,
    target_level_start: int,
    device: torch.device,
) -> tuple[
    CatalogTailCandidates | None,
    CatalogCandidateBank | None,
    ProposalCorruptionIndex | None,
]:
    if config.catalog_path is None:
        return None, None, None
    catalog_codes = _load_catalog_codes(
        config.catalog_path, n_codebooks=int(codebooks.shape[0])
    )
    corruption_index = (
        build_proposal_corruption_index(
            catalog_codes, codebook_size=int(codebooks.shape[1])
        ).to(device)
        if config.proposal_corruption_rate > 0.0
        else None
    )
    catalog_candidates = (
        CatalogTailCandidates.from_codes(
            catalog_codes, codebooks, prefix_depth=target_level_start
        ).to(device)
        if config.catalog_loss_weight > 0.0
        and config.catalog_decoder_mode == "nearest"
        else None
    )
    candidate_bank = (
        CatalogCandidateBank.from_codes(
            catalog_codes,
            codebooks,
            prefix_depth=target_level_start,
            observed_codes=train.codes,
            observed_valid_mask=train.valid,
            count_device=device,
            max_candidates=config.catalog_candidate_count,
        ).to(device)
        if config.catalog_decoder_mode == "calibrated"
        else None
    )
    return catalog_candidates, candidate_bank, corruption_index


def _load_codebooks(
    config: PrefixTimeSuffixLatentFlowRealizerConfig,
) -> SIDCodebookBundle:
    bundle = extract_codebook_bundle(
        config.codebooks_path,
        max_artifact_bytes=MAX_STANDALONE_CODEBOOK_ARTIFACT_BYTES,
    )
    if (
        config.codebook_size is not None
        and bundle.codebook_size != config.codebook_size
    ):
        raise ValueError(
            "codebook artifact size does not match config: "
            f"{bundle.codebook_size} != {config.codebook_size}"
        )
    if (
        config.latent_dim is not None
        and config.latent_dim != bundle.embedding_dim
    ):
        raise ValueError(
            "latent_dim does not match the codebook embedding dim"
        )
    return bundle


def _build_model(
    config: PrefixTimeSuffixLatentFlowRealizerConfig,
    codebooks: torch.Tensor,
    *,
    max_seq_len: int,
) -> PrefixTimeSuffixLatentFlowRealizer:
    return PrefixTimeSuffixLatentFlowRealizer(
        max_seq_len=max_seq_len,
        prefix_depth=config.prefix_depth,
        num_codebooks=int(codebooks.shape[0]),
        codebook_size=int(codebooks.shape[1]),
        time_vocab=config.time_vocab,
        latent_dim=int(codebooks.shape[-1]),
        model_dim=config.model_dim,
        condition_depth=config.condition_depth,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        prefix_embedding_mode=config.prefix_embedding_mode,
        prefix_plan_codebooks=(
            codebooks[: config.prefix_depth]
            if config.prefix_embedding_mode == "hybrid_frozen_codebook"
            else None
        ),
        tail_representation=config.tail_representation,
        latent_target_scope=config.latent_target_scope,
        catalog_decoder_mode=config.catalog_decoder_mode,
        city_vocab=config.city_vocab,
        spatial_conditioning=config.spatial_conditioning,
        spatial_num_basis=config.spatial_num_basis,
        spatial_max_km=config.spatial_max_km,
    )


def _log_runtime(
    config: PrefixTimeSuffixLatentFlowRealizerConfig,
    *,
    world_size: int,
    gradient_scaler: bool,
    fused_optimizer: bool,
    total_updates: int,
) -> None:
    logger.info(
        "runtime world_size=%d per_rank_batch=%d global_batch=%d "
        "mixed_precision=%s grad_scaler=%s tf32=%s fused_optimizer=%s "
        "val_every=%d workers=%d prefetch_factor=%d cpu_threads=%d",
        world_size,
        config.batch_size,
        config.batch_size * world_size,
        config.mixed_precision,
        gradient_scaler,
        config.tf32,
        fused_optimizer,
        config.val_every,
        config.num_workers,
        config.prefetch_factor,
        config.cpu_threads,
    )
    logger.info(
        "optimizer lr=%.6g scheduler=%s warmup_steps=%d "
        "total_updates=%d min_lr=%.6g early_stopping_patience=%s "
        "early_stopping_min_delta=%.6g",
        config.learning_rate,
        config.lr_scheduler,
        config.warmup_steps,
        total_updates,
        config.min_lr,
        config.early_stopping_patience,
        config.early_stopping_min_delta,
    )


def _optional_str(value: Path | None) -> str | None:
    return None if value is None else str(value)


def _tail_stats_record(
    config: PrefixTimeSuffixLatentFlowRealizerConfig,
    tail: dict[str, torch.Tensor | None],
    codebook_record: dict[str, Any],
    *,
    latent_dim: int,
    warm_started: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "prefix_depth": config.prefix_depth,
        "latent_dim": latent_dim,
        **tail,
        "normalization": config.tail_normalization,
        **codebook_record,
        "tail_representation": config.tail_representation,
        "latent_target_scope": config.latent_target_scope,
        "source": (
            str(config.initialize_from) if warm_started else "train_split_only"
        ),
        "statistics_source": (
            "initialization_checkpoint"
            if warm_started
            else "adaptation_train_split"
        ),
    }


def _checkpoint_extra(
    config: PrefixTimeSuffixLatentFlowRealizerConfig,
    tail: dict[str, torch.Tensor | None],
    codebook_record: dict[str, Any],
    *,
    num_codebooks: int,
    latent_dim: int,
) -> dict[str, Any]:
    """Everything a checkpoint needs besides weights to rebuild the model."""
    return {
        "model_type": model_type(config.latent_target_scope),
        "prefix_depth": config.prefix_depth,
        "num_codebooks": num_codebooks,
        "codebook_size": config.codebook_size,
        "time_vocab": config.time_vocab,
        "latent_dim": latent_dim,
        "tail_mean": tail["mean"],
        "tail_std": tail["std"],
        "tail_normalization": config.tail_normalization,
        "tail_whitener": tail["whitener"],
        "tail_dewhitener": tail["dewhitener"],
        **codebook_record,
        "prefix_embedding_mode": config.prefix_embedding_mode,
        "tail_representation": config.tail_representation,
        "latent_target_scope": config.latent_target_scope,
        "proposal_corruption_rate": config.proposal_corruption_rate,
        "catalog_decoder_mode": config.catalog_decoder_mode,
        "catalog_candidate_count": config.catalog_candidate_count,
        "endpoint_loss_weight": config.endpoint_loss_weight,
        "population_loss_weight": config.population_loss_weight,
        "catalog_prior_weight": config.catalog_prior_weight,
        "config": _jsonable(asdict(config)),
        "initialize_from": _optional_str(config.initialize_from),
        "optimizer_state_reused": False,
    }


def _log_epoch(
    epoch: int,
    *,
    train_loss: float,
    val_loss: float | None,
    learning_rate: float,
    stale_validations: int,
) -> None:
    if val_loss is None:
        logger.info(
            "latent suffix-flow epoch=%d train_loss=%.6f "
            "val_loss=skipped lr=%.6g",
            epoch,
            train_loss,
            learning_rate,
        )
    else:
        logger.info(
            "latent suffix-flow epoch=%d train_loss=%.6f "
            "val_loss=%.6f lr=%.6g stale_validations=%d",
            epoch,
            train_loss,
            val_loss,
            learning_rate,
            stale_validations,
        )


# Config settings copied verbatim into the training report.
_REPORTED_SETTINGS = (
    "codebook_size",
    "prefix_depth",
    "prefix_embedding_mode",
    "tail_representation",
    "latent_target_scope",
    "proposal_corruption_rate",
    "catalog_decoder_mode",
    "tail_normalization",
    "factorized_sum_consistency_weight",
    "catalog_loss_weight",
    "catalog_negative_count",
    "catalog_temperature",
    "catalog_candidate_count",
    "endpoint_loss_weight",
    "population_loss_weight",
    "catalog_prior_weight",
    "compile",
    "mixed_precision",
    "tf32",
    "lr_scheduler",
    "warmup_steps",
    "min_lr",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "val_every",
    "num_workers",
    "prefetch_factor",
)


def train_prefix_time_suffix_latent_flow_realizer(
    config: PrefixTimeSuffixLatentFlowRealizerConfig,
) -> dict[str, Any]:
    set_seed_all(config.seed)
    torch.set_num_threads(config.cpu_threads)
    device = _resolve_device(config.device)
    distributed, rank, world_size = _distributed_context()
    is_main = rank == 0
    if distributed and device.type != "cuda":
        raise ValueError("distributed latent realizer training requires CUDA")
    _configure_cuda_math(device=device, tf32=config.tf32)

    bundle = _load_codebooks(config)
    config = replace(config, codebook_size=bundle.codebook_size)
    train, val = load_bank_splits(config, codebook_size=bundle.codebook_size)
    num_codebooks = int(train.codes.shape[-1])
    if bundle.n_codebooks != num_codebooks:
        raise ValueError("codebooks and bank have different SID widths")
    codebooks = bundle.codebooks.float()
    latent_dim = int(codebooks.shape[-1])
    codebook_metadata = codebook_bundle_metadata(bundle)
    initialization = _load_initialization(config, codebooks)
    train_loader, val_loader = _build_loaders(
        config,
        train,
        val,
        device=device,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    target_level_start = (
        config.prefix_depth if config.latent_target_scope == "suffix" else 0
    )
    tail = _tail_stats(
        config,
        initialization,
        train,
        codebooks,
        target_level_start=target_level_start,
        device=device,
    ).to(device)
    catalog_candidates, candidate_bank, corruption_index = _catalog_objectives(
        config,
        train,
        codebooks,
        target_level_start=target_level_start,
        device=device,
    )
    model: torch.nn.Module = _build_model(
        config, codebooks, max_seq_len=int(train.codes.shape[1])
    ).to(device)
    if initialization is not None:
        model.load_state_dict(initialization["model_state_dict"], strict=True)
        if is_main:
            logger.info(
                "warm-started realizer from %s", config.initialize_from
            )
    if config.compile and device.type == "cuda":
        if is_main:
            logger.info("torch.compile enabled backend=inductor dynamic=True")
        model = torch.compile(model, dynamic=True)
    if distributed:
        model = DDP(
            model,
            device_ids=[device.index],
            broadcast_buffers=False,
        )
        set_seed_all(config.seed + rank)
    fused_optimizer = config.fused_optimizer and device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=fused_optimizer,
    )
    total_updates = len(train_loader) * config.epochs
    if config.warmup_steps >= total_updates:
        raise ValueError("warmup_steps must be smaller than total updates")
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=config.lr_scheduler,
        warmup_steps=config.warmup_steps,
        total_updates=total_updates,
        base_lr=config.learning_rate,
        min_lr=config.min_lr,
    )
    autocast, scaler = build_amp_context(config.mixed_precision, device)
    if is_main:
        _log_runtime(
            config,
            world_size=world_size,
            gradient_scaler=scaler is not None,
            fused_optimizer=fused_optimizer,
            total_updates=total_updates,
        )

    tail_payload = tail.cpu_payload()
    codebook_record = {
        "codebooks": codebooks.detach().cpu().contiguous(),
        "codebook_fields": list(bundle.code_fields),
        "codebook_metadata": codebook_metadata,
    }
    tail_stats_path = config.output_dir / "tail_stats.pt"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        torch.save(
            _tail_stats_record(
                config,
                tail_payload,
                codebook_record,
                latent_dim=latent_dim,
                warm_started=initialization is not None,
            ),
            tail_stats_path,
        )
    checkpoint_extra = _checkpoint_extra(
        config,
        tail_payload,
        codebook_record,
        num_codebooks=num_codebooks,
        latent_dim=latent_dim,
    )

    objective = functools.partial(
        prefix_time_suffix_latent_flow_loss,
        codebooks=codebooks.to(device),
        tail_mean=tail.mean,
        tail_std=tail.std,
        tail_whitener=tail.whitener,
        tail_dewhitener=tail.dewhitener,
        lognormal_t=config.lognormal_t,
        factorized_sum_consistency_weight=(
            config.factorized_sum_consistency_weight
        ),
        catalog_candidates=catalog_candidates,
        catalog_loss_weight=config.catalog_loss_weight,
        catalog_negative_count=config.catalog_negative_count,
        catalog_temperature=config.catalog_temperature,
        candidate_bank=candidate_bank,
        endpoint_loss_weight=config.endpoint_loss_weight,
        population_loss_weight=config.population_loss_weight,
        catalog_prior_weight=config.catalog_prior_weight,
    )
    run_epoch = functools.partial(
        _run_epoch,
        objective=objective,
        device=device,
        max_grad_norm=config.max_grad_norm,
        autocast=autocast,
        reduce_across_ranks=distributed,
        corruption_index=corruption_index,
        proposal_corruption_rate=config.proposal_corruption_rate,
    )
    best_path = config.output_dir / "best.pt"
    last_path = config.output_dir / "last.pt"
    best_val = float("inf")
    history: list[dict[str, float | None]] = []
    early_stopper = EarlyStopper(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )
    stopped_early = False
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        val_loss: float | None = None
        if _should_validate(
            epoch, epochs=config.epochs, val_every=config.val_every
        ):
            validation_model = (
                model.module if isinstance(model, DDP) else model
            )
            val_loss = run_epoch(validation_model, val_loader, scaler=None)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
            }
        )
        improved = stop_training = False
        if val_loss is not None:
            decision = early_stopper.step(val_loss, epoch=epoch)
            improved, stop_training = decision.improved, decision.should_stop
            best_val = early_stopper.best_value
        if is_main:
            _log_epoch(
                epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=learning_rate,
                stale_validations=early_stopper.epochs_without_improvement,
            )
            save = functools.partial(
                save_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra=checkpoint_extra,
            )
            save(
                last_path, val_loss=best_val if val_loss is None else val_loss
            )
            if val_loss is not None and improved:
                save(best_path, val_loss=val_loss)
        if distributed:
            best_tensor = torch.tensor(best_val, device=device)
            dist.broadcast(best_tensor, src=0)
            best_val = float(best_tensor.item())
        if stop_training:
            stopped_early = True
            if is_main:
                logger.info(
                    "latent suffix-flow early_stop epoch=%d best_epoch=%d "
                    "best_val_loss=%.6f patience=%d",
                    epoch,
                    early_stopper.best_epoch,
                    best_val,
                    config.early_stopping_patience,
                )
            break

    if distributed:
        dist.barrier()
    if is_main:
        load_checkpoint(best_path, model, device=device)
    result: dict[str, Any] = {
        **{name: getattr(config, name) for name in _REPORTED_SETTINGS},
        "model_type": model_type(config.latent_target_scope),
        "device": str(device),
        "train_rows": int(train.codes.size(0)),
        "val_rows": int(val.codes.size(0)),
        "num_codebooks": num_codebooks,
        "codebook_metadata": codebook_metadata,
        "latent_dim": latent_dim,
        "catalog_path": _optional_str(config.catalog_path),
        "parameter_count": unwrap_ddp(model).count_parameters(),
        "world_size": world_size,
        "per_rank_batch_size": config.batch_size,
        "global_batch_size": config.batch_size * world_size,
        "gradient_scaler": scaler is not None,
        "fused_optimizer": fused_optimizer,
        "stopped_early": stopped_early,
        "epochs_completed": len(history),
        "best_val_loss": best_val,
        "history": history,
        "bank_path": str(config.bank_path),
        "codebooks_path": str(config.codebooks_path),
        "tail_stats_path": str(tail_stats_path),
        "initialize_from": _optional_str(config.initialize_from),
    }
    if is_main:
        write_json(config.output_dir / "metrics.json", result)
    return result


__all__ = [
    "BankSplit",
    "load_bank_splits",
    "train_prefix_time_suffix_latent_flow_realizer",
]
