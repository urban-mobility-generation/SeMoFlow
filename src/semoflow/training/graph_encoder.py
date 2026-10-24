from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import polars as pl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from semoflow.config import ExperimentConfig
from semoflow.encoder.graph.data import build_hetero_data, build_key_index
from semoflow.encoder.graph.heterosage import HeteroSAGE
from semoflow.encoder.graph.losses import (
    contrastive_link_loss,
    variance_covariance_penalty,
)
from semoflow.encoder.graph.sampler import get_train_loader
from semoflow.preprocess.places_index import load_places_index
from semoflow.training.encoder.cache import (
    graph_data_cache_dir as _graph_data_cache_dir,
)
from semoflow.training.encoder.cache import (
    load_graph_data_cache as _load_graph_data_cache,
)
from semoflow.training.encoder.cache import (
    write_graph_data_cache as _write_graph_data_cache,
)
from semoflow.training.utils.distributed import (
    ddp_worker,
    find_free_port,
    unwrap_ddp,
)
from semoflow.training.utils.mixed_precision import build_amp_context
from semoflow.training.utils.runtime import (
    parse_device_spec,
    set_seed_all,
)
from semoflow.utils.serialization import write_json

logger = logging.getLogger(__name__)


class GraphBatch:
    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def to(
        self, device: torch.device, *, non_blocking: bool = False
    ) -> GraphBatch:
        return GraphBatch(self._raw.to(device, non_blocking=non_blocking))

    @property
    def node_features(self) -> dict[str, torch.Tensor]:
        return self._raw.x_dict

    @property
    def node_ids(self) -> dict[str, torch.Tensor]:
        ids: dict[str, torch.Tensor] = {}
        for node_type in getattr(self._raw, "node_types", ()):
            store = self._raw[node_type]
            n_id = getattr(store, "n_id", None)
            if n_id is not None:
                ids[node_type] = n_id
        return ids

    @property
    def edge_indices(self) -> dict[tuple[str, str, str], torch.Tensor]:
        return self._raw.edge_index_dict

    @property
    def edge_types(self) -> list[tuple[str, str, str]]:
        return list(self._raw.edge_types)

    def first_labelled_edge_type(self) -> tuple[str, str, str] | None:
        for edge_type in self.edge_types:
            edge_store = self._raw[edge_type]
            if (
                hasattr(edge_store, "edge_label_index")
                and edge_store.edge_label_index is not None
            ):
                return edge_type
        return None

    def edge_labels(
        self, edge_type: tuple[str, str, str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_store = self._raw[edge_type]
        return edge_store.edge_label_index, edge_store.edge_label


def _as_graph_batch(batch: Any) -> GraphBatch:
    if isinstance(batch, GraphBatch):
        return batch
    return GraphBatch(batch)


def _resolve_resume_checkpoint(
    graph_cfg: Any,
) -> Path | None:
    raw = graph_cfg.resume_checkpoint
    if not raw:
        return None
    p = Path(raw)
    if not p.exists():
        return None
    return p


def apply_graph_warm_start(
    raw_model: Any, source_path: Path, *, device: Any
) -> dict[str, Any]:
    raw = torch.load(source_path, map_location=device, weights_only=False)
    source_state = (
        raw["model_state_dict"]
        if isinstance(raw, dict) and "model_state_dict" in raw
        else raw
    )
    if not isinstance(source_state, dict):
        raise ValueError(
            f"graph warm-start checkpoint is not a state dict: {source_path}"
        )
    target_state = raw_model.state_dict()
    copied: list[str] = []
    reinitialized: list[str] = []
    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if (
            source_tensor is not None
            and hasattr(source_tensor, "shape")
            and source_tensor.shape == target_tensor.shape
        ):
            target_state[name] = source_tensor.to(target_tensor.dtype)
            copied.append(name)
        else:
            reinitialized.append(name)
    if not copied:
        raise ValueError(
            "graph warm start copied no tensors; the source checkpoint "
            f"architecture is incompatible: {source_path}"
        )
    raw_model.load_state_dict(target_state)
    return {
        "checkpoint": str(source_path),
        "copied_tensors": len(copied),
        "reinitialized_tensors": sorted(reinitialized),
        "source_only_tensors": sorted(set(source_state) - set(target_state)),
    }


def _load_cbg_attributes(
    data_root: str | None,
) -> pl.DataFrame | None:
    if not data_root:
        return None
    root = Path(data_root)
    if not root.is_dir():
        return None
    parts = []
    for city_dir in sorted(root.iterdir()):
        cbg_path = city_dir / "cbg_attributes.parquet"
        if cbg_path.exists():
            parts.append(pl.read_parquet(cbg_path))
    if not parts:
        logger.info("No cbg_attributes.parquet found under %s", root)
        return None
    df = pl.concat(parts, how="diagonal_relaxed")
    logger.info(
        "Loaded CBG attributes: %d rows from %d cities",
        df.height,
        len(parts),
    )
    return df


class GraphEncoderTrainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        mixed_precision: str = "bf16",
        variance_weight: float = 0.0,
        covariance_weight: float = 0.0,
        variance_target: float = 1.0,
        temperature: float = 0.07,
        max_negatives: int = 2048,
        max_queries: int = 65536,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.amp_ctx, self.scaler = build_amp_context(mixed_precision, device)
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        self.variance_target = float(variance_target)
        self.temperature = float(temperature)
        self.max_negatives = int(max_negatives)
        self.max_queries = int(max_queries)

    def train_batch_tensor(self, batch: Any) -> torch.Tensor:
        graph_batch = _as_graph_batch(batch).to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        self.optimizer.zero_grad(set_to_none=True)
        with self.amp_ctx:
            poi_emb, target_edge = self._forward_with_target_edge(graph_batch)

        if target_edge is None:
            dummy_loss = poi_emb.sum() * 0.0
            if self.scaler is not None:
                self.scaler.scale(dummy_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                dummy_loss.backward()
                self.optimizer.step()
            return dummy_loss.detach()

        with self.amp_ctx:
            loss = self._total_loss(graph_batch, poi_emb, target_edge)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        return loss.detach()

    def eval_epoch_totals(self, val_loader: Any) -> tuple[float, float]:
        self.model.eval()
        total_loss = torch.zeros((), device=self.device)
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                graph_batch = _as_graph_batch(batch).to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
                with self.amp_ctx:
                    poi_emb, target_edge = self._forward_with_target_edge(
                        graph_batch
                    )
                if target_edge is None:
                    continue
                with self.amp_ctx:
                    total_loss += self._total_loss(
                        graph_batch, poi_emb, target_edge
                    ).detach()
                n_batches += 1
        self.model.train()
        return float(total_loss.item()), float(n_batches)

    def _forward_with_target_edge(
        self, batch: GraphBatch
    ) -> tuple[torch.Tensor, tuple[str, str, str] | None]:
        raw = unwrap_ddp(self.model)
        node_ids = batch.node_ids
        ids = (
            node_ids
            if node_ids and getattr(raw, "id_embeddings", None)
            else None
        )
        z_dict = self.model(batch.node_features, batch.edge_indices, ids)
        poi_emb = z_dict["poi"]
        return poi_emb, batch.first_labelled_edge_type()

    def _link_loss(
        self,
        batch: GraphBatch,
        poi_emb: torch.Tensor,
        target_edge: tuple[str, str, str],
    ) -> torch.Tensor:
        edge_label_index, edge_label = batch.edge_labels(target_edge)

        pos_mask = edge_label == 1
        neg_mask = edge_label == 0

        pos_edges = (
            edge_label_index[0, pos_mask],
            edge_label_index[1, pos_mask],
        )
        neg_edges = (
            edge_label_index[0, neg_mask],
            edge_label_index[1, neg_mask],
        )
        return contrastive_link_loss(
            poi_emb,
            pos_edges,
            neg_edges=neg_edges,
            temperature=self.temperature,
            max_negatives=self.max_negatives,
            max_queries=self.max_queries,
        )

    def _total_loss(
        self,
        batch: GraphBatch,
        poi_emb: torch.Tensor,
        target_edge: tuple[str, str, str],
    ) -> torch.Tensor:
        loss = self._link_loss(batch, poi_emb, target_edge)
        if self.variance_weight == 0.0 and self.covariance_weight == 0.0:
            return loss
        variance_loss, covariance_loss = variance_covariance_penalty(
            poi_emb.float(), variance_target=self.variance_target
        )
        return (
            loss
            + self.variance_weight * variance_loss
            + self.covariance_weight * covariance_loss
        )


def _evaluate_validation(
    trainer: Any,
    val_loader: Any,
    device: torch.device,
    world_size: int,
) -> float:
    total, batches = trainer.eval_epoch_totals(val_loader)
    if world_size <= 1:
        return float(total / max(batches, 1.0))
    packed = torch.tensor([total, batches], dtype=torch.float64, device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    combined_total = float(packed[0].item())
    combined_batches = float(packed[1].item())
    return combined_total / max(combined_batches, 1.0)


def _wrap_graph_model_for_ddp(
    model: torch.nn.Module,
    device: torch.device,
    world_size: int,
) -> torch.nn.Module:
    if world_size > 1 and device.type == "cuda":
        return DDP(
            model,
            device_ids=[device.index],
            find_unused_parameters=True,
        )
    return model


def _build_graph_data(
    cfg: ExperimentConfig,
) -> tuple[Any, torch.Tensor | None]:
    graph_cfg = cfg.encoder_pipeline.graph

    processed_visits_path = cfg.tokenizer_config.processed_visits_output_path()
    places_index_path = cfg.tokenizer_config.places_index_output_path()
    text_emb_path = cfg.encoder_pipeline.text.out_path

    if not processed_visits_path.exists():
        raise FileNotFoundError(
            f"Missing stays path for the graph encoder: {processed_visits_path}"
        )
    if not places_index_path.exists():
        raise FileNotFoundError(
            f"Missing POI index for the graph encoder: {places_index_path}"
        )

    cache_dir = _graph_data_cache_dir(cfg)
    cached = _load_graph_data_cache(cache_dir, cfg)
    if cached is not None:
        return cached

    logger.info("Loading processed visits and POI index...")
    stays_df = pl.read_parquet(processed_visits_path)
    if "split" in stays_df.columns:
        n_all = stays_df.height
        stays_df = stays_df.filter(pl.col("split") == "train")
        logger.info(
            "graph build restricted to train split: %s/%s stays",
            f"{stays_df.height:,}",
            f"{n_all:,}",
        )
    places_index_df = load_places_index(places_index_path)
    cbg_df = _load_cbg_attributes(cfg.data.data_root)

    data, val_edges = build_hetero_data(
        stays_df=stays_df,
        places_index_df=places_index_df,
        text_embeddings_path=text_emb_path,
        cbg_attributes_df=cbg_df,
        hidden_dim=graph_cfg.hidden_dim,
        train_fraction=graph_cfg.hash_train_fraction,
        seed=graph_cfg.seed,
        covisit_edges=graph_cfg.covisit_edges,
        covisit_tc_hours=graph_cfg.covisit_tc_hours,
        covisit_ts_intermediate=graph_cfg.covisit_ts_intermediate,
        covisit_min_shift=graph_cfg.covisit_min_shift,
        covisit_min_covisits=graph_cfg.dependency_min_covisits,
        covisit_top_k=graph_cfg.covisit_top_k,
        include_user_nodes=graph_cfg.include_user_nodes,
    )
    _write_graph_data_cache(cfg, cache_dir, data, val_edges)

    from semoflow.utils.provenance import (
        closure_from_stays,
        write_closure,
    )
    from semoflow.utils.splits import GLOBAL_SPLIT_SEED

    out_dir = Path(graph_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_closure(
        closure_from_stays(
            stays_df,
            artifact=str(out_dir),
            source="graphsage_behavioral_graph",
            split_seed=GLOBAL_SPLIT_SEED,
        ),
        out_dir,
    )
    return data, val_edges


def _id_embedding_counts(graph_cfg: Any, data: Any) -> dict[str, int] | None:
    """Node count per node type that gets a learned ID embedding table."""
    if graph_cfg.id_embedding_dim <= 0:
        return None
    selected_types = graph_cfg.id_embedding_node_types
    if selected_types is None:
        embedded_types = list(data.node_types)
    else:
        unknown = sorted(set(selected_types).difference(data.node_types))
        if unknown:
            raise ValueError(
                "encoder_pipeline.graph.id_embedding_node_types names "
                f"node types absent from the graph: {unknown}; graph has "
                f"{sorted(data.node_types)}"
            )
        embedded_types = [
            ntype for ntype in data.node_types if ntype in selected_types
        ]
    id_counts = {ntype: int(data[ntype].num_nodes) for ntype in embedded_types}
    logger.info(
        "id embedding tables: %s  parameters=%s",
        ", ".join(
            f"{ntype}={count:,}x{graph_cfg.id_embedding_dim}"
            for ntype, count in id_counts.items()
        ),
        f"{sum(id_counts.values()) * graph_cfg.id_embedding_dim:,}",
    )
    return id_counts


def _warm_start_graph_model(
    graph_cfg: Any,
    raw_model: torch.nn.Module,
    *,
    device: torch.device,
    is_main: bool,
) -> None:
    if not graph_cfg.warm_start_checkpoint:
        return
    warm_start_path = Path(graph_cfg.warm_start_checkpoint)
    if not warm_start_path.exists():
        raise ValueError(
            f"graph warm-start checkpoint not found: {warm_start_path}"
        )
    if _resolve_resume_checkpoint(graph_cfg) is not None:
        raise ValueError(
            "warm_start_checkpoint and an existing resume_checkpoint are "
            "mutually exclusive; a warm start is initialization, not a "
            "resume"
        )
    record = apply_graph_warm_start(raw_model, warm_start_path, device=device)
    if is_main:
        logger.info(
            "Graph warm start from %s — copied %d tensors, "
            "reinitialized %d (%s)",
            warm_start_path,
            record["copied_tensors"],
            len(record["reinitialized_tensors"]),
            ", ".join(record["reinitialized_tensors"]),
        )
        write_json(Path(graph_cfg.out_dir) / "warm_start.json", record)


def _resume_graph_training(
    graph_cfg: Any,
    raw_model: torch.nn.Module,
    optimizer: optim.Optimizer,
    *,
    device: torch.device,
    is_main: bool,
) -> tuple[int, float]:
    """Restore an existing checkpoint; return (start epoch, best val loss)."""
    resume_path = _resolve_resume_checkpoint(graph_cfg)
    if resume_path is None:
        return 0, float("inf")
    start_epoch, best_val_loss = 0, float("inf")
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(_model_state(ckpt))
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
    if is_main:
        logger.info(
            "Resumed from %s — starting at epoch %d, best_val=%.4f",
            resume_path,
            start_epoch + 1,
            best_val_loss,
        )
    return start_epoch, best_val_loss


def _model_state(checkpoint: Any) -> Any:
    """The model state of a full training checkpoint or a bare state dict."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _export_poi_embeddings(
    cfg: ExperimentConfig,
    raw_model: torch.nn.Module,
    data: Any,
    out_dir: Path,
) -> tuple[torch.Tensor, Path, Path]:
    """Write CPU POI embeddings and the places index aligned to their rows."""
    logger.info("Extracting final POI embeddings...")
    raw_model.eval()
    raw_model.to("cpu")
    with torch.no_grad():
        data_cpu = data.to("cpu")
        ids = (
            {
                ntype: torch.arange(
                    int(data_cpu[ntype].num_nodes), dtype=torch.long
                )
                for ntype in data_cpu.node_types
            }
            if getattr(raw_model, "id_embeddings", None)
            else None
        )
        poi_graph_emb = raw_model(
            data_cpu.x_dict, data_cpu.edge_index_dict, ids
        )["poi"]
    emb_path = out_dir / "poi_graph_emb.pt"
    torch.save(poi_graph_emb, emb_path)

    places_index_df = load_places_index(
        cfg.tokenizer_config.places_index_output_path()
    )
    identity_cols = [
        c for c in ("placekey_id", "placekey") if c in places_index_df.columns
    ]
    poi_col = (
        "placekey_id"
        if "placekey_id" in places_index_df.columns
        else "placekey"
    )
    index_path = out_dir / "places_index.parquet"
    (
        build_key_index(places_index_df[poi_col], poi_col)
        .rename({"idx": "embedding_row"})
        .join(
            places_index_df.select(identity_cols).unique(
                poi_col, keep="first"
            ),
            on=poi_col,
            how="left",
            maintain_order="left",
        )
        .write_parquet(index_path)
    )
    return poi_graph_emb, emb_path, index_path


def _train_graph_single(
    cfg: ExperimentConfig,
    data: Any,
    val_edges: torch.Tensor | None,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    is_main = rank == 0
    graph_cfg = cfg.encoder_pipeline.graph
    out_dir = Path(graph_cfg.out_dir)
    checkpoints_dir = out_dir / "checkpoints"
    best_path = checkpoints_dir / "best.pt"
    if is_main:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    seed = graph_cfg.seed
    torch.manual_seed(seed + rank)
    if is_main:
        logger.info("Initializing LinkNeighborLoader...")
    loader, val_loader = (
        get_train_loader(
            data,
            batch_size=graph_cfg.batch_size,
            num_neighbors=graph_cfg.num_neighbors,
            train_edge_index=edges,
            num_workers=graph_cfg.num_workers,
            pin_memory=device.type == "cuda",
            seed=seed,
            rank=rank,
            world_size=world_size,
            city_balanced=graph_cfg.city_balanced_supervision,
        )
        for edges in (None, val_edges)
    )

    in_dims = {ntype: data[ntype].x.size(1) for ntype in data.node_types}
    model: torch.nn.Module = HeteroSAGE(
        metadata=data.metadata(),
        in_dims=in_dims,
        hidden_dim=graph_cfg.hidden_dim,
        out_dim=graph_cfg.out_dim,
        num_layers=graph_cfg.num_layers,
        dropout=graph_cfg.dropout,
        aggr=graph_cfg.aggr,
        layer_norm=graph_cfg.layer_norm,
        residual=graph_cfg.residual,
        id_embedding_dim=graph_cfg.id_embedding_dim,
        id_embedding_counts=_id_embedding_counts(graph_cfg, data),
    ).to(device)
    model = _wrap_graph_model_for_ddp(model, device, world_size)
    raw_model = unwrap_ddp(model)
    n_parameters = sum(p.numel() for p in raw_model.parameters())
    optimizer = optim.Adam(raw_model.parameters(), lr=graph_cfg.lr)
    trainer = GraphEncoderTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        mixed_precision=graph_cfg.mixed_precision,
        variance_weight=graph_cfg.variance_weight,
        covariance_weight=graph_cfg.covariance_weight,
        variance_target=graph_cfg.variance_target,
        temperature=graph_cfg.temperature,
        max_negatives=graph_cfg.max_negatives,
        max_queries=graph_cfg.max_queries,
    )
    _warm_start_graph_model(
        graph_cfg, raw_model, device=device, is_main=is_main
    )
    start_epoch, best_val_loss = _resume_graph_training(
        graph_cfg, raw_model, optimizer, device=device, is_main=is_main
    )

    epochs = graph_cfg.epochs
    history: list[dict[str, float]] = []
    model.train()
    if is_main:
        logger.info(
            "Starting %s  epochs=%d  lr=%g  device=%s  "
            "params=%d  world_size=%d",
            type(raw_model).__name__,
            epochs,
            graph_cfg.lr,
            device,
            n_parameters,
            world_size,
        )
    for epoch in range(start_epoch, epochs):
        t_epoch = time.perf_counter()
        epoch_loss = torch.tensor(0.0, device=device)
        epoch_batches = torch.tensor(0, device=device, dtype=torch.long)
        desc = f"Epoch {epoch + 1}/{epochs}"
        for batch in tqdm(loader, desc=desc) if is_main else loader:
            epoch_loss += trainer.train_batch_tensor(batch)
            epoch_batches += 1
        if world_size > 1:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_batches, op=dist.ReduceOp.SUM)
        train_loss = float(epoch_loss.item()) / max(
            int(epoch_batches.item()), 1
        )
        val_loss = _evaluate_validation(
            trainer, val_loader, device, world_size
        )
        epoch_secs = time.perf_counter() - t_epoch
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_s": round(epoch_secs, 1),
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if is_main:
                torch.save(
                    {
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                    },
                    best_path,
                )
        if is_main:
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  "
                "best_val=%.4f  %.1fs",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                best_val_loss,
                epoch_secs,
            )

    result: dict[str, Any] = {
        "best_val_loss": best_val_loss,
        "n_parameters": n_parameters,
        "checkpoint_path": str(best_path),
    }
    if not is_main:
        return result

    if best_path.exists():
        raw_model.load_state_dict(
            _model_state(
                torch.load(best_path, map_location="cpu", weights_only=False)
            )
        )
    poi_graph_emb, emb_path, index_path = _export_poi_embeddings(
        cfg, raw_model, data, out_dir
    )
    metrics_path = out_dir / "metrics.json"
    write_json(
        metrics_path,
        {
            "model": type(raw_model).__name__,
            "n_parameters": n_parameters,
            "n_pois": int(poi_graph_emb.shape[0]),
            "embedding_dim": int(poi_graph_emb.shape[1]),
            "best_val_loss": best_val_loss,
            "epochs": epochs,
            "world_size": world_size,
            "history": history,
        },
    )
    logger.info(
        "Saved GraphSAGE artifacts to %s  "
        "embeddings=%s  index=%s  metrics=%s  checkpoint=%s",
        out_dir,
        emb_path,
        index_path,
        metrics_path,
        best_path,
    )
    result.update(
        {
            "n_pois": int(poi_graph_emb.shape[0]),
            "embedding_dim": int(poi_graph_emb.shape[1]),
            "embedding_path": str(emb_path),
        }
    )
    return result


def _train_graph_worker(
    rank: int,
    device_ids: list[int],
    port: int,
    cfg: ExperimentConfig,
    graph_data_path: str,
    tmp_result_path: str,
    log_level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    try:
        with ddp_worker(rank, device_ids, port, log_level, log_file) as device:
            logger.info(
                "DDP worker started  rank=%d  device=%s  world_size=%d",
                rank,
                device,
                len(device_ids),
            )
            data, val_edges = torch.load(
                graph_data_path,
                map_location="cpu",
                weights_only=False,
            )
            result = _train_graph_single(
                cfg, data, val_edges, device, rank, len(device_ids)
            )
            del data, val_edges
            if rank == 0:
                Path(tmp_result_path).write_text(
                    json.dumps(result, default=str), encoding="utf-8"
                )
    except Exception:
        logger.exception(
            "DDP worker failed  rank=%d  device=cuda:%d",
            rank,
            device_ids[rank],
        )
        raise


def train_heterosage(
    cfg: ExperimentConfig,
) -> dict[str, Any]:
    graph_cfg = cfg.encoder_pipeline.graph
    set_seed_all(graph_cfg.seed)

    primary_device, device_ids = parse_device_spec(graph_cfg.device)

    data, val_edges = _build_graph_data(cfg)

    if not device_ids:
        return _train_graph_single(
            cfg, data, val_edges, primary_device, rank=0, world_size=1
        )

    from semoflow.logging.primitives import get_configured_log_file
    from semoflow.logging.training import log_ddp_summary

    port = find_free_port()
    log_ddp_summary(
        logger,
        device_ids=device_ids,
        port=port,
        batch_per_rank=graph_cfg.batch_size,
    )
    _log_level = logging.getLevelName(logging.getLogger().level)
    _log_file = get_configured_log_file()

    fd_graph, tmp_graph = tempfile.mkstemp(suffix=".pt")
    os.close(fd_graph)
    torch.save((data, val_edges), tmp_graph)
    del data, val_edges
    logger.info("Graph data saved to %s for DDP workers", tmp_graph)

    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        mp.spawn(
            _train_graph_worker,
            args=(
                device_ids,
                port,
                cfg,
                tmp_graph,
                tmp,
                _log_level,
                _log_file,
            ),
            nprocs=len(device_ids),
            join=True,
        )
        raw = Path(tmp).read_text(encoding="utf-8").strip()
        if not raw:
            raise RuntimeError(
                "DDP result file is empty — rank-0 worker "
                "likely crashed before writing results"
            )
        return json.loads(raw)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        with contextlib.suppress(OSError):
            os.unlink(tmp_graph)
