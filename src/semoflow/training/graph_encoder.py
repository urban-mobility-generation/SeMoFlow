"""Train the HeteroSAGE POI graph encoder and export its POI embeddings."""

from __future__ import annotations

import contextlib
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
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from semoflow.config import ExperimentConfig
from semoflow.config.schema import GraphEncoderConfig
from semoflow.encoder.graph.data import build_hetero_data, build_key_index
from semoflow.encoder.graph.heterosage import HeteroSAGE
from semoflow.encoder.graph.losses import (
    contrastive_link_loss,
    variance_covariance_penalty,
)
from semoflow.encoder.graph.sampler import get_train_loader
from semoflow.training.utils.distributed import (
    ddp_worker,
    find_free_port,
    unwrap_ddp,
)
from semoflow.training.utils.runtime import parse_device_spec, set_seed_all
from semoflow.utils.log import get_configured_log_file, log_ddp_summary
from semoflow.utils.serialization import write_json

logger = logging.getLogger(__name__)


def _autocast(device: torch.device) -> contextlib.AbstractContextManager:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _batch_loss(
    model: torch.nn.Module,
    batch: Any,
    graph_cfg: GraphEncoderConfig,
    device: torch.device,
) -> torch.Tensor:
    """Contrastive link loss over the batch's labelled POI-POI edges, plus
    the optional variance/covariance penalty on the POI embeddings."""
    batch = batch.to(device, non_blocking=device.type == "cuda")
    ids = (
        {ntype: batch[ntype].n_id for ntype in batch.node_types}
        if unwrap_ddp(model).id_embeddings
        else None
    )
    with _autocast(device):
        poi = model(batch.x_dict, batch.edge_index_dict, ids)["poi"]
        (target,) = (
            batch[edge_type]
            for edge_type in batch.edge_types
            if "edge_label_index" in batch[edge_type]
        )
        index, label = target.edge_label_index, target.edge_label
        loss = contrastive_link_loss(
            poi,
            (index[0, label == 1], index[1, label == 1]),
            neg_edges=(index[0, label == 0], index[1, label == 0]),
            temperature=graph_cfg.temperature,
            max_negatives=graph_cfg.max_negatives,
            max_queries=graph_cfg.max_queries,
        )
        if graph_cfg.variance_weight or graph_cfg.covariance_weight:
            variance, covariance = variance_covariance_penalty(
                poi.float(), variance_target=graph_cfg.variance_target
            )
            loss = (
                loss
                + graph_cfg.variance_weight * variance
                + graph_cfg.covariance_weight * covariance
            )
    return loss


def _mean_over_ranks(
    total: torch.Tensor, batches: int, world_size: int
) -> float:
    packed = torch.stack(
        [total.double(), torch.tensor(batches, device=total.device).double()]
    )
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return float(packed[0] / packed[1].clamp(min=1))


def _build_graph_data(
    cfg: ExperimentConfig,
) -> tuple[Any, torch.Tensor | None]:
    """The train-split heterogeneous POI graph and its held-out edges."""
    graph_cfg = cfg.encoder.graph
    logger.info("Loading processed visits and POI index...")
    stays = pl.read_parquet(cfg.layout.preprocessing.processed_visits)
    n_all = stays.height
    stays = stays.filter(pl.col("split") == "train")
    logger.info(
        "graph build restricted to train split: %s/%s stays",
        f"{stays.height:,}",
        f"{n_all:,}",
    )
    return build_hetero_data(
        stays_df=stays,
        places_index_df=pl.read_parquet(cfg.layout.preprocessing.places_index),
        text_embeddings_path=cfg.layout.text_embeddings,
        hidden_dim=graph_cfg.hidden_dim,
        train_fraction=graph_cfg.hash_train_fraction,
        seed=graph_cfg.seed,
    )


def _id_embedding_counts(
    graph_cfg: GraphEncoderConfig, data: Any
) -> dict[str, int] | None:
    """Node count per node type that gets a learned ID embedding table."""
    if graph_cfg.id_embedding_dim <= 0:
        return None
    selected = graph_cfg.id_embedding_node_types
    if selected is not None:
        unknown = sorted(set(selected).difference(data.node_types))
        if unknown:
            raise ValueError(
                "encoder.graph.id_embedding_node_types names node types "
                f"absent from the graph: {unknown}; graph has "
                f"{sorted(data.node_types)}"
            )
    counts = {
        ntype: int(data[ntype].num_nodes)
        for ntype in data.node_types
        if selected is None or ntype in selected
    }
    logger.info(
        "id embedding tables: %s  parameters=%s",
        ", ".join(
            f"{ntype}={count:,}x{graph_cfg.id_embedding_dim}"
            for ntype, count in counts.items()
        ),
        f"{sum(counts.values()) * graph_cfg.id_embedding_dim:,}",
    )
    return counts


def _export_poi_embeddings(
    cfg: ExperimentConfig, model: torch.nn.Module, data: Any, out_dir: Path
) -> torch.Tensor:
    """Write CPU POI embeddings and the places index aligned to their rows."""
    logger.info("Extracting final POI embeddings...")
    model.eval()
    model.to("cpu")
    data = data.to("cpu")
    ids = (
        {
            ntype: torch.arange(int(data[ntype].num_nodes), dtype=torch.long)
            for ntype in data.node_types
        }
        if model.id_embeddings
        else None
    )
    with torch.no_grad():
        embeddings = model(data.x_dict, data.edge_index_dict, ids)["poi"]
    torch.save(embeddings, out_dir / "poi_graph_emb.pt")
    places = pl.read_parquet(cfg.layout.preprocessing.places_index)
    (
        build_key_index(places["placekey_id"], "placekey_id")
        .rename({"idx": "embedding_row"})
        .join(
            places.select("placekey_id", "placekey"),
            on="placekey_id",
            how="left",
            maintain_order="left",
        )
        .write_parquet(out_dir / "places_index.parquet")
    )
    return embeddings


def _train(
    cfg: ExperimentConfig,
    data: Any,
    val_edges: torch.Tensor | None,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    is_main = rank == 0
    graph_cfg = cfg.encoder.graph
    out_dir = cfg.layout.graph_dir
    best_path = out_dir / "checkpoints" / "best.pt"
    if is_main:
        best_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(graph_cfg.seed + rank)
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
            seed=graph_cfg.seed,
            rank=rank,
            world_size=world_size,
        )
        for edges in (None, val_edges)
    )

    raw_model = HeteroSAGE(
        metadata=data.metadata(),
        in_dims={ntype: data[ntype].x.size(1) for ntype in data.node_types},
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
    model: torch.nn.Module = (
        DistributedDataParallel(
            raw_model, device_ids=[device.index], find_unused_parameters=True
        )
        if world_size > 1
        else raw_model
    )
    n_parameters = sum(p.numel() for p in raw_model.parameters())
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=graph_cfg.lr)

    # An existing best checkpoint resumes training after its epoch.
    start_epoch, best_val_loss = 0, float("inf")
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        if is_main:
            logger.info(
                "Resumed from %s — starting at epoch %d, best_val=%.4f",
                best_path,
                start_epoch + 1,
                best_val_loss,
            )

    epochs = graph_cfg.epochs
    history: list[dict[str, float]] = []
    if is_main:
        logger.info(
            "Starting HeteroSAGE  epochs=%d  lr=%g  device=%s  "
            "params=%d  world_size=%d",
            epochs,
            graph_cfg.lr,
            device,
            n_parameters,
            world_size,
        )
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        model.train()
        total, batches = torch.zeros((), device=device), 0
        desc = f"Epoch {epoch + 1}/{epochs}"
        for batch in tqdm(loader, desc=desc) if is_main else loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(model, batch, graph_cfg, device)
            loss.backward()
            optimizer.step()
            total += loss.detach()
            batches += 1
        train_loss = _mean_over_ranks(total, batches, world_size)

        model.eval()
        total, batches = torch.zeros((), device=device), 0
        with torch.no_grad():
            for batch in val_loader:
                total += _batch_loss(model, batch, graph_cfg, device)
                batches += 1
        val_loss = _mean_over_ranks(total, batches, world_size)

        elapsed = time.perf_counter() - started
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_s": round(elapsed, 1),
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
                elapsed,
            )
    if not is_main:
        return

    if best_path.exists():
        raw_model.load_state_dict(
            torch.load(best_path, map_location="cpu")["model_state_dict"]
        )
    embeddings = _export_poi_embeddings(cfg, raw_model, data, out_dir)
    write_json(
        out_dir / "metrics.json",
        {
            "model": "HeteroSAGE",
            "n_parameters": n_parameters,
            "n_pois": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "best_val_loss": best_val_loss,
            "epochs": epochs,
            "world_size": world_size,
            "history": history,
        },
    )
    logger.info(
        "Saved GraphSAGE artifacts to %s  checkpoint=%s", out_dir, best_path
    )


def _train_worker(
    rank: int,
    device_ids: list[int],
    port: int,
    cfg: ExperimentConfig,
    graph_data_path: str,
    log_level: str,
    log_file: Path | None,
) -> None:
    with ddp_worker(rank, device_ids, port, log_level, log_file) as device:
        data, val_edges = torch.load(graph_data_path, weights_only=False)
        _train(cfg, data, val_edges, device, rank, len(device_ids))


def train_heterosage(cfg: ExperimentConfig) -> None:
    """Train on one device, or spawn one DDP rank per listed GPU."""
    graph_cfg = cfg.encoder.graph
    set_seed_all(graph_cfg.seed)
    device, device_ids = parse_device_spec(graph_cfg.device)
    data, val_edges = _build_graph_data(cfg)
    if not device_ids:
        _train(cfg, data, val_edges, device)
        return

    port = find_free_port()
    log_ddp_summary(
        logger,
        device_ids=device_ids,
        port=port,
        batch_per_rank=graph_cfg.batch_size,
    )
    fd, graph_data_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save((data, val_edges), graph_data_path)
        del data, val_edges
        mp.spawn(
            _train_worker,
            args=(
                device_ids,
                port,
                cfg,
                graph_data_path,
                logging.getLevelName(logging.getLogger().level),
                get_configured_log_file(),
            ),
            nprocs=len(device_ids),
            join=True,
        )
    finally:
        os.unlink(graph_data_path)


__all__ = ["train_heterosage"]
