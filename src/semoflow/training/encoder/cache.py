from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from semoflow.config import ExperimentConfig

logger = logging.getLogger(__name__)


def path_fingerprint(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"path": None, "exists": False}
    candidate = Path(path)
    try:
        stat = candidate.stat()
    except OSError:
        return {"path": str(candidate), "exists": False}
    return {
        "path": str(candidate),
        "exists": True,
        "size": stat.st_size,
        "mtime_ms": int(stat.st_mtime * 1000),
    }


def cbg_attribute_fingerprints(data_root: str | None) -> list[dict[str, Any]]:
    if not data_root:
        return []
    root = Path(data_root)
    if not root.is_dir():
        return []
    infos: list[dict[str, Any]] = []
    for city_dir in sorted(root.iterdir()):
        cbg_path = city_dir / "cbg_attributes.parquet"
        if cbg_path.exists():
            infos.append(path_fingerprint(cbg_path))
    return infos


def graph_data_cache_payload(cfg: ExperimentConfig) -> dict[str, Any]:
    graph_cfg = cfg.encoder_pipeline.graph
    return {
        "version": graph_cfg.cache_version,
        "stays": path_fingerprint(
            cfg.tokenizer_config.processed_visits_output_path()
        ),
        "places_index": path_fingerprint(
            cfg.tokenizer_config.places_index_output_path()
        ),
        "text_embeddings": path_fingerprint(
            cfg.encoder_pipeline.text.out_path
        ),
        "cbg_attributes": cbg_attribute_fingerprints(cfg.data.data_root),
        "hidden_dim": graph_cfg.hidden_dim,
        "covisit_edges": graph_cfg.covisit_edges,
        "covisit_tc_hours": graph_cfg.covisit_tc_hours,
        "covisit_ts_intermediate": graph_cfg.covisit_ts_intermediate,
        "covisit_min_shift": graph_cfg.covisit_min_shift,
        "covisit_top_k": graph_cfg.covisit_top_k,
        "hash_train_fraction": graph_cfg.hash_train_fraction,
        "seed": graph_cfg.seed,
        "city_balanced_supervision": graph_cfg.city_balanced_supervision,
        "include_user_nodes": graph_cfg.include_user_nodes,
        "split_scope": "train",
    }


def graph_data_cache_dir(cfg: ExperimentConfig) -> Path:
    payload = graph_data_cache_payload(cfg)
    key = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return Path(cfg.encoder_pipeline.graph.out_dir) / "cache" / digest


def load_graph_data_cache(
    cache_dir: Path,
    cfg: ExperimentConfig,
) -> tuple[Any, torch.Tensor | None] | None:
    graph_cfg = cfg.encoder_pipeline.graph
    payload_path = cache_dir / graph_cfg.cache_payload
    meta_path = cache_dir / graph_cfg.cache_meta
    if not payload_path.exists() or not meta_path.exists():
        return None
    try:
        result = torch.load(
            payload_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception:
        logger.warning(
            "graph_data_cache load failed; rebuilding path=%s",
            cache_dir,
            exc_info=True,
        )
        return None
    logger.info("graph_data_cache hit path=%s", cache_dir)
    return result


def write_graph_data_cache(
    cfg: ExperimentConfig,
    cache_dir: Path,
    data: Any,
    val_edges: torch.Tensor | None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_cfg = cfg.encoder_pipeline.graph
    payload_path = cache_dir / graph_cfg.cache_payload
    meta_path = cache_dir / graph_cfg.cache_meta

    fd_payload, tmp_payload = tempfile.mkstemp(
        suffix=".pt", dir=str(cache_dir)
    )
    os.close(fd_payload)
    fd_meta, tmp_meta = tempfile.mkstemp(suffix=".json", dir=str(cache_dir))
    os.close(fd_meta)
    try:
        torch.save((data, val_edges), tmp_payload)
        meta = {
            "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "payload": graph_data_cache_payload(cfg),
        }
        Path(tmp_meta).write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_payload, payload_path)
        os.replace(tmp_meta, meta_path)
        logger.info("graph_data_cache wrote path=%s", cache_dir)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_payload)
        with contextlib.suppress(OSError):
            os.unlink(tmp_meta)


__all__ = [
    "cbg_attribute_fingerprints",
    "graph_data_cache_dir",
    "graph_data_cache_payload",
    "load_graph_data_cache",
    "path_fingerprint",
    "write_graph_data_cache",
]
