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
from semoflow.utils.serialization import write_json

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


def graph_data_cache_payload(cfg: ExperimentConfig) -> dict[str, Any]:
    graph_cfg = cfg.encoder.graph
    return {
        "version": 1,
        "stays": path_fingerprint(cfg.layout.preprocessing.processed_visits),
        "places_index": path_fingerprint(
            cfg.layout.preprocessing.places_index
        ),
        "text_embeddings": path_fingerprint(cfg.layout.text_embeddings),
        "hidden_dim": graph_cfg.hidden_dim,
        "hash_train_fraction": graph_cfg.hash_train_fraction,
        "seed": graph_cfg.seed,
        "split_scope": "train",
    }


def graph_data_cache_dir(cfg: ExperimentConfig) -> Path:
    payload = graph_data_cache_payload(cfg)
    key = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return cfg.layout.graph_dir / "cache" / digest


def load_graph_data_cache(
    cache_dir: Path,
    cfg: ExperimentConfig,
) -> tuple[Any, torch.Tensor | None] | None:
    payload_path = cache_dir / "graph_data.pt"
    meta_path = cache_dir / "meta.json"
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
    payload_path = cache_dir / "graph_data.pt"
    meta_path = cache_dir / "meta.json"

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
        write_json(Path(tmp_meta), meta)
        os.replace(tmp_payload, payload_path)
        os.replace(tmp_meta, meta_path)
        logger.info("graph_data_cache wrote path=%s", cache_dir)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_payload)
        with contextlib.suppress(OSError):
            os.unlink(tmp_meta)


__all__ = [
    "graph_data_cache_dir",
    "graph_data_cache_payload",
    "load_graph_data_cache",
    "path_fingerprint",
    "write_graph_data_cache",
]
