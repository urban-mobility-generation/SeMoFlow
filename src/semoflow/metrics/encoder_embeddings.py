from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch

from semoflow.utils.serialization import write_json

EmbeddingKind = Literal["text", "graph", "cf"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingEvalPaths:
    json_path: Path
    md_path: Path


def load_embedding_matrix(path: str | Path) -> np.ndarray:
    artifact = Path(path)
    if not artifact.exists():
        raise FileNotFoundError(f"Missing embedding artifact: {artifact}")
    if artifact.suffix == ".npy":
        arr = np.load(artifact)
    else:
        arr = _torch_artifact_to_array(
            torch.load(artifact, map_location="cpu", weights_only=False)
        )
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"Embedding artifact must be 2-D; got shape {arr.shape}"
        )
    return arr


def _torch_artifact_to_array(obj: Any) -> np.ndarray:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().float().numpy()
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        for key in (
            "embeddings",
            "poi_graph_emb",
            "weight",
            "embedding.weight",
            "node_embedding.weight",
            "emb.weight",
        ):
            val = obj.get(key)
            if isinstance(val, torch.Tensor) and val.ndim == 2:
                return val.detach().cpu().float().numpy()
            if isinstance(val, np.ndarray) and val.ndim == 2:
                return val
        tensors = {
            key: val
            for key, val in obj.items()
            if isinstance(val, torch.Tensor) and val.ndim == 2
        }
        if tensors:
            key = max(tensors, key=lambda k: tensors[k].shape[0])
            return tensors[key].detach().cpu().float().numpy()
    raise TypeError(
        f"Could not extract a 2-D embedding matrix from {type(obj)}"
    )


def evaluate_encoder_embeddings(
    embeddings: np.ndarray | torch.Tensor,
    *,
    index: pl.DataFrame | None = None,
    paired_embeddings: np.ndarray | torch.Tensor | None = None,
    paired_kind: EmbeddingKind | str | None = None,
    stays: pl.DataFrame | None = None,
    kind: EmbeddingKind | str | None = None,
    duplicate_sample_size: int = 5000,
    duplicate_cosine_threshold: float = 0.9999,
    near_zero_eps: float = 1e-8,
    purity_k: int = 5,
    purity_sample_size: int = 5000,
    purity_chunk_size: int = 512,
    geometry_sample_size: int = 5000,
    alignment_k: int = 10,
    alignment_sample_size: int = 5000,
    behavior_k: int = 10,
    behavior_sample_size: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    arr = _as_float_matrix(embeddings)
    logger.info(
        "evaluating encoder embeddings  kind=%s  rows=%d  dim=%d",
        kind,
        arr.shape[0],
        arr.shape[1],
    )
    finite_mask = np.isfinite(arr)
    row_finite = finite_mask.all(axis=1)
    finite_arr = arr[row_finite]
    norms = (
        np.linalg.norm(finite_arr, axis=1) if len(finite_arr) else np.array([])
    )

    result: dict[str, Any] = {
        "kind": kind,
        "n_items": int(arr.shape[0]),
        "embedding_dim": int(arr.shape[1]),
        "finite_count": int(finite_mask.sum()),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
        "nonfinite_row_count": int((~row_finite).sum()),
        "near_zero_row_count": int((norms <= near_zero_eps).sum()),
        "norm_min": _stat(norms, "min"),
        "norm_mean": _stat(norms, "mean"),
        "norm_std": _stat(norms, "std"),
        "norm_max": _stat(norms, "max"),
    }
    result.update(
        _near_duplicate_counts(
            finite_arr,
            sample_size=duplicate_sample_size,
            threshold=duplicate_cosine_threshold,
            near_zero_eps=near_zero_eps,
            seed=seed,
        )
    )
    from semoflow.metrics.tokenization import compute_cf_embedding_metrics
    from semoflow.metrics.tokenizer_diagnostics import (
        embedding_geometry_metrics,
        input_modality_alignment_metrics,
    )

    result.update(
        embedding_geometry_metrics(
            torch.from_numpy(finite_arr),
            max_rows=geometry_sample_size,
            seed=seed,
        )
    )
    if paired_embeddings is not None:
        paired = _as_float_matrix(paired_embeddings)
        logger.info(
            "computing paired embedding alignment  kind=%s  paired_kind=%s",
            kind,
            paired_kind,
        )
        result.update(
            input_modality_alignment_metrics(
                torch.from_numpy(arr),
                torch.from_numpy(paired),
                k=alignment_k,
                max_rows=alignment_sample_size,
            )
        )
    if index is not None and "top_category" in index.columns:
        result.update(
            _nearest_neighbor_category_purity(
                arr,
                index,
                k=purity_k,
                sample_size=purity_sample_size,
                chunk_size=purity_chunk_size,
                seed=seed,
            )
        )
    if (
        stays is not None
        and index is not None
        and "placekey_id" in index.columns
    ):
        logger.info(
            "computing behavior alignment metrics  k=%d  max_pois=%d",
            behavior_k,
            behavior_sample_size,
        )
        cf_metrics = compute_cf_embedding_metrics(
            torch.from_numpy(arr),
            index,
            stays,
            k=behavior_k,
            max_pois=behavior_sample_size,
            seed=seed,
        ).to_dict()
        for key, value in cf_metrics.items():
            if key in {"n_pois", "embedding_dim", "norm_mean", "norm_std"}:
                result[f"behavior_{key}"] = value
            else:
                result[key] = value
    return result


def _as_float_matrix(embeddings: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(embeddings, torch.Tensor):
        arr = embeddings.detach().cpu().float().numpy()
    else:
        arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings; got shape {arr.shape}")
    return arr


def _stat(values: np.ndarray, name: str) -> float | None:
    if values.size == 0:
        return None
    fn = getattr(np, name)
    return float(fn(values))


def _near_duplicate_counts(
    arr: np.ndarray,
    *,
    sample_size: int,
    threshold: float,
    near_zero_eps: float,
    seed: int,
) -> dict[str, Any]:
    valid_norms = np.linalg.norm(arr, axis=1) if len(arr) else np.array([])
    valid = valid_norms > near_zero_eps
    work = arr[valid]
    n = len(work)
    if n < 2:
        return {
            "duplicate_sample_size": int(n),
            "near_duplicate_row_count": 0,
            "near_duplicate_pair_count": 0,
            "duplicate_cosine_threshold": float(threshold),
        }
    if n > sample_size:
        rng = np.random.default_rng(seed)
        work = work[rng.choice(n, size=sample_size, replace=False)]
        n = sample_size
    norms = np.linalg.norm(work, axis=1, keepdims=True)
    unit = np.divide(work, norms, out=np.zeros_like(work), where=norms > 0)
    sim = unit @ unit.T
    upper = np.triu(sim >= threshold, k=1)
    duplicate_rows = np.where(upper.any(axis=0))[0]
    return {
        "duplicate_sample_size": int(n),
        "near_duplicate_row_count": len(duplicate_rows),
        "near_duplicate_pair_count": int(upper.sum()),
        "duplicate_cosine_threshold": float(threshold),
    }


def _nearest_neighbor_category_purity(
    arr: np.ndarray,
    index: pl.DataFrame,
    *,
    k: int,
    sample_size: int,
    chunk_size: int,
    seed: int,
) -> dict[str, Any]:
    n = min(arr.shape[0], index.height)
    if n < 2 or k < 1:
        return {
            "nn_category_purity": None,
            "nn_category_purity_k": int(k),
            "nn_category_purity_sample_size": 0,
        }
    cats = np.asarray(index["top_category"].head(n).to_list(), dtype=object)
    valid_categories = np.fromiter(
        (category is not None for category in cats), dtype=bool, count=n
    )
    valid_rows = np.isfinite(arr[:n]).all(axis=1) & valid_categories
    valid_idx = np.where(valid_rows)[0]
    if len(valid_idx) < 2:
        return {
            "nn_category_purity": None,
            "nn_category_purity_k": int(k),
            "nn_category_purity_sample_size": 0,
        }
    if len(valid_idx) > sample_size:
        rng = np.random.default_rng(seed)
        query_idx = rng.choice(valid_idx, size=sample_size, replace=False)
    else:
        query_idx = valid_idx
    if chunk_size <= 0:
        raise ValueError("purity_chunk_size must be positive")
    work = arr[valid_idx]
    norms = np.linalg.norm(work, axis=1, keepdims=True)
    unit = np.divide(work, norms, out=np.zeros_like(work), where=norms > 0)
    query_pos = np.searchsorted(valid_idx, query_idx)
    pos_lookup = {int(row): pos for pos, row in enumerate(valid_idx)}
    scores: list[float] = []
    logger.info(
        "computing NN category purity  queries=%d  candidates=%d  k=%d  chunk_size=%d",
        len(query_idx),
        len(valid_idx),
        k,
        chunk_size,
    )
    for start in range(0, len(query_idx), chunk_size):
        end = min(start + chunk_size, len(query_idx))
        chunk_query_idx = query_idx[start:end]
        chunk_pos = query_pos[start:end]
        sim = unit[chunk_pos] @ unit.T
        for out_row, orig_row in enumerate(chunk_query_idx):
            self_pos = pos_lookup[int(orig_row)]
            sim[out_row, self_pos] = -np.inf
            nn_pos = np.argsort(-sim[out_row])[: min(k, len(valid_idx) - 1)]
            target = cats[orig_row]
            nn_cats = cats[valid_idx[nn_pos]]
            scores.append(float(np.mean(nn_cats == target)))
    return {
        "nn_category_purity": float(np.mean(scores)) if scores else None,
        "nn_category_purity_k": int(k),
        "nn_category_purity_sample_size": len(scores),
    }


def write_embedding_eval(
    embeddings: str | Path,
    *,
    out_dir: str | Path,
    index_path: str | Path | None = None,
    paired_embeddings: str | Path | None = None,
    paired_kind: EmbeddingKind | str | None = None,
    processed_visits_path: str | Path | None = None,
    kind: EmbeddingKind | str | None = None,
) -> EmbeddingEvalPaths:
    result = evaluate_encoder_embeddings(
        load_embedding_matrix(embeddings),
        index=None if index_path is None else pl.read_parquet(index_path),
        paired_embeddings=(
            None
            if paired_embeddings is None
            else load_embedding_matrix(paired_embeddings)
        ),
        paired_kind=paired_kind,
        stays=(
            None
            if processed_visits_path is None
            else pl.read_parquet(processed_visits_path)
        ),
        kind=kind,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "embedding_eval.json"
    md_path = out / "embedding_eval.md"
    write_json(json_path, result)
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    logger.info(
        "wrote embedding eval  json=%s  markdown=%s",
        json_path,
        md_path,
    )
    return EmbeddingEvalPaths(json_path=json_path, md_path=md_path)


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Encoder Embedding Evaluation",
        "",
        f"- kind: `{result.get('kind')}`",
        f"- items: {result['n_items']}",
        f"- dimensions: {result['embedding_dim']}",
        f"- finite values: {result['finite_count']}",
        f"- NaN values: {result['nan_count']}",
        f"- Inf values: {result['inf_count']}",
        f"- non-finite rows: {result['nonfinite_row_count']}",
        f"- near-zero rows: {result['near_zero_row_count']}",
        f"- near-duplicate rows: {result['near_duplicate_row_count']}",
        "",
        "## Norms",
        "",
        f"- min: {result.get('norm_min')}",
        f"- mean: {result.get('norm_mean')}",
        f"- std: {result.get('norm_std')}",
        f"- max: {result.get('norm_max')}",
    ]
    if "nn_category_purity" in result:
        lines.extend(
            [
                "",
                "## Category Purity",
                "",
                f"- k: {result.get('nn_category_purity_k')}",
                f"- sample size: {result.get('nn_category_purity_sample_size')}",
                f"- purity: {result.get('nn_category_purity')}",
            ]
        )
    if "cosine_similarity_mean" in result:
        lines.extend(
            [
                "",
                "## Geometry",
                "",
                f"- cosine similarity mean: {result.get('cosine_similarity_mean')}",
                f"- cosine similarity std: {result.get('cosine_similarity_std')}",
                f"- anisotropy score: {result.get('anisotropy_score')}",
                f"- effective rank: {result.get('effective_rank')}",
                f"- embedding dim std mean: {result.get('embedding_dim_std_mean')}",
                f"- embedding dim std min: {result.get('embedding_dim_std_min')}",
            ]
        )
    if "text_graph_dim_match" in result:
        lines.extend(
            [
                "",
                "## Text-Graph Alignment",
                "",
                f"- dimension match: {result.get('text_graph_dim_match')}",
                f"- sample size: {result.get('text_graph_alignment_sample_size')}",
                f"- row cosine mean: {result.get('text_graph_cosine_mean')}",
                f"- row cosine std: {result.get('text_graph_cosine_std')}",
            ]
        )
        for key in sorted(result):
            if key.startswith("text_graph_retrieval_recall_at_"):
                lines.append(f"- retrieval recall: {result.get(key)}")
            if key.startswith("text_graph_mutual_nn_rate_at_"):
                lines.append(f"- mutual NN rate: {result.get(key)}")
            if key.startswith("text_graph_neighbor_overlap_at_"):
                lines.append(f"- neighbor overlap: {result.get(key)}")
    behavior_keys = [
        "transition_neighbor_recall_at_k",
        "covisit_neighbor_recall_at_k",
        "same_category_purity_at_k",
        "temporal_profile_similarity_at_k",
        "long_tail_transition_neighbor_recall_at_k",
    ]
    if any(key in result for key in behavior_keys):
        lines.extend(["", "## Behavior Alignment", ""])
        for key in behavior_keys:
            if key in result:
                lines.append(f"- {key}: {result.get(key)}")
    return "\n".join(lines) + "\n"


__all__ = [
    "EmbeddingEvalPaths",
    "EmbeddingKind",
    "evaluate_encoder_embeddings",
    "load_embedding_matrix",
    "write_embedding_eval",
]
