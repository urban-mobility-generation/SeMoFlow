from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "embedding_geometry_metrics",
    "input_modality_alignment_metrics",
]


def _as_2d_float(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {tuple(x.shape)}")
    return x.detach().float().cpu()


def _finite_rows(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x).all(dim=1)


def _sample_rows_for_pairwise(
    x: torch.Tensor, *, max_rows: int, seed: int
) -> torch.Tensor:
    if x.shape[0] <= max_rows:
        return x
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=gen)[:max_rows]
    return x[idx]


def embedding_geometry_metrics(
    embeddings: torch.Tensor,
    *,
    max_rows: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    x = _as_2d_float(embeddings, name="embeddings")
    x = x[_finite_rows(x)]
    if x.shape[0] == 0:
        return {
            "cosine_similarity_mean": 0.0,
            "cosine_similarity_std": 0.0,
            "anisotropy_score": 0.0,
            "effective_rank": 0.0,
            "embedding_dim_std_mean": 0.0,
            "embedding_dim_std_min": 0.0,
        }
    xs = _sample_rows_for_pairwise(x, max_rows=max_rows, seed=seed)
    z = F.normalize(xs, dim=1)
    sim = z @ z.T
    if sim.shape[0] > 1:
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool)
        off_diag = sim[mask]
    else:
        off_diag = torch.empty(0)
    centered = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(centered)
    if s.numel() == 0 or float(s.sum().item()) <= 0:
        effective_rank = 0.0
    else:
        probs = s / s.sum()
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
        effective_rank = float(entropy.exp().item())
    dim_std = x.std(dim=0, unbiased=False)
    mean_vec = F.normalize(x, dim=1).mean(dim=0)
    return {
        "cosine_similarity_mean": (
            float(off_diag.mean().item()) if off_diag.numel() else 0.0
        ),
        "cosine_similarity_std": (
            float(off_diag.std(unbiased=False).item())
            if off_diag.numel()
            else 0.0
        ),
        "anisotropy_score": float(mean_vec.norm().item()),
        "effective_rank": effective_rank,
        "embedding_dim_std_mean": float(dim_std.mean().item()),
        "embedding_dim_std_min": float(dim_std.min().item()),
    }


def _topk_indices(embeddings: torch.Tensor, *, k: int) -> torch.Tensor:
    if embeddings.shape[0] <= 1:
        return torch.empty((embeddings.shape[0], 0), dtype=torch.long)
    kk = min(k, embeddings.shape[0] - 1)
    z = F.normalize(embeddings, dim=1)
    sim = z @ z.T
    sim.fill_diagonal_(float("-inf"))
    return sim.topk(kk, dim=1).indices


def _neighbor_sets(embeddings: torch.Tensor, *, k: int) -> list[set[int]]:
    top = _topk_indices(embeddings, k=k)
    return [set(int(v) for v in row.tolist()) for row in top]


def input_modality_alignment_metrics(
    text_embeddings: torch.Tensor,
    graph_embeddings: torch.Tensor,
    *,
    k: int = 10,
    max_rows: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    text = _as_2d_float(text_embeddings, name="text_embeddings")
    graph = _as_2d_float(graph_embeddings, name="graph_embeddings")
    if text.shape[0] != graph.shape[0]:
        raise ValueError("text and graph embeddings must share row count")
    valid = _finite_rows(text) & _finite_rows(graph)
    text = text[valid]
    graph = graph[valid]
    if text.shape[0] > max_rows:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        idx = torch.randperm(text.shape[0], generator=gen)[:max_rows]
        text = text[idx]
        graph = graph[idx]
    if text.shape[0] == 0:
        return {
            "text_graph_cosine_mean": None,
            "text_graph_cosine_std": None,
            f"text_graph_retrieval_recall_at_{k}": None,
            f"text_graph_mutual_nn_rate_at_{k}": None,
            f"text_graph_neighbor_overlap_at_{k}": None,
            "text_graph_alignment_sample_size": 0,
        }

    kk = min(k, max(text.shape[0] - 1, 1))
    text_neighbors = _neighbor_sets(text, k=kk)
    graph_neighbors = _neighbor_sets(graph, k=kk)
    overlaps = [
        len(t & g) / max(len(t), 1)
        for t, g in zip(text_neighbors, graph_neighbors, strict=True)
        if t
    ]
    result: dict[str, Any] = {
        "text_graph_cosine_mean": None,
        "text_graph_cosine_std": None,
        f"text_graph_retrieval_recall_at_{k}": None,
        f"text_graph_mutual_nn_rate_at_{k}": None,
        f"text_graph_neighbor_overlap_at_{k}": (
            float(np.mean(overlaps)) if overlaps else None
        ),
        "text_graph_alignment_sample_size": int(text.shape[0]),
    }
    if text.shape[1] != graph.shape[1]:
        result["text_graph_dim_match"] = False
        return result

    result["text_graph_dim_match"] = True
    text_z = F.normalize(text, dim=1)
    graph_z = F.normalize(graph, dim=1)
    row_cos = (text_z * graph_z).sum(dim=1)
    sim = text_z @ graph_z.T
    cross_k = min(k, sim.shape[1])
    text_to_graph = sim.topk(cross_k, dim=1).indices
    graph_to_text = sim.T.topk(cross_k, dim=1).indices
    row_ids = torch.arange(sim.shape[0])
    recall = (text_to_graph == row_ids[:, None]).any(dim=1).float().mean()
    mutual = [
        bool((text_to_graph[i] == i).any() and (graph_to_text[i] == i).any())
        for i in range(sim.shape[0])
    ]
    result.update(
        {
            "text_graph_cosine_mean": float(row_cos.mean().item()),
            "text_graph_cosine_std": float(row_cos.std(unbiased=False).item()),
            f"text_graph_retrieval_recall_at_{k}": float(recall.item()),
            f"text_graph_mutual_nn_rate_at_{k}": float(np.mean(mutual)),
        }
    )
    return result
