from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "contrastive_link_loss",
    "variance_covariance_penalty",
]


def variance_covariance_penalty(
    poi_emb: torch.Tensor,
    *,
    variance_target: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if poi_emb.ndim != 2:
        raise ValueError("poi_emb must be a two-dimensional [N, D] tensor")
    n_rows, n_dims = poi_emb.shape
    if n_rows < 2:
        zero = poi_emb.sum() * 0.0
        return zero, zero

    centered = poi_emb - poi_emb.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0) + eps)
    variance_loss = F.relu(variance_target - std).mean()

    covariance = (centered.T @ centered) / (n_rows - 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    covariance_loss = off_diagonal.pow(2).sum() / n_dims
    return variance_loss, covariance_loss


def contrastive_link_loss(
    poi_emb: torch.Tensor,
    pos_edges: tuple[torch.Tensor, torch.Tensor],
    *,
    neg_edges: tuple[torch.Tensor, torch.Tensor] | None = None,
    temperature: float = 0.07,
    max_negatives: int = 2048,
    max_queries: int = 65536,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if max_negatives < 1:
        raise ValueError("max_negatives must be positive")
    if max_queries < 1:
        raise ValueError("max_queries must be positive")
    source, target = pos_edges
    if source.numel() == 0:
        return poi_emb.sum() * 0.0

    n_edges = int(source.shape[0])
    if n_edges > max_queries:
        keep = torch.randperm(n_edges, device=poi_emb.device)[:max_queries]
        source, target = source[keep], target[keep]

    query = F.normalize(poi_emb[source].float(), dim=-1)
    key = F.normalize(poi_emb[target].float(), dim=-1)
    n_queries = query.shape[0]

    positive_logits = (query * key).sum(dim=-1, keepdim=True) / temperature

    explicit_target = None
    matched = None
    if neg_edges is not None and neg_edges[0].numel() > 0:
        negative_source, negative_target = neg_edges
        if negative_source.shape != negative_target.shape:
            raise ValueError("negative edge source and target shapes differ")
        order = torch.argsort(negative_source)
        sorted_source = negative_source[order]
        positions = torch.searchsorted(sorted_source, source)
        clamped = positions.clamp(max=sorted_source.numel() - 1)
        matched = (positions < sorted_source.numel()) & (
            sorted_source[clamped] == source
        )

        explicit_target = target.clone()
        explicit_target[matched] = negative_target[order[clamped[matched]]]

    n_negatives = min(int(max_negatives), n_queries)
    negative_index = torch.randint(
        n_queries, (n_negatives,), device=query.device
    )
    negative_logits = (query @ key[negative_index].T) / temperature
    negative_targets = target[negative_index]
    if (
        explicit_target is not None
        and matched is not None
        and bool(matched.any())
    ):
        explicit_logits = (
            query[matched].float()
            * F.normalize(poi_emb[explicit_target[matched]].float(), dim=-1)
        ).sum(dim=-1) / temperature
        negative_logits[matched, 0] = explicit_logits.to(negative_logits.dtype)
        negative_targets = (
            negative_targets.unsqueeze(0).expand(n_queries, -1).clone()
        )
        negative_targets[matched, 0] = explicit_target[matched]
    else:
        negative_targets = negative_targets.unsqueeze(0).expand(n_queries, -1)
    collides = negative_targets == target.unsqueeze(1)
    negative_logits = negative_logits.masked_fill(collides, float("-inf"))

    logits = torch.cat([positive_logits, negative_logits], dim=1)
    labels = torch.zeros(n_queries, dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, labels)
