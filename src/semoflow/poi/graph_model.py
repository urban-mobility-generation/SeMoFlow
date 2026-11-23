from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroSAGE(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_dims: dict[str, int],
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        *,
        aggr: str = "sum",
        layer_norm: bool = False,
        residual: bool = False,
        id_embedding_dim: int = 0,
        id_embedding_counts: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.id_embeddings = nn.ModuleDict()
        if id_embedding_dim > 0:
            assert id_embedding_counts is not None
            for node_type, count in id_embedding_counts.items():
                table = nn.Embedding(int(count), id_embedding_dim)
                nn.init.normal_(table.weight, std=0.01)
                self.id_embeddings[node_type] = table
        self.lin_dict = nn.ModuleDict(
            {
                node_type: nn.Linear(
                    in_dim
                    + (
                        id_embedding_dim
                        if node_type in self.id_embeddings
                        else 0
                    ),
                    hidden_dim,
                )
                for node_type, in_dim in in_dims.items()
            }
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            out_channels = hidden_dim if i < num_layers - 1 else out_dim
            self.convs.append(
                HeteroConv(
                    {
                        edge_type: SAGEConv(
                            (hidden_dim, hidden_dim), out_channels
                        )
                        for edge_type in metadata[1]
                    },
                    aggr=aggr,
                )
            )
            self.norms.append(
                nn.LayerNorm(out_channels)
                if layer_norm and i < num_layers - 1
                else nn.Identity()
            )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        n_id_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        features = dict(x_dict)
        if self.id_embeddings:
            assert n_id_dict is not None
            for node_type, table in self.id_embeddings.items():
                if node_type in features:
                    features[node_type] = torch.cat(
                        [features[node_type], table(n_id_dict[node_type])],
                        dim=-1,
                    )
        x_out = {
            node_type: self.lin_dict[node_type](x)
            for node_type, x in features.items()
        }
        for i, conv in enumerate(self.convs):
            previous = x_out
            x_out = conv(x_out, edge_index_dict)
            if i == self.num_layers - 1:
                break
            for node_type, x in x_out.items():
                x = self.norms[i](x)
                skip = previous.get(node_type)
                if (
                    self.residual
                    and skip is not None
                    and skip.shape == x.shape
                ):
                    x = x + skip
                x_out[node_type] = F.dropout(
                    F.relu(x), p=self.dropout, training=self.training
                )
        return x_out


def variance_covariance_penalty(
    poi_emb: torch.Tensor,
    *,
    variance_target: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
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


__all__ = [
    "HeteroSAGE",
    "contrastive_link_loss",
    "variance_covariance_penalty",
]
