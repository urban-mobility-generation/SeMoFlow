from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def build_id_embeddings(
    dim: int, counts: dict[str, int] | None
) -> nn.ModuleDict:
    tables = nn.ModuleDict()
    if dim <= 0:
        return tables
    if not counts:
        raise ValueError(
            "id_embedding_counts is required when id_embedding_dim is positive"
        )
    for node_type, count in counts.items():
        table = nn.Embedding(int(count), dim)
        nn.init.normal_(table.weight, std=0.01)
        tables[node_type] = table
    return tables


def build_input_projections(
    in_dims: dict[str, int],
    hidden_dim: int,
    id_embeddings: nn.ModuleDict,
    id_embedding_dim: int,
) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            node_type: nn.Linear(
                in_dim
                + (id_embedding_dim if node_type in id_embeddings else 0),
                hidden_dim,
            )
            for node_type, in_dim in in_dims.items()
        }
    )


def project_inputs(
    x_dict: dict[str, torch.Tensor],
    n_id_dict: dict[str, torch.Tensor] | None,
    id_embeddings: nn.ModuleDict,
    lin_dict: nn.ModuleDict,
) -> dict[str, torch.Tensor]:
    features = dict(x_dict)
    if id_embeddings:
        if n_id_dict is None:
            raise ValueError(
                "n_id_dict is required when id_embedding_dim is positive"
            )
        for node_type, table in id_embeddings.items():
            if node_type not in features:
                continue
            if node_type not in n_id_dict:
                raise ValueError(
                    f"n_id_dict is missing node type {node_type!r}"
                )
            features[node_type] = torch.cat(
                [features[node_type], table(n_id_dict[node_type])], dim=-1
            )
    return {
        node_type: lin_dict[node_type](x) for node_type, x in features.items()
    }


def intermediate_activation(
    x_dict: dict[str, torch.Tensor],
    previous: dict[str, torch.Tensor],
    *,
    norm: nn.Module,
    residual: bool,
    dropout: float,
    training: bool,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for node_type, x in x_dict.items():
        x = norm(x)
        skip = previous.get(node_type)
        if residual and skip is not None and skip.shape == x.shape:
            x = x + skip
        out[node_type] = F.dropout(F.relu(x), p=dropout, training=training)
    return out


__all__ = [
    "build_id_embeddings",
    "build_input_projections",
    "intermediate_activation",
    "project_inputs",
]
