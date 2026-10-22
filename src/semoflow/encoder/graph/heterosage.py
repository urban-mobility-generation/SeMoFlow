from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from semoflow.encoder.graph.layers import (
    build_id_embeddings,
    build_input_projections,
    intermediate_activation,
    project_inputs,
)


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
        self.id_embedding_dim = int(id_embedding_dim)
        self.id_embeddings = build_id_embeddings(
            self.id_embedding_dim, id_embedding_counts
        )
        self.lin_dict = build_input_projections(
            in_dims, hidden_dim, self.id_embeddings, self.id_embedding_dim
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            out_channels = hidden_dim if i < num_layers - 1 else out_dim
            self.convs.append(
                HeteroConv(
                    {
                        # SAGEConv does not use the optional edge_update hook.
                        edge_type: SAGEConv(  # ty: ignore[call-non-callable]
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
        x_out = project_inputs(
            x_dict, n_id_dict, self.id_embeddings, self.lin_dict
        )
        for i, conv in enumerate(self.convs):
            previous = x_out
            x_out = conv(x_out, edge_index_dict)
            if i < self.num_layers - 1:
                x_out = intermediate_activation(
                    x_out,
                    previous,
                    norm=self.norms[i],
                    residual=self.residual,
                    dropout=self.dropout,
                    training=self.training,
                )
        return x_out
