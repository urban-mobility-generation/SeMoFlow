from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from semoflow.tokenization.residual_vq import ResidualVQ

if TYPE_CHECKING:
    from semoflow.config.schemas.tokenizer_config import TokenizerConfig

__all__ = ["RQVAE", "SemanticIDOutput", "build_tokenizer_model"]


@dataclass
class SemanticIDOutput:
    loss: torch.Tensor | None
    codes: torch.Tensor
    losses: dict[str, torch.Tensor] | None = None


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class ResMLPBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class ResMLPEncoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int
    ) -> None:
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResMLPBlock(hidden_dim) for _ in range(n_layers)]
        )
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h)
        return self.proj_out(h)


class MLPDecoder(nn.Module):
    def __init__(
        self, *, latent_dim: int, output_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        self.net = _mlp(latent_dim, hidden_dim, output_dim)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.net(z_q)


def graph_cosine_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    valid = (torch.linalg.vector_norm(target.float(), dim=1) > 1e-8).to(
        pred.dtype
    )
    cosine = F.cosine_similarity(
        F.normalize(pred, dim=1), F.normalize(target, dim=1), dim=1
    )
    return ((1.0 - cosine) * valid).sum() / valid.sum().clamp_min(1.0)


class RQVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        quantizer: ResidualVQ,
        decoder: nn.Module,
        *,
        commit_weight: float,
        graph_projector: nn.Module | None = None,
        lambda_graph: float = 0.0,
        lambda_prefix_graph: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.graph_projector = graph_projector
        self.n_codebooks = quantizer.num_codebooks
        self.codebook_size = quantizer.codebook_size
        self.commit_weight = commit_weight
        self.lambda_graph = lambda_graph
        self.lambda_prefix_graph = lambda_prefix_graph

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_loss: bool = True,
    ) -> SemanticIDOutput:
        x = batch["x"].float()
        z = self.encoder(x)
        z_q, codes, vq_loss, quantized = self.quantizer(z)
        if not return_loss:
            return SemanticIDOutput(loss=None, codes=codes.long())

        recon_loss = F.mse_loss(self.decoder(z_q), x)
        loss = recon_loss + self.commit_weight * vq_loss
        losses = {"recon_loss": recon_loss, "vq_loss": vq_loss}

        target = batch.get("graph_target")
        if self.graph_projector is not None and target is not None:
            target = target.float()
            if self.lambda_graph > 0:
                graph_loss = graph_cosine_loss(
                    self.graph_projector(z_q), target
                )
                loss = loss + self.lambda_graph * graph_loss
                losses["graph_loss"] = graph_loss
            if self.lambda_prefix_graph > 0:
                prefix_sum = torch.zeros_like(z)
                prefix_losses = []
                for q in quantized:
                    prefix_sum = prefix_sum + q
                    prefix_st = z + (prefix_sum - z).detach()
                    prefix_losses.append(
                        graph_cosine_loss(
                            self.graph_projector(prefix_st), target
                        )
                    )
                prefix_graph_loss = torch.stack(prefix_losses).mean()
                loss = loss + self.lambda_prefix_graph * prefix_graph_loss
                losses["prefix_graph_loss"] = prefix_graph_loss

        losses["loss"] = loss
        return SemanticIDOutput(loss=loss, codes=codes.long(), losses=losses)

    @torch.no_grad()
    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        was_training = self.training
        self.eval()
        codes = self.forward(batch, return_loss=False).codes
        self.train(was_training)
        return codes

    def get_codebooks(self) -> torch.Tensor:
        return torch.stack(list(self.quantizer.codebooks), dim=0)


def build_tokenizer_model(
    config: TokenizerConfig,
    *,
    input_dim: int,
    graph_dim: int | None = None,
) -> RQVAE:
    graph_active = config.lambda_graph > 0 or config.lambda_prefix_graph > 0
    return RQVAE(
        ResMLPEncoder(
            input_dim,
            config.hidden_dim,
            config.latent_dim,
            config.n_encoder_layers,
        ),
        ResidualVQ(
            config.n_codebooks,
            config.codebook_size,
            config.latent_dim,
            sk_epsilons=config.sk_epsilons,
        ),
        MLPDecoder(
            latent_dim=config.latent_dim,
            output_dim=input_dim,
            hidden_dim=config.hidden_dim,
        ),
        commit_weight=config.beta,
        graph_projector=(
            _mlp(config.latent_dim, config.hidden_dim, graph_dim)
            if graph_active and graph_dim is not None
            else None
        ),
        lambda_graph=config.lambda_graph,
        lambda_prefix_graph=config.lambda_prefix_graph,
    )
