from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from semoflow.tokenization.residual_vq import ResidualVQ

if TYPE_CHECKING:
    from semoflow.config import TokenizerConfig

__all__ = ["RQVAE", "build_tokenizer_model"]


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


class RQVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        quantizer: ResidualVQ,
        decoder: nn.Module,
        *,
        commit_weight: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.n_codebooks = quantizer.num_codebooks
        self.codebook_size = quantizer.codebook_size
        self.commit_weight = commit_weight

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reconstruction, quantization and total losses for a batch."""
        x = x.float()
        z_q, _, vq_loss = self.quantizer(self.encoder(x))
        recon_loss = F.mse_loss(self.decoder(z_q), x)
        loss = recon_loss + self.commit_weight * vq_loss
        return {"recon_loss": recon_loss, "vq_loss": vq_loss, "loss": loss}

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """The [N, n_codebooks] semantic codes of each row."""
        was_training = self.training
        self.eval()
        codes = self.quantizer(self.encoder(x.float()))[1].long()
        self.train(was_training)
        return codes

    def get_codebooks(self) -> torch.Tensor:
        return torch.stack(list(self.quantizer.codebooks), dim=0)


def build_tokenizer_model(config: TokenizerConfig, *, input_dim: int) -> RQVAE:
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
    )
