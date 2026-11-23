from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from semoflow.config import TokenizerConfig

_EMA_DECAY = 0.99
_DEAD_THRESHOLD = 2
_SINKHORN_ITERATIONS = 100


@torch.no_grad()
def sinkhorn_assign(dist: torch.Tensor, epsilon: float) -> torch.Tensor:
    batch_size, codebook_size = dist.shape
    d_max = dist.max()
    d_min = dist.min()
    middle = (d_max + d_min) / 2.0
    amplitude = d_max - middle + 1e-5
    centered = (dist - middle) / amplitude

    q = torch.exp(-centered.double() / epsilon)
    q /= q.sum()

    for _ in range(_SINKHORN_ITERATIONS):
        q /= q.sum(dim=1, keepdim=True)
        q /= batch_size
        q /= q.sum(dim=0, keepdim=True)
        q /= codebook_size

    q *= batch_size
    return q.argmax(dim=1)


class ResidualVQ(nn.Module):
    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        dim: int,
        *,
        sk_epsilons: list[float] | None,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.sk_epsilons = sk_epsilons or [0.0] * num_codebooks

        self.codebooks = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(codebook_size, dim), requires_grad=False
                )
                for _ in range(num_codebooks)
            ]
        )
        for i in range(num_codebooks):
            self.register_buffer(
                f"_usage_{i}",
                torch.zeros(codebook_size, dtype=torch.float32),
            )
        self.register_buffer(
            "_initialized",
            torch.tensor(False, dtype=torch.bool),
        )

    def _init_from_data(self, z: torch.Tensor) -> None:
        with torch.no_grad():
            residual = z.detach()
            n_rows = residual.shape[0]
            for cb in self.codebooks:
                if n_rows >= self.codebook_size:
                    idx = torch.randperm(n_rows, device=z.device)[
                        : self.codebook_size
                    ]
                else:
                    idx = torch.randint(
                        n_rows,
                        (self.codebook_size,),
                        device=z.device,
                    )
                cb.copy_(residual[idx])
                cb.add_(torch.randn_like(cb) * 1e-3)

                dist = (
                    residual.pow(2).sum(1, keepdim=True)
                    - 2.0 * (residual @ cb.T)
                    + cb.pow(2).sum(1)
                )
                nearest = dist.argmin(1)
                residual = residual - cb[nearest].detach()
        self._initialized.fill_(True)

    def _reinit_dead_entries(
        self,
        cb_idx: int,
        cb: nn.Parameter,
        residual: torch.Tensor,
    ) -> None:
        usage: torch.Tensor = getattr(self, f"_usage_{cb_idx}")
        dead = usage < _DEAD_THRESHOLD
        replacement_rows = torch.randint(
            residual.shape[0],
            (self.codebook_size,),
            device=residual.device,
        )
        replacements = residual[replacement_rows].detach()
        replacements = replacements + torch.randn_like(replacements) * 1e-3
        cb.data.copy_(torch.where(dead.unsqueeze(1), replacements, cb.data))
        usage.masked_fill_(dead, _DEAD_THRESHOLD)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._initialized:
            self._init_from_data(z)

        residual = z
        q_sum = torch.zeros_like(z)
        code_list: list[torch.Tensor] = []
        commit_loss = z.new_tensor(0.0)

        for i, cb in enumerate(self.codebooks):
            if self.training:
                self._reinit_dead_entries(i, cb, residual)

            dist = (
                residual.pow(2).sum(1, keepdim=True)
                - 2.0 * (residual @ cb.T)
                + cb.pow(2).sum(1)
            )
            sk_eps = self.sk_epsilons[i]
            if sk_eps > 0 and self.training:
                idx = sinkhorn_assign(dist, sk_eps)
            else:
                idx = dist.argmin(1)

            q = cb[idx]
            code_list.append(idx)
            q_sum = q_sum + q
            commit_loss = commit_loss + F.mse_loss(residual, q.detach())

            if self.training:
                usage: torch.Tensor = getattr(self, f"_usage_{i}")
                usage.mul_(_EMA_DECAY).clamp_(min=0)
                usage.add_(
                    torch.bincount(idx, minlength=self.codebook_size).float()
                )

                onehot = F.one_hot(idx, self.codebook_size).float()
                raw_counts = onehot.sum(0)
                counts = raw_counts.clamp(min=1)
                centroids = (onehot.T @ residual.detach()) / counts.unsqueeze(
                    1
                )
                assigned = (raw_counts > 0).unsqueeze(1)
                updated = _EMA_DECAY * cb.data + (1 - _EMA_DECAY) * centroids
                cb.data.copy_(torch.where(assigned, updated, cb.data))

            residual = residual - q.detach()

        q_st = z + (q_sum - z).detach()
        return q_st, torch.stack(code_list, 1), commit_loss


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
        self.commit_weight = commit_weight

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.float()
        z_q, _, vq_loss = self.quantizer(self.encoder(x))
        recon_loss = F.mse_loss(self.decoder(z_q), x)
        loss = recon_loss + self.commit_weight * vq_loss
        return {"recon_loss": recon_loss, "vq_loss": vq_loss, "loss": loss}

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantizer(self.encoder(x.float()))[1]

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
        nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, input_dim),
        ),
        commit_weight=config.beta,
    )


__all__ = [
    "RQVAE",
    "build_tokenizer_model",
]
