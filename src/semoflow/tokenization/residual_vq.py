from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = ["ResidualVQ", "sinkhorn_assign"]


@torch.no_grad()
def sinkhorn_assign(
    dist: torch.Tensor,
    epsilon: float,
    n_iters: int = 100,
) -> torch.Tensor:
    batch_size, codebook_size = dist.shape
    d_max = dist.max()
    d_min = dist.min()
    if torch.isclose(d_max, d_min):
        return dist.argmin(dim=1)
    middle = (d_max + d_min) / 2.0
    amplitude = d_max - middle + 1e-5
    centered = (dist - middle) / amplitude

    q = torch.exp(-centered.double() / epsilon)
    q /= q.sum()

    for _ in range(n_iters):
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
        ema_decay: float = 0.99,
        dead_threshold: int = 2,
        sk_epsilons: list[float] | None = None,
    ) -> None:
        super().__init__()
        if num_codebooks <= 0:
            raise ValueError("num_codebooks must be positive")
        if codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.dim = dim
        self.ema_decay = ema_decay
        self.dead_threshold = dead_threshold
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
        dead = usage < self.dead_threshold
        n_rows = residual.shape[0]
        if n_rows == 0:
            return

        replacement_rows = torch.randint(
            n_rows,
            (self.codebook_size,),
            device=residual.device,
        )
        replacements = residual[replacement_rows].detach()
        replacements = replacements + torch.randn_like(replacements) * 1e-3
        cb.data.copy_(torch.where(dead.unsqueeze(1), replacements, cb.data))
        usage.masked_fill_(dead, self.dead_threshold)

        if logger.isEnabledFor(logging.DEBUG):
            n_dead = int(dead.sum().detach().cpu().item())
            if n_dead > 0:
                logger.debug(
                    "ResidualVQ codebook %d: reinitialized %d entries",
                    cb_idx,
                    n_dead,
                )

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        if not self._initialized:
            self._init_from_data(z)

        residual = z
        q_sum = torch.zeros_like(z)
        code_list: list[torch.Tensor] = []
        quantized_per_cb: list[torch.Tensor] = []
        commit_loss = z.new_tensor(0.0)

        for i, cb in enumerate(self.codebooks):
            if self.training:
                self._reinit_dead_entries(i, cb, residual)

            dist = (
                residual.pow(2).sum(1, keepdim=True)
                - 2.0 * (residual @ cb.T)
                + cb.pow(2).sum(1)
            )
            sk_eps = self.sk_epsilons[i] if i < len(self.sk_epsilons) else 0.0
            if sk_eps > 0 and self.training:
                idx = sinkhorn_assign(dist, sk_eps)
            else:
                idx = dist.argmin(1)

            q = cb[idx]
            code_list.append(idx)
            quantized_per_cb.append(q)
            q_sum = q_sum + q
            commit_loss = commit_loss + F.mse_loss(residual, q.detach())

            if self.training:
                usage: torch.Tensor = getattr(self, f"_usage_{i}")
                usage.mul_(self.ema_decay).clamp_(min=0)
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
                updated = (
                    self.ema_decay * cb.data + (1 - self.ema_decay) * centroids
                )
                cb.data.copy_(torch.where(assigned, updated, cb.data))

            residual = residual - q.detach()

        q_st = z + (q_sum - z).detach()
        return q_st, torch.stack(code_list, 1), commit_loss, quantized_per_cb
