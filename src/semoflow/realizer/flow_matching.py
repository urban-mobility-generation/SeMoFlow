from __future__ import annotations

import torch


def sample_uniform_timesteps(
    batch_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.rand(batch_size, device=device)


def sample_logit_normal_timesteps(
    batch_size: int,
    device: torch.device | str,
    *,
    p_mean: float = -0.5,
    p_std: float = 1.2,
) -> torch.Tensor:
    z = torch.randn(batch_size, device=device) * p_std + p_mean
    return torch.sigmoid(z)


def linear_interpolate(
    z0: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    t_bc = t[:, None, None]
    return (1.0 - t_bc) * z0 + t_bc * eps


def compute_target_velocity(
    z0: torch.Tensor,
    eps: torch.Tensor,
) -> torch.Tensor:
    return eps - z0


def velocity_from_x0(
    zt: torch.Tensor,
    x0_pred: torch.Tensor,
    t: torch.Tensor,
    *,
    t_eps: float = 1e-6,
) -> torch.Tensor:
    t_bc = t[:, None, None].clamp_min(t_eps)
    return (zt - x0_pred) / t_bc


def euler_step(
    z: torch.Tensor,
    v: torch.Tensor,
    dt: float | torch.Tensor,
) -> torch.Tensor:
    return z + dt * v


def heun_step(
    z: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    dt: float | torch.Tensor,
) -> torch.Tensor:
    return z + dt * (v1 + v2) * 0.5


__all__ = [
    "compute_target_velocity",
    "euler_step",
    "heun_step",
    "linear_interpolate",
    "sample_logit_normal_timesteps",
    "sample_uniform_timesteps",
    "velocity_from_x0",
]
