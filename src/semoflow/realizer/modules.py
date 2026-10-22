from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        half = freq_dim // 2
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half) / max(half - 1, 1)
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.freq_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class CircadianTokenEncoder(nn.Module):
    def __init__(
        self,
        time_vocab: int,
        output_dim: int,
        *,
        n_harmonics: int = 3,
        n_parts_of_day: int = 6,
    ) -> None:
        super().__init__()
        if time_vocab < 1 or output_dim < 1:
            raise ValueError("time_vocab and output_dim must be positive")
        if n_harmonics < 1 or n_parts_of_day < 2:
            raise ValueError("circadian feature dimensions are too small")
        self.time_vocab = time_vocab
        self.n_harmonics = n_harmonics
        self.n_parts_of_day = n_parts_of_day
        self.n_weekdays = max(1, math.ceil(time_vocab / n_parts_of_day))
        self.weekday_embedding = nn.Embedding(self.n_weekdays, output_dim)
        self.part_embedding = nn.Embedding(n_parts_of_day, output_dim)
        self.projection = nn.Sequential(
            nn.Linear(4 * n_harmonics, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim < 1:
            raise ValueError("time tokens must have at least one dimension")
        valid = (tokens >= 0) & (tokens < self.time_vocab)
        safe = tokens.clamp(min=0, max=self.time_vocab - 1)
        weekday = (safe // self.n_parts_of_day).clamp(max=self.n_weekdays - 1)
        part = safe % self.n_parts_of_day
        weekly_phase = 2.0 * math.pi * safe.float() / max(self.time_vocab, 1)
        daily_phase = 2.0 * math.pi * part.float() / self.n_parts_of_day
        harmonic_ids = torch.arange(
            1, self.n_harmonics + 1, device=tokens.device, dtype=tokens.dtype
        )
        weekly = weekly_phase[..., None] * harmonic_ids
        daily = daily_phase[..., None] * harmonic_ids
        periodic = torch.cat(
            [weekly.sin(), weekly.cos(), daily.sin(), daily.cos()], dim=-1
        )
        encoded = (
            self.weekday_embedding(weekday)
            + self.part_embedding(part)
            + self.projection(periodic)
        )
        return encoded.masked_fill(~valid[..., None], 0.0)


__all__ = ["CircadianTokenEncoder", "TimestepEmbedder"]
