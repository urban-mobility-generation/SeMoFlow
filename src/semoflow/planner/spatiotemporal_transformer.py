from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.weight


class SwiGLU(nn.Module):
    def __init__(
        self, model_dim: int, hidden_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(model_dim, hidden_dim, bias=False)
        self.up = nn.Linear(model_dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate(x)) * self.up(x)
        return self.dropout(self.down(hidden))


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        inv_freq = 1.0 / (
            10_000.0
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def at(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[positions], self.sin[positions]


def _apply_rope(
    values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    even = values[..., 0::2]
    odd = values[..., 1::2]
    while cos.ndim < even.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def _relative_position_bucket(delta: torch.Tensor) -> torch.Tensor:
    delta = delta.clamp_min(0)
    result = torch.zeros_like(delta, dtype=torch.long)
    result = torch.where(delta == 1, 1, result)
    result = torch.where(delta == 2, 2, result)
    result = torch.where(delta == 3, 3, result)
    result = torch.where(delta == 4, 4, result)
    result = torch.where((delta >= 5) & (delta <= 8), 5, result)
    result = torch.where((delta >= 9) & (delta <= 16), 6, result)
    result = torch.where((delta >= 17) & (delta <= 32), 7, result)
    result = torch.where(delta >= 33, 8, result)
    return result


_TIME_BUCKET_BOUNDARIES_MINUTES = (
    0.0,
    10.0,
    30.0,
    60.0,
    120.0,
    240.0,
    480.0,
    960.0,
    1440.0,
    2880.0,
    4320.0,
    7200.0,
    10080.0,
)


class SpatioTemporalRelationBias(nn.Module):
    def __init__(self, *, n_heads: int, prefix_depth: int) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.prefix_depth = int(prefix_depth)
        self.position_bias = nn.Embedding(9, n_heads)
        self.time_bias = nn.Embedding(
            len(_TIME_BUCKET_BOUNDARIES_MINUTES) + 1, n_heads
        )
        self.lcp_bias = nn.Embedding(prefix_depth + 1, n_heads)
        nn.init.zeros_(self.position_bias.weight)
        nn.init.zeros_(self.time_bias.weight)
        nn.init.zeros_(self.lcp_bias.weight)
        self.register_buffer(
            "time_boundaries",
            torch.tensor(_TIME_BUCKET_BOUNDARIES_MINUTES, dtype=torch.float32),
            persistent=False,
        )

    def _lcp(
        self,
        query_prefixes: torch.Tensor,
        key_prefixes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_valid = (query_prefixes >= 0).all(dim=-1)
        key_valid = (key_prefixes >= 0).all(dim=-1)
        relation_valid = query_valid.unsqueeze(-1) & key_valid.unsqueeze(-2)
        matches = query_prefixes.unsqueeze(-2) == key_prefixes.unsqueeze(-3)
        matches = matches & relation_valid.unsqueeze(-1)
        prefix_matches = torch.cumprod(matches.to(torch.long), dim=-1)
        return prefix_matches.sum(dim=-1), relation_valid

    def _time_bucket(
        self,
        query_times: torch.Tensor,
        key_times: torch.Tensor,
    ) -> torch.Tensor:
        delta = (
            query_times.unsqueeze(-1) - key_times.unsqueeze(-2)
        ).clamp_min(0.0)
        boundaries = self.time_boundaries.to(device=delta.device)
        return torch.bucketize(delta.float(), boundaries, right=False)

    def _combine(
        self,
        position_bucket: torch.Tensor,
        lcp: torch.Tensor,
        spatial_valid: torch.Tensor,
        time_bucket: torch.Tensor | None,
    ) -> torch.Tensor:
        spatial = self.lcp_bias(lcp)
        spatial = spatial * spatial_valid.unsqueeze(-1).to(spatial.dtype)
        bias = self.position_bias(position_bucket) + spatial
        if time_bucket is not None:
            bias = bias + self.time_bias(time_bucket)
        return bias.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        prefixes: torch.Tensor,
        times: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, steps, _ = prefixes.shape
        positions = torch.arange(steps, device=prefixes.device)
        delta = positions[:, None] - positions[None, :]
        position_bucket = _relative_position_bucket(delta)
        position_bucket = position_bucket.expand(batch, -1, -1)
        lcp, spatial_valid = self._lcp(prefixes, prefixes)
        time_bucket = (
            None if times is None else self._time_bucket(times, times)
        )
        return self._combine(
            position_bucket,
            lcp,
            spatial_valid,
            time_bucket,
        )

    def step(
        self,
        prefixes: torch.Tensor,
        times: torch.Tensor | None,
        *,
        step: int,
    ) -> torch.Tensor:
        keys = prefixes[:, : step + 1]
        query = prefixes[:, step : step + 1]
        key_positions = torch.arange(step + 1, device=prefixes.device)
        delta = step - key_positions
        position_bucket = _relative_position_bucket(delta)
        position_bucket = position_bucket.view(1, 1, -1).expand(
            prefixes.shape[0], 1, -1
        )
        lcp, spatial_valid = self._lcp(query, keys)
        time_bucket = None
        if times is not None:
            time_bucket = self._time_bucket(
                times[:, step : step + 1],
                times[:, : step + 1],
            )
        return self._combine(
            position_bucket,
            lcp,
            spatial_valid,
            time_bucket,
        )


class SpatioTemporalTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        if model_dim % n_heads:
            raise ValueError("model_dim must be divisible by n_heads")
        self.model_dim = int(model_dim)
        self.n_heads = int(n_heads)
        self.head_dim = model_dim // n_heads
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.out = nn.Linear(model_dim, model_dim, bias=False)
        hidden_dim = max(1, int(math.ceil((8.0 / 3.0) * model_dim / 8.0) * 8))
        self.ffn = SwiGLU(model_dim, hidden_dim, dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        self.attn_dropout = float(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = tensor.shape
        return tensor.view(
            batch, steps, self.n_heads, self.head_dim
        ).transpose(1, 2)

    def _project(
        self,
        normalized: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        query = self._split(query)
        key = self._split(key)
        value = self._split(value)
        cos, sin = self.rope.at(positions)
        query = _apply_rope(query, cos, sin)
        key = _apply_rope(key, cos, sin)
        return query, key, value

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        relation_bias: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, _ = hidden.shape
        normalized = self.norm1(hidden)
        positions = torch.arange(steps, device=hidden.device)
        query, key, value = self._project(normalized, positions)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        scores = scores + relation_bias.to(dtype=scores.dtype)
        causal = torch.triu(
            torch.ones(steps, steps, dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal[None, None], float("-inf"))
        scores = scores.masked_fill(
            (~valid_mask)[:, None, None, :], float("-inf")
        )
        probabilities = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
        probabilities = F.dropout(
            probabilities,
            p=self.attn_dropout,
            training=self.training,
        )
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).reshape(
            batch, steps, self.model_dim
        )
        hidden = hidden + self.resid_dropout(self.out(attended))
        hidden = hidden + self.ffn(self.norm2(hidden))
        return hidden

    def forward_step(
        self,
        hidden: torch.Tensor,
        *,
        step: int,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        relation_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.norm1(hidden)
        positions = torch.tensor([step], device=hidden.device)
        query, current_key, current_value = self._project(
            normalized, positions
        )
        key = torch.cat((past_key, current_key), dim=2)
        value = torch.cat((past_value, current_value), dim=2)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        scores = scores + relation_bias.to(dtype=scores.dtype)
        probabilities = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).reshape(
            hidden.shape[0], 1, self.model_dim
        )
        hidden = hidden + self.resid_dropout(self.out(attended))
        hidden = hidden + self.ffn(self.norm2(hidden))
        return hidden, current_key, current_value


class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SpatioTemporalTransformerLayer(
                    model_dim=model_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(model_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        relation_bias: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(
                hidden,
                relation_bias=relation_bias,
                valid_mask=valid_mask,
            )
        return self.final_norm(hidden)


__all__ = [
    "RMSNorm",
    "SpatioTemporalRelationBias",
    "SpatioTemporalTransformer",
    "SpatioTemporalTransformerLayer",
    "SwiGLU",
]
