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


_POSITION_BUCKET_STARTS = (1, 2, 3, 4, 5, 9, 17, 33)


def _relative_position_bucket(delta: torch.Tensor) -> torch.Tensor:
    starts = torch.tensor(_POSITION_BUCKET_STARTS, device=delta.device)
    return torch.bucketize(delta.clamp_min(0), starts, right=True)


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
        self.position_bias = nn.Embedding(
            len(_POSITION_BUCKET_STARTS) + 1, n_heads
        )
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

    def _bias(
        self,
        query_prefixes: torch.Tensor,
        key_prefixes: torch.Tensor,
        query_times: torch.Tensor,
        key_times: torch.Tensor,
        position_delta: torch.Tensor,
    ) -> torch.Tensor:
        query_valid = (query_prefixes >= 0).all(dim=-1)
        key_valid = (key_prefixes >= 0).all(dim=-1)
        relation_valid = query_valid.unsqueeze(-1) & key_valid.unsqueeze(-2)
        matches = query_prefixes.unsqueeze(-2) == key_prefixes.unsqueeze(-3)
        matches = matches & relation_valid.unsqueeze(-1)
        lcp = torch.cumprod(matches.to(torch.long), dim=-1).sum(dim=-1)
        time_delta = (
            query_times.unsqueeze(-1) - key_times.unsqueeze(-2)
        ).clamp_min(0.0)
        time_bucket = torch.bucketize(
            time_delta.float(), self.time_boundaries, right=False
        )
        position_bucket = _relative_position_bucket(position_delta).expand(
            query_prefixes.shape[0], -1, -1
        )
        spatial = self.lcp_bias(lcp)
        spatial = spatial * relation_valid.unsqueeze(-1).to(spatial.dtype)
        bias = self.position_bias(position_bucket) + spatial
        bias = bias + self.time_bias(time_bucket)
        return bias.permute(0, 3, 1, 2).contiguous()

    def forward(
        self, prefixes: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        positions = torch.arange(prefixes.shape[1], device=prefixes.device)
        delta = positions[:, None] - positions[None, :]
        return self._bias(prefixes, prefixes, times, times, delta)

    def step(
        self, prefixes: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        steps = prefixes.shape[1]
        delta = steps - 1 - torch.arange(steps, device=prefixes.device)
        return self._bias(
            prefixes[:, -1:], prefixes, times[:, -1:], times, delta[None, :]
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

    def _attend(
        self,
        hidden: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        relation_bias: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        scores = scores + relation_bias.to(dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        probabilities = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
        probabilities = F.dropout(
            probabilities, p=self.attn_dropout, training=self.training
        )
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).reshape(hidden.shape)
        hidden = hidden + self.resid_dropout(self.out(attended))
        return hidden + self.ffn(self.norm2(hidden))

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        relation_bias: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        steps = hidden.shape[1]
        positions = torch.arange(steps, device=hidden.device)
        query, key, value = self._project(self.norm1(hidden), positions)
        causal = torch.triu(
            torch.ones(steps, steps, dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        mask = causal[None, None] | (~valid_mask)[:, None, None, :]
        return self._attend(hidden, query, key, value, relation_bias, mask)

    def forward_step(
        self,
        hidden: torch.Tensor,
        *,
        active: torch.Tensor,
        cache: torch.Tensor,
        step: int,
        relation_bias: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.tensor([step], device=hidden.device)
        query, key, value = self._project(self.norm1(hidden), positions)
        cache[0, active, :, step] = key[:, :, 0]
        cache[1, active, :, step] = value[:, :, 0]
        return self._attend(
            hidden,
            query,
            cache[0, active, :, : step + 1],
            cache[1, active, :, : step + 1],
            relation_bias,
            None,
        )


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

    def forward_step(
        self,
        hidden: torch.Tensor,
        *,
        active: torch.Tensor,
        cache: torch.Tensor,
        relation_bias: torch.Tensor,
    ) -> torch.Tensor:
        step = relation_bias.shape[-1] - 1
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden = layer.forward_step(
                hidden,
                active=active,
                cache=layer_cache,
                step=step,
                relation_bias=relation_bias,
            )
        return self.final_norm(hidden)


__all__ = [
    "SpatioTemporalRelationBias",
    "SpatioTemporalTransformer",
]
