from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from semoflow.tokenization.sids import find_sorted, packed_code_keys


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


class LegalPrefixIndex(nn.Module):
    def __init__(
        self,
        legal_prefixes: torch.Tensor,
        *,
        codebook_size: int,
        prefix_depth: int,
        city_legal_prefix_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        codes = legal_prefixes.long()
        self.codebook_size = int(codebook_size)
        self.prefix_depth = int(prefix_depth)
        self.register_buffer("codes", codes.clone(), persistent=False)

        city_mask = city_legal_prefix_mask.bool()

        flat = self._flatten(codes)
        sorted_flat_ids, token_ids = torch.sort(flat)
        self.register_buffer(
            "sorted_flat_ids", sorted_flat_ids, persistent=False
        )
        self.register_buffer(
            "token_ids_by_sorted_flat", token_ids, persistent=False
        )

        for level in range(prefix_depth):
            if level == 0:
                parent_ids = torch.zeros(codes.shape[0], dtype=torch.long)
            else:
                parent_ids = self._flatten(codes[:, :level])
            unique_parent_ids = torch.unique(parent_ids, sorted=True)
            parent_rows = torch.searchsorted(unique_parent_ids, parent_ids)
            self.register_buffer(
                f"level_{level}_parent_ids",
                unique_parent_ids,
                persistent=False,
            )
            city_masks = torch.zeros(
                (
                    city_mask.shape[0],
                    unique_parent_ids.shape[0],
                    codebook_size,
                ),
                dtype=torch.bool,
            )
            city_rows = torch.arange(city_mask.shape[0])[:, None].expand(
                -1, codes.shape[0]
            )
            expanded_parent_rows = parent_rows[None, :].expand(
                city_mask.shape[0], -1
            )
            expanded_children = codes[:, level][None, :].expand(
                city_mask.shape[0], -1
            )
            city_masks[
                city_rows[city_mask],
                expanded_parent_rows[city_mask],
                expanded_children[city_mask],
            ] = True
            self.register_buffer(
                f"level_{level}_city_masks",
                city_masks,
                persistent=False,
            )

    def _flatten(self, codes: torch.Tensor) -> torch.Tensor:
        return packed_code_keys(codes, codebook_size=self.codebook_size)

    def _find(
        self, sorted_ids: torch.Tensor, codes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        in_range = ((codes >= 0) & (codes < self.codebook_size)).all(dim=-1)
        safe = codes.clamp(min=0, max=self.codebook_size - 1)
        positions, found = find_sorted(sorted_ids, self._flatten(safe))
        return positions, in_range & found

    def resolve_token_ids(
        self, codes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions, found = self._find(self.sorted_flat_ids, codes)
        return self.token_ids_by_sorted_flat[positions], found

    def parent_rows(
        self, level: int, parents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if level == 0:
            rows = torch.zeros(
                parents.shape[0], dtype=torch.long, device=parents.device
            )
            return rows, torch.ones_like(rows, dtype=torch.bool)
        return self._find(getattr(self, f"level_{level}_parent_ids"), parents)

    def next_mask(
        self,
        level: int,
        parents: torch.Tensor,
        catalog_rows: torch.Tensor,
    ) -> torch.Tensor:
        city_masks = getattr(self, f"level_{level}_city_masks")
        rows, found = self.parent_rows(level, parents)
        return city_masks[catalog_rows, rows] & found[..., None]


EARTH_RADIUS_KM = 6371.0088
DISTANCE_BUCKET_EDGES_KM = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0)
DISTANCE_BUCKETS = len(DISTANCE_BUCKET_EDGES_KM) + 1


def unit_vectors(coordinates: torch.Tensor) -> torch.Tensor:
    latitude = torch.deg2rad(coordinates[..., 0].double())
    longitude = torch.deg2rad(coordinates[..., 1].double())
    return torch.stack(
        (
            latitude.cos() * longitude.cos(),
            latitude.cos() * longitude.sin(),
            latitude.sin(),
        ),
        dim=-1,
    )


class PrefixDistancePrior(nn.Module):
    def __init__(
        self,
        *,
        legality: LegalPrefixIndex,
        centroids: torch.Tensor,
        centroid_available: torch.Tensor,
    ) -> None:
        super().__init__()
        coordinates = centroids.float()
        available = centroid_available.bool()
        object.__setattr__(self, "legality", legality)
        self.codebook_size = legality.codebook_size
        self.prefix_depth = legality.prefix_depth

        leaves = unit_vectors(coordinates)
        leaves = leaves * available[:, None].double()
        self.register_buffer("leaf_units", leaves.float(), persistent=False)
        self.register_buffer(
            "leaf_available", available.clone(), persistent=False
        )

        for level in range(self.prefix_depth):
            units, covered_children = self._subtree_centroids(
                level, leaves, available
            )
            self.register_buffer(
                f"level_{level}_child_units", units, persistent=False
            )
            self.register_buffer(
                f"level_{level}_child_available",
                covered_children,
                persistent=False,
            )

        self.register_buffer(
            "bucket_edges_km",
            torch.tensor(DISTANCE_BUCKET_EDGES_KM, dtype=torch.float32),
            persistent=False,
        )
        self.bucket_bias = nn.Embedding(DISTANCE_BUCKETS, 1)
        nn.init.zeros_(self.bucket_bias.weight)

    def _subtree_centroids(
        self,
        level: int,
        leaves: torch.Tensor,
        available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codes = self.legality.codes
        parent_ids = getattr(self.legality, f"level_{level}_parent_ids")
        parent_rows, _ = self.legality.parent_rows(level, codes[:, :level])
        slots = parent_rows * self.codebook_size + codes[:, level]
        total = parent_ids.shape[0] * self.codebook_size

        summed = torch.zeros((total, 3), dtype=torch.float64).index_add_(
            0, slots, leaves
        )
        counts = torch.zeros(total, dtype=torch.float64).index_add_(
            0, slots, available.double()
        )
        norm = summed.norm(dim=-1, keepdim=True)
        units = torch.where(norm > 0.0, summed / norm.clamp_min(1e-12), summed)
        return (
            units.float().reshape(parent_ids.shape[0], self.codebook_size, 3),
            (counts > 0.0).reshape(parent_ids.shape[0], self.codebook_size),
        )

    def source_state(
        self, prefix_codes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids, contained = self.legality.resolve_token_ids(prefix_codes)
        units = self.leaf_units[token_ids]
        valid = contained & self.leaf_available[token_ids]
        return units * valid[..., None].to(units.dtype), valid

    def _bias(
        self,
        candidate_units: torch.Tensor,
        candidate_available: torch.Tensor,
        source_units: torch.Tensor,
        source_valid: torch.Tensor,
    ) -> torch.Tensor:
        cosine = (candidate_units * source_units[..., None, :]).sum(dim=-1)
        distance = (
            EARTH_RADIUS_KM * cosine.clamp(-1.0, 1.0).acos()
        ).nan_to_num(0.0)
        buckets = torch.bucketize(
            distance, self.bucket_edges_km.to(distance.device), right=False
        )
        bias = self.bucket_bias(buckets).squeeze(-1)
        defined = candidate_available & source_valid[..., None]
        return bias * defined.to(bias.dtype)

    def level_bias(
        self,
        level: int,
        parents: torch.Tensor,
        source_units: torch.Tensor,
        source_valid: torch.Tensor,
    ) -> torch.Tensor:
        units = getattr(self, f"level_{level}_child_units")
        covered = getattr(self, f"level_{level}_child_available")
        rows, found = self.legality.parent_rows(level, parents)
        return self._bias(
            units[rows],
            covered[rows] & found[..., None],
            source_units,
            source_valid,
        )


_ILLEGAL_LOGIT = float("-inf")


@dataclass(frozen=True, slots=True)
class PrefixDistanceContext:
    prior: PrefixDistancePrior
    source_units: torch.Tensor
    source_valid: torch.Tensor

    def level_bias(self, level: int, parents: torch.Tensor) -> torch.Tensor:
        return self.prior.level_bias(
            level, parents, self.source_units, self.source_valid
        )

    def flatten(self) -> PrefixDistanceContext:
        return PrefixDistanceContext(
            prior=self.prior,
            source_units=self.source_units.reshape(-1, 3),
            source_valid=self.source_valid.reshape(-1),
        )


def sample_categorical_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    probabilities = torch.log_softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(
        probabilities.exp(), 1, generator=generator
    ).squeeze(-1)


class _CompileSafeGRUCell(nn.GRUCell):
    def forward(self, input: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        input_reset, input_update, input_new = F.linear(
            input, self.weight_ih, self.bias_ih
        ).chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = F.linear(
            hx, self.weight_hh, self.bias_hh
        ).chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        new = torch.tanh(input_new + reset * hidden_new)
        return new + update * (hx - new)


class HierarchicalPrefixDecoder(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        codebook_size: int,
        prefix_depth: int,
        legality: LegalPrefixIndex,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.prefix_depth = prefix_depth
        object.__setattr__(self, "legality", legality)
        self.context = nn.Linear(model_dim, model_dim)
        self.cell = _CompileSafeGRUCell(model_dim, model_dim)
        self.code_embedding = nn.Embedding(codebook_size, model_dim)
        self.level_embedding = nn.Embedding(prefix_depth, model_dim)
        self.start_embedding = nn.Parameter(torch.zeros(model_dim))
        self.heads = nn.ModuleList(
            nn.Linear(model_dim, codebook_size) for _ in range(prefix_depth)
        )

    def _level_logits(
        self,
        state: torch.Tensor,
        level: int,
        parents: torch.Tensor,
        catalog_rows: torch.Tensor,
        distance: PrefixDistanceContext | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoder_input = (
            self.start_embedding.expand(state.shape[0], -1)
            if level == 0
            else self.code_embedding(parents[:, level - 1])
        )
        decoder_input = decoder_input + self.level_embedding.weight[level]
        state = self.cell(decoder_input, state)
        logits = self.heads[level](state)
        valid = self.legality.next_mask(level, parents, catalog_rows)
        if distance is not None:
            logits = logits + distance.level_bias(level, parents)
        return state, logits.masked_fill(~valid, _ILLEGAL_LOGIT)

    def forward(
        self,
        hidden: torch.Tensor,
        target_codes: torch.Tensor,
        *,
        catalog_rows: torch.Tensor,
        distance: PrefixDistanceContext | None,
    ) -> torch.Tensor:
        shape = hidden.shape[:-1]
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_targets = target_codes.reshape(-1, self.prefix_depth)
        flat_catalog_rows = (
            catalog_rows.reshape(shape[0], *([1] * (len(shape) - 1)))
            .expand(*shape)
            .reshape(-1)
        )
        flat_distance = None if distance is None else distance.flatten()
        state = self.context(flat_hidden)
        logits: list[torch.Tensor] = []
        for level in range(self.prefix_depth):
            state, level_logits = self._level_logits(
                state,
                level,
                flat_targets[:, :level],
                flat_catalog_rows,
                flat_distance,
            )
            logits.append(level_logits)
        return torch.stack(logits, dim=1).reshape(
            *shape, self.prefix_depth, -1
        )

    def sample(
        self,
        hidden: torch.Tensor,
        *,
        temperature: float,
        generator: torch.Generator | None,
        catalog_rows: torch.Tensor,
        distance: PrefixDistanceContext | None,
    ) -> torch.Tensor:
        state = self.context(hidden)
        codes = torch.empty(
            (hidden.shape[0], 0), dtype=torch.long, device=hidden.device
        )
        for level in range(self.prefix_depth):
            state, logits = self._level_logits(
                state, level, codes, catalog_rows, distance
            )
            choice = sample_categorical_logits(
                logits, temperature=temperature, generator=generator
            )
            codes = torch.cat((codes, choice[:, None]), dim=-1)
        return codes


__all__ = [
    "HierarchicalPrefixDecoder",
    "LegalPrefixIndex",
    "PrefixDistanceContext",
    "PrefixDistancePrior",
    "SpatioTemporalRelationBias",
    "SpatioTemporalTransformer",
    "sample_categorical_logits",
]
