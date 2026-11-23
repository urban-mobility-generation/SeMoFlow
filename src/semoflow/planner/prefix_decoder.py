from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from semoflow.poi.sids import find_sorted, packed_code_keys


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
    "sample_categorical_logits",
]
