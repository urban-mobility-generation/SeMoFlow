from __future__ import annotations

import torch
import torch.nn as nn

from semoflow.planner.legality import LegalPrefixIndex

EARTH_RADIUS_KM = 6371.0088
DISTANCE_BUCKET_EDGES_KM = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0)
DISTANCE_BUCKETS = len(DISTANCE_BUCKET_EDGES_KM) + 1


def unit_vectors(coordinates: torch.Tensor) -> torch.Tensor:
    if coordinates.shape[-1] != 2:
        raise ValueError("coordinates must end with (latitude, longitude)")
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
        vocabulary = legality.vocabulary_size
        coordinates = torch.as_tensor(centroids, dtype=torch.float32)
        available = torch.as_tensor(centroid_available, dtype=torch.bool)
        if coordinates.shape != (vocabulary, 2):
            raise ValueError(
                "prefix centroids must have shape [vocabulary, 2]"
            )
        if available.shape != (vocabulary,):
            raise ValueError(
                "centroid availability must have shape [vocabulary]"
            )
        if not bool(torch.isfinite(coordinates).all()):
            raise ValueError("prefix centroids must be finite")
        covered = coordinates[available]
        if covered.numel() and (
            bool((covered[:, 0].abs() > 90.0).any())
            or bool((covered[:, 1].abs() > 180.0).any())
        ):
            raise ValueError("prefix centroids are outside WGS84")
        if not bool(available.any()):
            raise ValueError(
                "distance prior requires at least one prefix centroid"
            )

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
        if level == 0:
            parent_rows = torch.zeros(codes.shape[0], dtype=torch.long)
        else:
            flat = self._flatten(codes[:, :level])
            parent_rows = torch.searchsorted(parent_ids, flat)
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

    def _flatten(self, codes: torch.Tensor) -> torch.Tensor:
        depth = codes.shape[-1]
        weights = torch.tensor(
            [self.codebook_size ** (depth - i - 1) for i in range(depth)],
            dtype=torch.long,
            device=codes.device,
        )
        return (codes * weights).sum(dim=-1)

    def source_state(
        self, prefix_codes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prefix_codes.shape[-1] != self.prefix_depth:
            raise ValueError("source prefixes must end with prefix_depth")
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
        if not 0 <= level < self.prefix_depth:
            raise ValueError("level is outside prefix_depth")
        if parents.shape[-1] != level:
            raise ValueError("parents must contain exactly `level` codes")
        units = getattr(self, f"level_{level}_child_units")
        covered = getattr(self, f"level_{level}_child_available")
        if level == 0:
            rows = torch.zeros(
                parents.shape[0], dtype=torch.long, device=parents.device
            )
            found = torch.ones_like(rows, dtype=torch.bool)
        else:
            parent_ids = getattr(self.legality, f"level_{level}_parent_ids")
            in_range = ((parents >= 0) & (parents < self.codebook_size)).all(
                dim=-1
            )
            safe = parents.clamp(min=0, max=self.codebook_size - 1)
            positions = torch.searchsorted(parent_ids, self._flatten(safe))
            rows = positions.clamp(max=parent_ids.shape[0] - 1)
            found = in_range & (parent_ids[rows] == self._flatten(safe))
        return self._bias(
            units[rows],
            covered[rows] & found[..., None],
            source_units,
            source_valid,
        )

    def flat_bias(
        self, source_units: torch.Tensor, source_valid: torch.Tensor
    ) -> torch.Tensor:
        return self._bias(
            self.leaf_units,
            self.leaf_available.expand(*source_valid.shape, -1),
            source_units,
            source_valid,
        )


__all__ = [
    "DISTANCE_BUCKETS",
    "DISTANCE_BUCKET_EDGES_KM",
    "EARTH_RADIUS_KM",
    "PrefixDistancePrior",
    "unit_vectors",
]
