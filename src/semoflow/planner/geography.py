from __future__ import annotations

import torch
import torch.nn as nn

from semoflow.planner.legality import LegalPrefixIndex

EARTH_RADIUS_KM = 6371.0088
DISTANCE_BUCKET_EDGES_KM = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0)
DISTANCE_BUCKETS = len(DISTANCE_BUCKET_EDGES_KM) + 1


def unit_vectors(coordinates: torch.Tensor) -> torch.Tensor:
    """Unit vectors on the sphere of ``[..., (latitude, longitude)]``."""
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


__all__ = [
    "DISTANCE_BUCKETS",
    "DISTANCE_BUCKET_EDGES_KM",
    "EARTH_RADIUS_KM",
    "PrefixDistancePrior",
    "unit_vectors",
]
