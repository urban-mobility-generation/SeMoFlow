from __future__ import annotations

import torch

EARTH_RADIUS_KM = 6371.0088


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


def haversine_km(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor,
    lon2: torch.Tensor,
) -> torch.Tensor:
    phi1 = torch.deg2rad(lat1)
    phi2 = torch.deg2rad(lat2)
    dphi = torch.deg2rad(lat2 - lat1)
    dlam = torch.deg2rad(lon2 - lon1)
    a = (
        torch.sin(dphi / 2) ** 2
        + torch.cos(phi1) * torch.cos(phi2) * torch.sin(dlam / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * torch.arcsin(torch.sqrt(a.clamp(0.0, 1.0)))


__all__ = [
    "EARTH_RADIUS_KM",
    "haversine_km",
    "unit_vectors",
]
