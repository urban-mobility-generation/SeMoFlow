from __future__ import annotations

import random

import numpy as np
import torch


def set_seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_device_spec(spec: str) -> tuple[torch.device, list[int]]:
    """A device, or ``[i,j,...]``: a GPU list whose first entry is primary."""
    stripped = spec.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        ids = [int(x) for x in stripped[1:-1].split(",") if x.strip()]
        if not ids:
            raise ValueError(f"Empty device list in spec: {spec!r}")
        return torch.device(f"cuda:{ids[0]}"), ids
    return torch.device(stripped), []


__all__ = ["parse_device_spec", "set_seed_all"]
