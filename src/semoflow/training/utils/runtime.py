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


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def parse_device_spec(spec: str) -> tuple[torch.device, list[int]]:
    stripped = spec.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1]
        ids = [int(x.strip()) for x in inner.split(",") if x.strip()]
        if not ids:
            raise ValueError(f"Empty device list in spec: {spec!r}")
        return torch.device(f"cuda:{ids[0]}"), ids
    return resolve_device(stripped), []


__all__ = [
    "parse_device_spec",
    "resolve_device",
    "set_seed_all",
]
