from __future__ import annotations

from collections.abc import Iterable

import torch

SID_CODE_PREFIX = "sem_code_"


def sid_code_names(n_codebooks: int) -> tuple[str, ...]:
    return tuple(f"{SID_CODE_PREFIX}{i}" for i in range(n_codebooks))


def sid_code_columns(columns: Iterable[str]) -> tuple[str, ...]:
    names = [c for c in columns if c.startswith(SID_CODE_PREFIX)]
    return tuple(sorted(names, key=lambda c: int(c.rsplit("_", 1)[-1])))


def packed_code_keys(
    codes: torch.Tensor, *, codebook_size: int
) -> torch.LongTensor:
    """Each ``[..., width]`` code row as one base-``codebook_size`` integer,
    first level most significant, so key order is lexicographic code order."""
    width = codes.shape[-1]
    powers = torch.tensor(
        [codebook_size**power for power in range(width - 1, -1, -1)],
        dtype=torch.long,
        device=codes.device,
    )
    return (codes.long() * powers).sum(dim=-1)


def find_sorted(
    sorted_keys: torch.Tensor, keys: torch.Tensor
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    """Each key's position in ``sorted_keys`` and whether it is there.

    Positions of absent keys are clamped into range, so they index safely.
    """
    positions = torch.searchsorted(sorted_keys, keys)
    safe = positions.clamp_max(max(sorted_keys.numel() - 1, 0))
    found = (positions < sorted_keys.numel()) & (sorted_keys[safe] == keys)
    return safe, found


__all__ = [
    "SID_CODE_PREFIX",
    "find_sorted",
    "packed_code_keys",
    "sid_code_columns",
    "sid_code_names",
]
