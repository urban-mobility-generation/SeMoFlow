from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from semoflow.config.constants import (
    MASK_ID,
    PAD_ID,
    REAL_TOKEN_OFFSET,
    UNK_ID,
)

SID_CODE_PREFIX = "sem_code_"


def sid_code_names(n_codebooks: int) -> tuple[str, ...]:
    return tuple(f"{SID_CODE_PREFIX}{i}" for i in range(n_codebooks))


def sid_code_columns(columns: Iterable[str]) -> tuple[str, ...]:
    names = [c for c in columns if c.startswith(SID_CODE_PREFIX)]
    return tuple(sorted(names, key=lambda c: int(c.rsplit("_", 1)[-1])))


def to_tokens(codes: torch.Tensor) -> torch.Tensor:
    return codes + REAL_TOKEN_OFFSET


def to_token_fields(
    sid_codes: torch.Tensor,
    code_fields: Sequence[str],
) -> dict[str, torch.Tensor]:
    if sid_codes.ndim != 3:
        raise ValueError(
            f"sid_codes must be [N, T, M], got {tuple(sid_codes.shape)}"
        )
    if sid_codes.shape[-1] != len(code_fields):
        raise ValueError(
            "sid_codes last dimension must match code field count: "
            f"{sid_codes.shape[-1]} vs {len(code_fields)}"
        )
    codes = sid_codes.long()
    return {
        field_name: to_tokens(codes[:, :, level])
        for level, field_name in enumerate(code_fields)
    }


def vocab_size(codebook_size: int) -> int:
    return codebook_size + REAL_TOKEN_OFFSET


def apply_trajectory_valid_mask(
    tokens: dict[str, torch.Tensor],
    valid_mask: torch.Tensor | None,
    *,
    pad_id: int = PAD_ID,
) -> dict[str, torch.Tensor]:
    if valid_mask is None:
        return tokens
    mask = valid_mask.bool().cpu()
    masked: dict[str, torch.Tensor] = {}
    for field_name, value in tokens.items():
        if value.shape[:2] != mask.shape:
            raise ValueError(
                f"valid_mask shape {tuple(mask.shape)} does not match "
                f"{field_name} token shape {tuple(value.shape)}"
            )
        field_mask = mask.to(device=value.device)
        masked[field_name] = value.masked_fill(~field_mask, pad_id)
    return masked


__all__ = [
    "MASK_ID",
    "PAD_ID",
    "REAL_TOKEN_OFFSET",
    "SID_CODE_PREFIX",
    "UNK_ID",
    "apply_trajectory_valid_mask",
    "sid_code_columns",
    "sid_code_names",
    "to_token_fields",
    "to_tokens",
    "vocab_size",
]
