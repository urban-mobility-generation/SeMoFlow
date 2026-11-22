from __future__ import annotations

from pathlib import Path

import torch


def load_codebooks(path: str | Path) -> torch.Tensor:
    """The ``[levels, codebook_size, dim]`` codebooks a tokenizer published."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return payload["codebooks"].detach().float().contiguous()


def codes_to_embeddings(
    codes: torch.Tensor, codebooks: torch.Tensor
) -> torch.Tensor:
    """Sum of each level's codebook row: ``[..., levels]`` to ``[..., dim]``."""
    codes = codes.long()
    codebooks = codebooks.to(codes.device)
    if codes.numel() and (
        int(codes.min()) < 0 or int(codes.max()) >= codebooks.shape[1]
    ):
        raise ValueError(
            f"semantic code out of range for codebook_size={codebooks.shape[1]}"
        )
    levels = torch.arange(codebooks.shape[0], device=codes.device)
    return codebooks[levels, codes].sum(dim=-2)


__all__ = ["codes_to_embeddings", "load_codebooks"]
