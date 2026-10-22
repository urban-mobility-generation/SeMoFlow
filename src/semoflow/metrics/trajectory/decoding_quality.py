from __future__ import annotations

import torch

from semoflow.config.constants import PAD_ID


def valid_flat(tokens: torch.Tensor, *, pad_id: int = PAD_ID) -> torch.Tensor:
    flat = tokens.long().flatten()
    return flat[flat != pad_id]
