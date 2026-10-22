from __future__ import annotations

import contextlib
from typing import Any

import torch


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if mixed_precision not in {"none", "bf16", "fp16"}:
        raise ValueError(
            "mixed_precision must be one of 'none', 'bf16', or 'fp16'"
        )
    return mixed_precision


def build_amp_context(
    mixed_precision: str,
    device: torch.device,
) -> tuple[Any, torch.amp.GradScaler | None]:
    mode = _normalize_mixed_precision(mixed_precision)
    if mode != "none" and device.type == "cuda":
        amp_dtype = torch.bfloat16 if mode == "bf16" else torch.float16
        ctx: Any = torch.autocast(device_type="cuda", dtype=amp_dtype)
        scaler = (
            torch.amp.GradScaler("cuda")
            if amp_dtype == torch.float16
            else None
        )
        return ctx, scaler
    return contextlib.nullcontext(), None
