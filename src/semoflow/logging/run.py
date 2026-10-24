from __future__ import annotations

import logging
from pathlib import Path

from semoflow.logging.primitives import (
    log_kv_block,
    log_section,
    log_section_end,
)


def _human_size(n_bytes: int) -> str:
    for unit, threshold in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n_bytes >= threshold:
            return f"{n_bytes / threshold:.1f} {unit}"
    return f"{n_bytes} B"


def log_run_start(
    logger: logging.Logger,
    *,
    name: str,
    phase: str,
    output_dir: Path,
    device: str | None = None,
    seed: int | None = None,
) -> None:
    rows = [
        ("experiment", name),
        ("phase", phase),
        ("output_dir", output_dir),
        ("device", device),
        ("seed", seed),
    ]
    log_section(logger, "Run Start")
    log_kv_block(logger, None, [row for row in rows if row[1] is not None])
    log_section_end(logger)


def log_run_complete(
    logger: logging.Logger,
    *,
    elapsed: float,
    status: str,
    checkpoint_path: str = "",
    best_val_loss: float | None = None,
    level: int = logging.INFO,
) -> None:
    logger.log(
        level,
        "Run %s  elapsed=%.1fs%s%s",
        status,
        elapsed,
        ""
        if best_val_loss is None
        else f"  best_val_loss={best_val_loss:.4f}",
        f"  checkpoint={checkpoint_path}" if checkpoint_path else "",
    )


def log_artifact_inventory(logger: logging.Logger, output_dir: Path) -> None:
    files = sorted(f for f in output_dir.glob("*") if f.is_file())
    if not files:
        return
    sizes = [f.stat().st_size for f in files]
    log_section(logger, "Output Artifacts")
    for path, size in zip(files, sizes, strict=True):
        logger.info("    %-36s %s", path.name, _human_size(size))
    logger.info(
        "    %d file(s)   %s total", len(files), _human_size(sum(sizes))
    )
    log_section_end(logger)


__all__ = ["log_artifact_inventory", "log_run_complete", "log_run_start"]
