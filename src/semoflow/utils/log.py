from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

_PACKAGE_ROOT = "semoflow."
_LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(subsystem)-22s | %(message)s"
_LOG_FORMAT_DEBUG = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_CONSOLE_HANDLER = "semoflow-console"
_FILE_HANDLER = "semoflow-file"
_RULE = "─" * 56


class _SubsystemFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.subsystem = record.name.removeprefix(_PACKAGE_ROOT)
        return True


def _handler(name: str) -> logging.Handler | None:
    return next(
        (h for h in logging.getLogger().handlers if h.name == name), None
    )


def configure_logging(
    *, level: str = "INFO", log_file: Path | None = None
) -> None:
    numeric_level = logging.getLevelNamesMapping().get(level.upper())
    if numeric_level is None:
        raise ValueError(
            f"Unknown log level {level!r}. Expected DEBUG/INFO/WARNING/ERROR."
        )
    root = logging.getLogger()
    root.setLevel(numeric_level)
    formatter = logging.Formatter(
        _LOG_FORMAT_DEBUG if numeric_level <= logging.DEBUG else _LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )

    console = _handler(_CONSOLE_HANDLER)
    if console is None:
        console = logging.StreamHandler()
        console.name = _CONSOLE_HANDLER
        console.addFilter(_SubsystemFilter())
        root.addHandler(console)
    console.setFormatter(formatter)

    path = None if log_file is None else log_file.expanduser().resolve()
    current = _handler(_FILE_HANDLER)
    if current is not None and path != get_configured_log_file():
        root.removeHandler(current)
        current.close()
        current = None
    if path is not None and current is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        current = logging.FileHandler(path, encoding="utf-8")
        current.name = _FILE_HANDLER
        current.addFilter(_SubsystemFilter())
        root.addHandler(current)
    if current is not None:
        current.setFormatter(formatter)


def get_configured_log_file() -> Path | None:
    handler = _handler(_FILE_HANDLER)
    if isinstance(handler, logging.FileHandler):
        return Path(handler.baseFilename)
    return None


def _fmt_value(value: object) -> str:
    if isinstance(value, float) and not (
        value == 0.0 or 1e-3 <= abs(value) < 1e4
    ):
        return f"{value:.3e}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _section(logger: logging.Logger, title: str) -> None:
    logger.info("")
    logger.info(_RULE)
    logger.info("  %s", title)
    logger.info(_RULE)


def _section_end(logger: logging.Logger) -> None:
    logger.info(_RULE)


def _kv_block(
    logger: logging.Logger,
    title: str | None,
    items: Mapping[str, object] | Sequence[tuple[str, object]],
) -> None:
    pairs = list(items.items()) if isinstance(items, Mapping) else list(items)
    if not pairs:
        return
    width = min(max(len(str(key)) for key, _ in pairs), 30)
    if title is not None:
        logger.info("  %s", title)
    for key, value in pairs:
        logger.info("    %-*s  %s", width, key, _fmt_value(value))


def log_config_resolution(
    logger: logging.Logger,
    *,
    config_path: Path | None,
    overrides: Sequence[str],
    experiment_name: str,
) -> None:
    _section(logger, "Config Resolution")
    _kv_block(
        logger,
        None,
        [
            ("config_path", config_path or "defaults only"),
            ("experiment", experiment_name),
        ],
    )
    if overrides:
        logger.info("  Overrides (%d)", len(overrides))
        for override in overrides:
            logger.info("    %s", override)
    _section_end(logger)


def _progress(current: int, total: int) -> str:
    return f"{current:0{len(str(total))}d}/{total}"


def log_training_step(
    logger: logging.Logger,
    *,
    step: int,
    total_steps: int,
    epoch: int,
    total_epochs: int,
    metrics: Mapping[str, float],
    lr: float,
    samples_per_sec: float | None = None,
) -> None:
    parts = [
        f"Epoch {_progress(epoch, total_epochs)}",
        f"step {_progress(step, total_steps)}",
        *(f"{name} {_fmt_value(value)}" for name, value in metrics.items()),
        f"lr {lr:.2e}",
    ]
    if samples_per_sec is not None:
        parts.append(f"{int(samples_per_sec):,} sps")
    logger.info("%s", " | ".join(parts))


def log_epoch_summary(
    logger: logging.Logger,
    *,
    epoch: int,
    total_epochs: int,
    epoch_elapsed_s: float,
    samples_per_sec: float,
    gpu_allocated_gb: float,
    gpu_peak_gb: float,
) -> None:
    logger.info(
        "Epoch %s | %.1fs | %s sps | GPU %.2fGB / peak %.2fGB",
        _progress(epoch, total_epochs),
        epoch_elapsed_s,
        f"{int(samples_per_sec):,}",
        gpu_allocated_gb,
        gpu_peak_gb,
    )


def log_ddp_summary(
    logger: logging.Logger,
    *,
    device_ids: list[int],
    port: int,
    batch_per_rank: int,
) -> None:
    logger.info(
        "DDP world_size=%d  devices=%s  port=%d  batch_per_rank=%d",
        len(device_ids),
        ", ".join(f"cuda:{device}" for device in device_ids),
        port,
        batch_per_rank,
    )


__all__ = [
    "configure_logging",
    "get_configured_log_file",
    "log_config_resolution",
    "log_ddp_summary",
    "log_epoch_summary",
    "log_training_step",
]
