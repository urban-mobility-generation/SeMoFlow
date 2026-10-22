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


def fmt_value(value: object) -> str:
    if isinstance(value, float) and not (
        value == 0.0 or 1e-3 <= abs(value) < 1e4
    ):
        return f"{value:.3e}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def log_section(logger: logging.Logger, title: str) -> None:
    logger.info("")
    logger.info(_RULE)
    logger.info("  %s", title)
    logger.info(_RULE)


def log_section_end(logger: logging.Logger) -> None:
    logger.info(_RULE)


def log_kv_block(
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
        logger.info("    %-*s  %s", width, key, fmt_value(value))


__all__ = [
    "configure_logging",
    "fmt_value",
    "get_configured_log_file",
    "log_kv_block",
    "log_section",
    "log_section_end",
]
