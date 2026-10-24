from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from semoflow.config import (
    ExperimentConfig,
    resolve_experiment_config,
)
from semoflow.logging.primitives import configure_logging
from semoflow.logging.run import (
    log_artifact_inventory,
    log_run_complete,
    log_run_start,
)
from semoflow.utils.run_metadata import (
    RESOLVED_CONFIG_NAME,
    RUN_MANIFEST_NAME,
    write_run_manifest,
)
from semoflow.utils.serialization import dump_json, write_json

logger = logging.getLogger(__name__)

ConfigOpt = Annotated[
    Path | None,
    typer.Option(
        "--config",
        exists=True,
        dir_okay=False,
        help="Path to experiment YAML (under configs/).",
    ),
]
SetArgsOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        help="Override a config field: section.field=value. May be repeated.",
    ),
]
LogLevelOpt = Annotated[
    str,
    typer.Option(
        "--log-level", help="Logging level (DEBUG/INFO/WARNING/ERROR)."
    ),
]
LogFileOpt = Annotated[
    Path | None,
    typer.Option("--log-file", help="Optional log file path."),
]
DryRunConfigOpt = Annotated[
    bool,
    typer.Option(
        "--dry-run-config",
        help="Print the resolved config YAML and exit.",
    ),
]
ForceOpt = Annotated[
    bool,
    typer.Option("--force/--no-force", help="Overwrite existing outputs."),
]


def resolve_cli_config(
    config: Path | None,
    set_args: list[str] | None,
    *,
    log_level: str = "INFO",
    log_file: Path | None = None,
) -> ExperimentConfig:
    configure_logging(level=log_level, log_file=log_file)
    with bad_parameter_on(ValueError):
        return resolve_experiment_config(
            config_path=config, overrides=list(set_args or [])
        )


@contextmanager
def bad_parameter_on(*errors: type[Exception]) -> Generator[None, None, None]:
    try:
        yield
    except errors as exc:
        raise typer.BadParameter(str(exc)) from exc


def echo_json(payload: object) -> None:
    typer.echo(dump_json(payload))


def validate_device(device: str, *, name: str = "--device") -> None:
    import torch

    try:
        resolved = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(f"invalid {name}: {device!r}") from exc
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise typer.BadParameter(
            f"{name} requests cuda, but cuda is unavailable"
        )
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise typer.BadParameter(
            f"{name} requests mps, but mps is unavailable"
        )


@dataclass(slots=True)
class RunState:
    output_dir: Path
    result: dict[str, Any] | None = None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@contextmanager
def cli_run(
    cfg: ExperimentConfig,
    *,
    phase: str,
    output_dir: Path,
    config_phase: str | None = None,
    device: str | None = None,
    seed: int | None = None,
    log_level: str = "INFO",
    log_file: Path | None = None,
) -> Generator[RunState, None, None]:
    """Log to ``output_dir``, snapshot the config, and keep a run manifest.

    The manifest records ``status`` (running, then completed or failed)
    and, on success, ``RunState.result``. Under torchrun only rank zero
    records the run.
    """
    output_dir = Path(output_dir)
    if int(os.environ.get("RANK", "0")) != 0:
        yield RunState(output_dir=output_dir)
        return
    if log_file is None:
        configure_logging(
            level=log_level, log_file=output_dir / "logs" / "run.log"
        )
    log_run_start(
        logger,
        name=cfg.experiment.name,
        phase=phase,
        output_dir=output_dir,
        device=device,
        seed=seed,
    )
    config_path = output_dir / RESOLVED_CONFIG_NAME
    cfg.save_yaml(config_path, phase=config_phase or phase)
    logger.info("Resolved config written: %s", config_path)
    manifest = write_run_manifest(
        output_dir,
        experiment=cfg.experiment.name,
        phase=phase,
        config_path=config_path,
        device=device,
        seed=seed,
        status="running",
        started_at=_now(),
    )
    state = RunState(output_dir=output_dir)
    started = time.perf_counter()

    def finish(status: str, **fields: Any) -> float:
        elapsed = time.perf_counter() - started
        manifest.update(
            status=status,
            finished_at=_now(),
            elapsed_seconds=round(elapsed, 1),
            **fields,
        )
        write_json(output_dir / RUN_MANIFEST_NAME, manifest)
        return elapsed

    try:
        yield state
    except Exception as exc:
        logger.exception("%s run failed", phase)
        elapsed = finish("failed", error=f"{type(exc).__name__}: {exc}")
        log_run_complete(
            logger,
            elapsed=elapsed,
            status=f"failed: {type(exc).__name__}",
            level=logging.ERROR,
        )
        raise
    result = state.result or {}
    elapsed = finish("completed", result=result)
    log_artifact_inventory(logger, output_dir)
    log_run_complete(
        logger,
        elapsed=elapsed,
        status="completed",
        checkpoint_path=str(result.get("checkpoint_path") or ""),
        best_val_loss=result.get("best_val_loss"),
    )


__all__ = [
    "ConfigOpt",
    "DryRunConfigOpt",
    "ForceOpt",
    "LogFileOpt",
    "LogLevelOpt",
    "RunState",
    "SetArgsOpt",
    "bad_parameter_on",
    "cli_run",
    "echo_json",
    "resolve_cli_config",
    "validate_device",
]
