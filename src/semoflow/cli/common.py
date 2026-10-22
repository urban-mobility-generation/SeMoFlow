from __future__ import annotations

import json
import logging
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
from semoflow.logging.config_resolution import (
    log_config_summary,
    log_full_config_yaml,
)
from semoflow.logging.primitives import configure_logging
from semoflow.logging.run import (
    log_artifact_inventory,
    log_run_complete,
    log_run_start,
)
from semoflow.utils import (
    RunMetadata,
    finish_run,
    init_run_directory,
    start_run,
    write_environment_info,
    write_git_info,
    write_run_manifest,
)

logger = logging.getLogger(__name__)

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", help="Path to experiment YAML (under configs/)."),
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
        help="Print the fully resolved config YAML and exit.",
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
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def validate_device(device: str) -> None:
    import torch

    try:
        resolved = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(f"invalid --device: {device!r}") from exc
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise typer.BadParameter(
            "--device requests cuda, but cuda is unavailable"
        )
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise typer.BadParameter(
            "--device requests mps, but mps is unavailable"
        )


@dataclass(slots=True)
class RunState:
    output_dir: Path
    metadata_path: Path
    meta: RunMetadata
    started_at: float
    result: dict[str, Any] | None = None


_RESULT_METADATA_FIELDS = (
    "checkpoint_path",
    "best_val_loss",
    "n_train",
    "n_val",
    "n_test",
    "n_parameters",
)


def _completion_fields(result: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in _RESULT_METADATA_FIELDS:
        value = result.get(key)
        if value is None:
            continue
        fields[key] = str(value) if key == "checkpoint_path" else value
    extra = {
        key: value
        for key, value in result.items()
        if key not in _RESULT_METADATA_FIELDS and value is not None
    }
    if extra:
        fields["extra"] = extra
    return fields


def _start_run(
    cfg: ExperimentConfig,
    *,
    phase: str,
    output_dir: Path,
    config_phase: str,
    device: str,
    seed: int,
    metric_artifact_paths: dict[str, Path | str | None] | None,
) -> RunState:
    log_run_start(
        logger,
        name=cfg.experiment.name,
        phase=phase,
        output_dir=output_dir,
        device=device,
        seed=seed,
    )
    paths = init_run_directory(output_dir)
    cfg.save_yaml(paths.config_path, phase=config_phase)
    logger.info("Resolved config written: %s", paths.config_path)
    write_run_manifest(
        paths.output_dir,
        config_path=paths.config_path,
        output_dir=paths.output_dir,
        data_artifact_paths={
            "data_root": cfg.data.data_root,
            "preprocessed_processed_visits_path": (
                cfg.paths.preprocessed_processed_visits_path
            ),
            "semantic_id_lookup_path": cfg.paths.semantic_id_lookup_path,
            "semantic_id_grounding_path": cfg.paths.semantic_id_grounding_path,
            "tokenizer_preprocessing_dir": (
                cfg.tokenizer_config.preprocessing_dir
            ),
            "tokenizer_graphsage_artifact_dir": (
                cfg.tokenizer_config.graphsage_artifact_dir
            ),
        },
        metric_artifact_paths=metric_artifact_paths,
    )
    if cfg.runtime.print_full_config:
        log_full_config_yaml(logger, cfg)
    else:
        log_config_summary(logger, cfg)
    write_git_info(paths.output_dir)
    write_environment_info(paths.output_dir)
    meta = start_run(
        experiment_name=cfg.experiment.name,
        phase=phase,
        seed=seed,
        device=device,
        output_dir=output_dir,
        config_dict=cfg.to_dict(),
    )
    meta.config_path = str(paths.config_path)
    meta.to_json(paths.metadata_path)
    return RunState(
        output_dir=paths.output_dir,
        metadata_path=paths.metadata_path,
        meta=meta,
        started_at=time.perf_counter(),
    )


@contextmanager
def cli_run(
    cfg: ExperimentConfig,
    *,
    phase: str | None = None,
    output_dir: Path | None = None,
    config_phase: str | None = None,
    device: str | None = None,
    seed: int | None = None,
    metric_artifact_paths: dict[str, Path | str | None] | None = None,
    log_level: str = "INFO",
    log_file: Path | None = None,
) -> Generator[RunState, None, None]:
    phase = phase or cfg.experiment.phase
    output_dir = Path(output_dir or cfg.output.output_dir)
    if log_file is None:
        (output_dir / "logs").mkdir(parents=True, exist_ok=True)
        configure_logging(
            level=log_level, log_file=output_dir / "logs" / "run.log"
        )
    state = _start_run(
        cfg,
        phase=phase,
        output_dir=output_dir,
        config_phase=config_phase or phase,
        device=str(device if device is not None else cfg.runtime.device),
        seed=int(seed if seed is not None else cfg.runtime.seed),
        metric_artifact_paths=metric_artifact_paths,
    )
    try:
        yield state
    except Exception as exc:
        logger.exception("%s run failed", phase)
        log_run_complete(
            logger,
            elapsed=time.perf_counter() - state.started_at,
            status=f"failed: {type(exc).__name__}",
            level=logging.ERROR,
        )
        state.meta = finish_run(state.meta, status=f"failed: {exc}")
        state.meta.to_json(state.metadata_path)
        raise
    fields = _completion_fields(state.result or {})
    state.meta = finish_run(state.meta, status="completed", **fields)
    state.meta.to_json(state.metadata_path)
    log_artifact_inventory(logger, state.output_dir)
    log_run_complete(
        logger,
        elapsed=time.perf_counter() - state.started_at,
        status="completed",
        checkpoint_path=fields.get("checkpoint_path", ""),
        best_val_loss=fields.get("best_val_loss"),
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
