from __future__ import annotations

import hashlib
import json
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentPaths:
    output_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    samples_dir: Path
    metrics_path: Path
    config_path: Path
    metadata_path: Path
    git_info_path: Path
    environment_path: Path


def build_experiment_paths(output_dir: str | Path) -> ExperimentPaths:
    root = Path(output_dir)
    return ExperimentPaths(
        output_dir=root,
        checkpoints_dir=root / "checkpoints",
        logs_dir=root / "logs",
        samples_dir=root / "samples",
        metrics_path=root / "metrics.json",
        config_path=root / "config_resolved.yaml",
        metadata_path=root / "metadata.json",
        git_info_path=root / "git_info.json",
        environment_path=root / "environment.json",
    )


def init_run_directory(output_dir: str | Path) -> ExperimentPaths:
    paths = build_experiment_paths(output_dir)
    for d in (
        paths.output_dir,
        paths.checkpoints_dir,
        paths.logs_dir,
        paths.samples_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
    return paths


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def collect_git_info() -> dict[str, str | bool]:
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": _run(["git", "status", "--porcelain"]) != "",
        "origin": _run(["git", "remote", "get-url", "origin"]),
    }


def collect_environment_info() -> dict[str, Any]:
    import sys

    pkgs: dict[str, str] = {}
    for name in ("torch", "polars", "numpy", "yaml", "pytest"):
        try:
            import importlib

            m = importlib.import_module(name if name != "yaml" else "yaml")
            pkgs[name] = getattr(m, "__version__", "?")
        except ImportError:
            pkgs[name] = "not installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": pkgs,
    }


def config_hash(config_dict: dict[str, Any]) -> str:
    serialised = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:8]


@dataclass
class RunMetadata:
    experiment_name: str = "unnamed"
    phase: str = "unknown"
    run_id: str = ""

    start_time: str = ""
    end_time: str = ""
    elapsed_seconds: float = 0.0

    git: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    n_train: int = 0
    n_val: int = 0
    n_test: int = 0

    model_type: str = ""
    n_parameters: int = 0

    output_dir: str = ""
    checkpoint_path: str = ""
    config_path: str = ""

    device: str = ""
    seed: int = 0

    status: str = "started"
    best_val_loss: float = float("inf")
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)
            + "\n",
            encoding="utf-8",
        )


def make_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def start_run(
    *,
    experiment_name: str,
    phase: str,
    seed: int,
    device: str,
    output_dir: str | Path,
    config_dict: dict[str, Any] | None = None,
) -> RunMetadata:
    run_id = make_run_id()
    return RunMetadata(
        experiment_name=experiment_name,
        phase=phase,
        run_id=run_id,
        start_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git=collect_git_info(),
        environment=collect_environment_info(),
        config_hash=config_hash(config_dict or {}),
        output_dir=str(output_dir),
        device=device,
        seed=seed,
        status="running",
    )


def finish_run(
    meta: RunMetadata,
    *,
    status: str = "completed",
    best_val_loss: float = float("inf"),
    n_train: int = 0,
    n_val: int = 0,
    n_test: int = 0,
    n_parameters: int = 0,
    checkpoint_path: str = "",
    extra: dict[str, Any] | None = None,
) -> RunMetadata:
    meta.end_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta.status = status
    meta.best_val_loss = best_val_loss
    meta.n_train = n_train
    meta.n_val = n_val
    meta.n_test = n_test
    meta.n_parameters = n_parameters
    meta.checkpoint_path = checkpoint_path
    if extra:
        meta.extra.update(extra)
    try:
        from datetime import datetime

        fmt = "%Y-%m-%dT%H:%M:%S"
        dt = datetime.strptime(meta.end_time, fmt) - datetime.strptime(
            meta.start_time, fmt
        )
        meta.elapsed_seconds = dt.total_seconds()
    except Exception:
        pass
    return meta


def write_git_info(directory: str | Path) -> None:
    p = Path(directory) / "git_info.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(collect_git_info(), indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_environment_info(directory: str | Path) -> None:
    p = Path(directory) / "environment.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(collect_environment_info(), indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_run_manifest(
    directory: str | Path,
    *,
    command_argv: list[str] | tuple[str, ...] | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    data_artifact_paths: dict[str, str | Path | None] | None = None,
    metric_artifact_paths: dict[str, str | Path | None] | None = None,
) -> None:
    argv = list(command_argv) if command_argv is not None else sys.argv
    command = " ".join(shlex.quote(str(part)) for part in argv)
    root = Path(directory)
    payload = {
        "command": command,
        "argv": [str(part) for part in argv],
        "cwd": str(Path.cwd()),
        "output_dir": str(output_dir if output_dir is not None else root),
        "config_path": str(config_path) if config_path is not None else None,
        "git": collect_git_info(),
        "data_artifact_paths": {
            key: (str(value) if value is not None else None)
            for key, value in (data_artifact_paths or {}).items()
        },
        "metric_artifact_paths": {
            key: (str(value) if value is not None else None)
            for key, value in (metric_artifact_paths or {}).items()
        },
    }
    path = root / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ExperimentPaths",
    "RunMetadata",
    "build_experiment_paths",
    "collect_environment_info",
    "collect_git_info",
    "config_hash",
    "finish_run",
    "init_run_directory",
    "make_run_id",
    "start_run",
    "write_environment_info",
    "write_git_info",
    "write_run_manifest",
]
