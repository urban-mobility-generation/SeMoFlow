from __future__ import annotations

import platform
import shlex
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from semoflow.utils.serialization import write_json

RESOLVED_CONFIG_NAME = "config_resolved.yaml"
RUN_MANIFEST_NAME = "run_manifest.json"

_RECORDED_PACKAGES = ("torch", "polars", "numpy", "pyyaml")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_info() -> dict[str, str | bool]:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": _git("status", "--porcelain") != "",
        "origin": _git("remote", "get-url", "origin"),
    }


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _environment_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name) for name in _RECORDED_PACKAGES
        },
    }


def write_run_manifest(directory: str | Path, **fields: Any) -> dict[str, Any]:
    """Write ``run_manifest.json``: the invocation, code, and environment.

    ``fields`` (inputs, outputs, status, ...) are recorded alongside; the
    returned manifest can be re-written later with updated fields.
    """
    argv = [str(part) for part in sys.argv]
    manifest = {
        "command": shlex.join(argv),
        "argv": argv,
        "cwd": str(Path.cwd()),
        "git": _git_info(),
        "environment": _environment_info(),
        **fields,
    }
    write_json(Path(directory) / RUN_MANIFEST_NAME, manifest)
    return manifest


__all__ = ["RESOLVED_CONFIG_NAME", "RUN_MANIFEST_NAME", "write_run_manifest"]
