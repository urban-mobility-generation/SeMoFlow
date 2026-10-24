from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import yaml


def normalize_value(value: object) -> object:
    # Sparse evaluation panels can have undefined metrics (e.g. no returns).
    # JSON has no NaN/Infinity literals; preserve missingness with null.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return normalize_mapping(dataclasses.asdict(value))
    if (
        not isinstance(value, type)
        and hasattr(value, "to_dict")
        and callable(value.to_dict)
    ):
        to_dict = cast(Callable[[], object], value.to_dict)
        return normalize_value(to_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return normalize_mapping(cast(Mapping[Any, Any], value))
    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]
    return value


def normalize_mapping(mapping: Mapping[Any, Any]) -> dict[str, Any]:
    return {
        str(key): cast(Any, normalize_value(value))
        for key, value in mapping.items()
    }


def to_yaml_string(obj: object) -> str:
    return yaml.dump(
        normalize_value(obj),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=True,
    )


def dump_json(payload: object, *, indent: int | None = 2) -> str:
    """Strict, deterministic JSON: sorted keys, non-finite floats as null."""
    return json.dumps(
        normalize_value(payload),
        indent=indent,
        sort_keys=True,
        default=str,
        allow_nan=False,
    )


def write_json(path: str | Path, payload: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_json(payload) + "\n", encoding="utf-8")
    return target


def write_json_atomic(
    path: str | Path, payload: object, *, force: bool = True
) -> None:
    target = Path(path)
    if target.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing artifact: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(dump_json(payload) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


__all__ = [
    "dump_json",
    "normalize_mapping",
    "normalize_value",
    "to_yaml_string",
    "write_json",
    "write_json_atomic",
]
