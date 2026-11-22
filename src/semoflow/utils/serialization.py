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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return normalize_value(dataclasses.asdict(value))
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
        return {
            str(k): normalize_value(v)
            for k, v in cast(Mapping[Any, Any], value).items()
        }
    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]
    return value


def to_yaml_string(obj: object) -> str:
    return yaml.dump(
        normalize_value(obj),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
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
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(dump_json(payload) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


__all__ = [
    "dump_json",
    "normalize_value",
    "to_yaml_string",
    "write_json",
]
