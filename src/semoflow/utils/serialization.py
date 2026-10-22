from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import yaml


def normalize_value(value: object) -> object:
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


def normalize_config(config: object) -> dict[str, Any]:
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return normalize_mapping(dataclasses.asdict(config))
    if (
        not isinstance(config, type)
        and hasattr(config, "to_dict")
        and callable(config.to_dict)
    ):
        to_dict = cast(Callable[[], object], config.to_dict)
        return normalize_config(to_dict())
    if isinstance(config, Mapping):
        return normalize_mapping(cast(Mapping[Any, Any], config))
    return {"value": repr(config)}


def to_serialisable_dict(obj: object) -> Any:
    return normalize_value(obj)


def to_yaml_string(obj: object) -> str:
    return yaml.dump(
        to_serialisable_dict(obj),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=True,
    )


def write_json_atomic(
    path: str | Path, payload: dict[str, Any], *, force: bool
) -> None:
    target = Path(path)
    if target.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing artifact: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


__all__ = [
    "normalize_config",
    "normalize_mapping",
    "normalize_value",
    "to_serialisable_dict",
    "to_yaml_string",
]
