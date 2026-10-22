from __future__ import annotations

import ast
import copy
import dataclasses
import json
import typing
from pathlib import Path
from typing import Any


def _is_null(text: str) -> bool:
    return text.lower() in ("", "null", "none")


def _infer_scalar(text: str) -> object:
    for converter in (int, float):
        try:
            return converter(text)
        except ValueError:
            pass
    return text


def _type_hints(cls: type) -> dict[str, Any]:
    try:
        return typing.get_type_hints(cls)
    except Exception:
        return {}


def _coerce(raw: str, owner: object, name: str) -> object:
    text = raw.strip()
    current = getattr(owner, name)
    if current is None:
        return None if _is_null(text) else _infer_scalar(text)
    if not isinstance(current, str) and _is_null(text):
        return None
    if isinstance(current, bool):
        return text.lower() in ("1", "true", "yes", "y", "on")
    if isinstance(current, int | float | Path):
        return type(current)(text)
    if isinstance(current, list):
        item = type(current[0]) if current else _infer_scalar
        return [
            item(v.strip()) for v in text.strip("[]").split(",") if v.strip()
        ]
    if isinstance(current, dict):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                raise ValueError(
                    f"Cannot parse dict override {text!r}; provide JSON or "
                    "a Python dict literal."
                ) from None
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a dict override, got {text!r}")
        return parsed
    return text


def apply_overrides[T](config: T, overrides: list[str]) -> T:
    cfg = copy.deepcopy(config)
    for override in overrides:
        key, sep, raw_value = override.partition("=")
        parts = key.strip().split(".")
        if not sep or len(parts) < 2:
            raise ValueError(
                f"Override must be section.field=value, got {override!r}"
            )
        target = cfg
        for depth, part in enumerate(parts[:-1]):
            if not dataclasses.is_dataclass(target) or not hasattr(
                target, part
            ):
                where = ".".join(parts[:depth]) or "config"
                raise ValueError(
                    f"Unknown config section {part!r} in {where!r}"
                )
            target = getattr(target, part)
        name = parts[-1]
        if not dataclasses.is_dataclass(target) or not hasattr(target, name):
            valid = (
                sorted(f.name for f in dataclasses.fields(target))
                if dataclasses.is_dataclass(target)
                else []
            )
            raise ValueError(
                f"Unknown field {name!r} in section "
                f"{'.'.join(parts[:-1])!r}. Valid fields: {valid}"
            )
        setattr(target, name, _coerce(raw_value, target, name))
    return cfg


def from_dict[T](cls: type[T], data: dict[str, Any]) -> T:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a dataclass")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    hints = _type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        field_type = hints.get(name)
        if isinstance(value, dict) and dataclasses.is_dataclass(field_type):
            value = from_dict(field_type, value)
        elif isinstance(value, str) and (
            field_type is Path or Path in typing.get_args(field_type)
        ):
            value = Path(value)
        kwargs[name] = value
    return cls(**kwargs)


__all__ = ["apply_overrides", "from_dict"]
