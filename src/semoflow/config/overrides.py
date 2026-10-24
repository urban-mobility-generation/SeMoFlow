from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any

import yaml

_NULL = {"", "null", "none", "~"}
_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def _non_null_types(annotation: Any) -> tuple[list[Any], bool]:
    """Union members other than ``None``, and whether ``None`` is allowed."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        members = typing.get_args(annotation)
        kinds = [kind for kind in members if kind is not type(None)]
        return kinds, len(kinds) < len(members)
    return [annotation], False


def _dataclass_type(annotation: Any) -> type | None:
    kinds, _ = _non_null_types(annotation)
    if len(kinds) == 1 and dataclasses.is_dataclass(kinds[0]):
        return kinds[0]
    return None


def _coerce(text: str, annotation: Any) -> object:
    """Parse ``text`` as a value of the field type ``annotation``.

    Used for ``--set`` values and for YAML strings, which PyYAML also
    produces for exponent floats written without a dot (``2e-4``).
    """
    kinds, optional = _non_null_types(annotation)
    if optional and text.strip().lower() in _NULL:
        return None
    kind = kinds[0] if len(kinds) == 1 else None
    if kind is str:
        return text
    if kind is Path:
        return Path(text)
    if kind is bool:
        if text.strip().lower() in _TRUE | _FALSE:
            return text.strip().lower() in _TRUE
        raise ValueError(f"expected a boolean, got {text!r}")
    if kind in (int, float):
        return kind(text)
    if typing.get_origin(kind) in (list, tuple) and not text.startswith("["):
        text = f"[{text}]"
    return yaml.safe_load(text)


def apply_override(data: dict[str, Any], override: str, cls: type) -> None:
    """Set ``section.field=value`` in raw config ``data`` for dataclass ``cls``."""
    key, sep, raw_value = override.partition("=")
    parts = key.strip().split(".")
    if not sep or len(parts) < 2:
        raise ValueError(
            f"Override must be section.field=value, got {override!r}"
        )
    target, owner = data, cls
    for depth, part in enumerate(parts):
        types_by_name = typing.get_type_hints(owner)
        if part not in types_by_name:
            where = ".".join(parts[:depth]) or "config"
            raise ValueError(
                f"Unknown field {part!r} in {where!r}. "
                f"Valid fields: {sorted(types_by_name)}"
            )
        if depth == len(parts) - 1:
            try:
                target[part] = _coerce(raw_value.strip(), types_by_name[part])
            except ValueError as exc:
                raise ValueError(f"Cannot set {key.strip()}: {exc}") from exc
            return
        section = _dataclass_type(types_by_name[part])
        if section is None:
            raise ValueError(f"{'.'.join(parts[: depth + 1])!r} is a field")
        if not isinstance(target.get(part), dict):
            target[part] = {}
        target, owner = target[part], section


def from_dict[T](cls: type[T], data: dict[str, Any]) -> T:
    """Build dataclass ``cls`` from parsed YAML, recursing into sections."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a dataclass")
    types_by_name = typing.get_type_hints(cls)
    unknown = set(data) - set(types_by_name)
    if unknown:
        raise ValueError(
            f"Unknown keys for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(types_by_name)}"
        )
    missing = [
        f.name
        for f in dataclasses.fields(cls)
        if f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
        and f.name not in data
    ]
    if missing:
        raise ValueError(f"{cls.__name__} requires keys: {missing}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        annotation = types_by_name[name]
        section = _dataclass_type(annotation)
        if section is not None and isinstance(value, dict):
            value = from_dict(section, value)
        elif isinstance(value, str):
            try:
                value = _coerce(value, annotation)
            except ValueError as exc:
                raise ValueError(f"{cls.__name__}.{name}: {exc}") from exc
        kwargs[name] = value
    return cls(**kwargs)


__all__ = ["apply_override", "from_dict"]
