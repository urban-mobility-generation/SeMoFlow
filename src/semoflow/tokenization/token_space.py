from __future__ import annotations

from collections.abc import Iterable

SID_CODE_PREFIX = "sem_code_"


def sid_code_names(n_codebooks: int) -> tuple[str, ...]:
    return tuple(f"{SID_CODE_PREFIX}{i}" for i in range(n_codebooks))


def sid_code_columns(columns: Iterable[str]) -> tuple[str, ...]:
    names = [c for c in columns if c.startswith(SID_CODE_PREFIX)]
    return tuple(sorted(names, key=lambda c: int(c.rsplit("_", 1)[-1])))


__all__ = ["SID_CODE_PREFIX", "sid_code_columns", "sid_code_names"]
