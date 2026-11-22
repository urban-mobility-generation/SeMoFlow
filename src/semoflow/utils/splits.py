from __future__ import annotations

import hashlib

GLOBAL_SPLIT_SEED = 20260712
SPLIT_COLUMN = "split"


def stable_key(value: str) -> str:
    return hashlib.blake2b(value.encode(), digest_size=12).hexdigest()


def hash_fraction(value: str, seed: int) -> float:
    digest = hashlib.blake2b(
        f"{seed}:{value}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") / 2**64


def assign_split(key: str, *, seed: int = GLOBAL_SPLIT_SEED) -> str:
    fraction = hash_fraction(key, seed)
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "val"
    return "test"


__all__ = [
    "GLOBAL_SPLIT_SEED",
    "SPLIT_COLUMN",
    "assign_split",
    "hash_fraction",
    "stable_key",
]
