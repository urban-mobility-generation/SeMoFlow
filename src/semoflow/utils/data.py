from __future__ import annotations

import hashlib

import polars as pl


def stable_device_key(city: str | int, caid: str) -> str:
    return hashlib.blake2b(
        f"{city}|{caid}".encode(),
        digest_size=12,
    ).hexdigest()


def hash_fraction(value: str, seed: int) -> float:
    payload = f"{seed}:{value}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2**64


def local_timestamp_expr(
    timestamp_col: str,
    tz: str,
    *,
    alias: str = "timestamp",
) -> pl.Expr:
    return (
        pl.col(timestamp_col)
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(tz)
        .dt.replace_time_zone(None)
        .alias(alias)
    )
