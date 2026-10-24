from __future__ import annotations

import hashlib


def stable_device_key(city: str | int, caid: str) -> str:
    return hashlib.blake2b(
        f"{city}|{caid}".encode(),
        digest_size=12,
    ).hexdigest()


def hash_fraction(value: str, seed: int) -> float:
    payload = f"{seed}:{value}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2**64
