from __future__ import annotations

_GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash_center(label: str) -> tuple[float, float]:
    latitude = [-90.0, 90.0]
    longitude = [-180.0, 180.0]
    longitude_bit = True
    for character in label.lower():
        try:
            value = _GEOHASH_ALPHABET.index(character)
        except ValueError as exc:
            raise ValueError(
                f"invalid geohash character: {character!r}"
            ) from exc
        for mask in (16, 8, 4, 2, 1):
            interval = longitude if longitude_bit else latitude
            midpoint = (interval[0] + interval[1]) / 2.0
            if value & mask:
                interval[0] = midpoint
            else:
                interval[1] = midpoint
            longitude_bit = not longitude_bit
    return (
        (latitude[0] + latitude[1]) / 2.0,
        (longitude[0] + longitude[1]) / 2.0,
    )


__all__ = ["geohash_center"]
