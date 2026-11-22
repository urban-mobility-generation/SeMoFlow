from __future__ import annotations

from typing import Final

PAD_ID: Final[int] = 0

CITY_ID: Final[dict[str, int]] = {
    "boston": 0,
    "chicago": 1,
    "dallas": 2,
    "new_york_city": 3,
    "pittsburgh": 4,
    "los_angeles": 7,
    "washington_dc": 8,
    "atlanta": 9,
    "san_francisco": 10,
    "milwaukee": 11,
}


__all__ = [
    "CITY_ID",
    "PAD_ID",
]
