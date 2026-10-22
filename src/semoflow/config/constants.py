from __future__ import annotations

from typing import Final

PAD_ID: Final[int] = 0
UNK_ID: Final[int] = 1
MASK_ID: Final[int] = 2
REAL_TOKEN_OFFSET: Final[int] = 3

CITY_TZ: Final[dict[str, str]] = {
    "boston": "America/New_York",
    "chicago": "America/Chicago",
    "dallas": "America/Chicago",
    "new_york_city": "America/New_York",
    "pittsburgh": "America/New_York",
    "los_angeles": "America/Los_Angeles",
    "washington_dc": "America/New_York",
    "atlanta": "America/New_York",
    "san_francisco": "America/Los_Angeles",
    "milwaukee": "America/Chicago",
}
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
    "CITY_TZ",
    "MASK_ID",
    "PAD_ID",
    "REAL_TOKEN_OFFSET",
    "UNK_ID",
]
