from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """The Foursquare global check-in release and the cities taken from it.

    ``raw_dir`` is relative to ``root`` unless absolute.
    """

    root: Path = Path("data")
    raw_dir: Path = Path("foursquare_global_2015")
    cities: list[str] = field(
        default_factory=lambda: [
            "los_angeles",
            "washington_dc",
            "atlanta",
            "san_francisco",
            "milwaukee",
        ]
    )

    @property
    def source_dir(self) -> Path:
        return Path(self.root).expanduser() / Path(self.raw_dir).expanduser()

    def validate(self) -> None:
        if not self.cities:
            raise ValueError("data.cities must name at least one city")


__all__ = ["DataConfig"]
