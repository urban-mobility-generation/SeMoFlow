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
    global_checkins_filename: str = "dataset_TIST2015_Checkins.txt"
    global_pois_filename: str = "dataset_TIST2015_POIs.txt"
    global_cities_filename: str = "dataset_TIST2015_Cities.txt"
    global_batch_size: int = 250_000
    min_quality_score: float = 0.0
    drop_missing_coordinates: bool = True

    @property
    def source_dir(self) -> Path:
        return Path(self.root).expanduser() / Path(self.raw_dir).expanduser()

    def validate(self) -> None:
        if not self.cities:
            raise ValueError("data.cities must name at least one city")
        if not 0.0 <= self.min_quality_score <= 1.0:
            raise ValueError("data.min_quality_score must be in [0, 1]")
        if self.global_batch_size <= 0:
            raise ValueError("data.global_batch_size must be positive")


__all__ = ["DataConfig"]
