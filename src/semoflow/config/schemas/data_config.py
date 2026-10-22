from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FoursquareAdapterConfig:
    raw_dir: str = "foursquare_global_2015"
    archive_path: str | None = None
    global_checkins_filename: str = "dataset_TIST2015_Checkins.txt"
    global_pois_filename: str = "dataset_TIST2015_POIs.txt"
    global_cities_filename: str = "dataset_TIST2015_Cities.txt"
    global_batch_size: int = 250_000
    min_quality_score: float = 0.0
    drop_missing_coordinates: bool = True

    def validate(self) -> None:
        if not self.raw_dir:
            raise ValueError("data.foursquare.raw_dir must be non-empty")
        if self.archive_path is not None and not self.archive_path:
            raise ValueError("data.foursquare.archive_path must be non-empty")
        if not 0.0 <= self.min_quality_score <= 1.0:
            raise ValueError(
                "data.foursquare.min_quality_score must be in [0, 1]"
            )
        if (
            not self.global_checkins_filename
            or not self.global_pois_filename
            or not self.global_cities_filename
        ):
            raise ValueError("global Foursquare filenames must be non-empty")
        if self.global_batch_size <= 0:
            raise ValueError(
                "data.foursquare.global_batch_size must be positive"
            )


@dataclass
class DataConfig:
    dataset_name: str = "foursquare"
    cities: list[str] = field(
        default_factory=lambda: [
            "los_angeles",
            "washington_dc",
            "atlanta",
            "san_francisco",
            "milwaukee",
        ]
    )
    data_root: str = "data"
    foursquare: FoursquareAdapterConfig = field(
        default_factory=FoursquareAdapterConfig
    )

    def validate(self) -> None:
        if self.dataset_name != "foursquare":
            raise ValueError(
                "data.dataset_name must be 'foursquare'; "
                f"got {self.dataset_name!r}"
            )
        self.foursquare.validate()
