from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PreprocessingPlacesIndexConfig:
    path: str | None = None


@dataclass
class PreprocessingConfig:
    places_index: PreprocessingPlacesIndexConfig = field(
        default_factory=PreprocessingPlacesIndexConfig
    )
