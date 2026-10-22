from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from semoflow.config.artifacts import PreprocessingArtifacts
from semoflow.utils import write_run_manifest

if TYPE_CHECKING:
    from semoflow.config import ExperimentConfig

logger = logging.getLogger(__name__)

__all__ = [
    "PLACE_INDEX_COLUMNS",
    "PreprocessWorkspace",
    "log_pipeline_done",
    "run_pipeline",
]


PLACE_INDEX_COLUMNS = (
    "placekey",
    "location_name",
    "top_category",
    "sub_category",
    "naics_code",
    "street_address",
    "city",
    "state",
    "zipcode",
    "census_block_group",
    "brands",
    "geohash_5",
    "latitude",
    "longitude",
)


@dataclass(frozen=True)
class PreprocessWorkspace:
    cfg: ExperimentConfig
    data_root: Path
    output_dir: Path
    artifacts: PreprocessingArtifacts

    @classmethod
    def from_config(
        cls,
        cfg: ExperimentConfig,
    ) -> PreprocessWorkspace:
        return cls(
            cfg=cfg,
            data_root=Path(cfg.data.data_root).expanduser().resolve(),
            output_dir=Path(cfg.output.output_dir).expanduser().resolve(),
            artifacts=PreprocessingArtifacts.from_config(cfg),
        )

    def prepare_output(self) -> Path:
        self.artifacts.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts.vocabs_dir.mkdir(exist_ok=True)
        config_path = self.artifacts.run_dir / self.cfg.output.config_filename
        self.cfg.save_yaml(config_path, phase="preprocessing")
        logger.info("resolved config written  path=%s", config_path)
        write_run_manifest(
            self.artifacts.run_dir,
            config_path=config_path,
            output_dir=self.artifacts.run_dir,
            data_artifact_paths={"data_root": self.data_root},
            metric_artifact_paths={
                "processed_visits": self.artifacts.processed_visits,
                "quality_scores": self.artifacts.quality_scores,
                "places_index": self.artifacts.places_index,
            },
        )
        return config_path


def run_pipeline(cfg: ExperimentConfig) -> None:
    cfg.data.validate()
    from semoflow.preprocess.foursquare import run_foursquare_pipeline

    run_foursquare_pipeline(cfg)


def log_pipeline_done(t0: float, out: Path) -> None:
    logger.info(
        "pipeline done  total_elapsed=%.1fs  out_dir=%s",
        time.perf_counter() - t0,
        out,
    )
