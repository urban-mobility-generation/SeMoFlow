from __future__ import annotations

import logging
from dataclasses import dataclass, field

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class QualityGateConfig:
    quality_event_min: float = 4.0
    quality_event_range: float = 60.0
    quality_span_days: float = 90.0
    quality_poi_range: float = 5.0

    min_quality_score: float = 0.3
    drop_missing_coords: bool = False

    event_coverage: float = 0.45
    temporal_span: float = 0.30
    poi_diversity: float = 0.25


@dataclass(frozen=True)
class QualityGate:
    config: QualityGateConfig = field(default_factory=QualityGateConfig)

    def apply(self, stays: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        scored_stays = self._drop_missing_coordinates(stays)
        stats = self._score_devices(scored_stays)
        qualified = stats.filter(
            pl.col("quality_score") >= self.config.min_quality_score
        ).select("device_key")
        filtered = scored_stays.join(
            qualified, on="device_key", how="semi", maintain_order="left"
        )

        logger.info(
            "quality_gate  devices_in=%d  devices_kept=%d  "
            "stays_in=%d  stays_kept=%d  threshold=%.2f",
            stats.height,
            qualified.height,
            scored_stays.height,
            filtered.height,
            self.config.min_quality_score,
        )
        return stats, filtered

    def _drop_missing_coordinates(self, stays: pl.DataFrame) -> pl.DataFrame:
        has_coords = (
            "latitude" in stays.columns and "longitude" in stays.columns
        )
        if not self.config.drop_missing_coords or not has_coords:
            return stays

        before = stays.height
        filtered = stays.filter(
            pl.col("latitude").is_not_null()
            & pl.col("longitude").is_not_null()
        )
        logger.info(
            "quality_gate  dropped %d stays with missing coords",
            before - filtered.height,
        )
        return filtered

    def _score_devices(self, stays: pl.DataFrame) -> pl.DataFrame:
        ts_col = "timestamp"
        poi_col = (
            "placekey_id"
            if "placekey_id" in stays.columns
            else ("placekey" if "placekey" in stays.columns else None)
        )
        poi_expr = pl.col(poi_col).n_unique() if poi_col else pl.lit(1)

        stats = stays.group_by("device_key", maintain_order=True).agg(
            [
                pl.len().cast(pl.Int32).alias("n_stays"),
                pl.col(ts_col).min().alias("first_stay"),
                pl.col(ts_col).max().alias("last_stay"),
                poi_expr.cast(pl.Int32).alias("n_unique_pois"),
            ]
        )
        stats = stats.with_columns(
            (
                (pl.col("last_stay") - pl.col("first_stay"))
                .dt.total_days()
                .cast(pl.Int32)
                + 1
            ).alias("temporal_span_days")
        )
        stats = stats.with_columns(
            [
                (
                    (
                        pl.col("n_stays").cast(pl.Float32)
                        - self.config.quality_event_min
                    )
                    / self.config.quality_event_range
                )
                .clip(lower_bound=0.0, upper_bound=1.0)
                .alias("s_event_coverage"),
                (
                    pl.col("temporal_span_days").cast(pl.Float32)
                    / self.config.quality_span_days
                )
                .clip(upper_bound=1.0)
                .alias("s_temporal_span"),
                (
                    pl.col("n_unique_pois").cast(pl.Float32)
                    / self.config.quality_poi_range
                )
                .clip(upper_bound=1.0)
                .alias("s_poi_diversity"),
            ]
        )
        return stats.with_columns(
            (
                pl.col("s_event_coverage") * self.config.event_coverage
                + pl.col("s_temporal_span") * self.config.temporal_span
                + pl.col("s_poi_diversity") * self.config.poi_diversity
            )
            .cast(pl.Float32)
            .alias("quality_score")
        )


def filter_quality(
    stays: pl.DataFrame,
    *,
    config: QualityGateConfig | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return QualityGate(config=config or QualityGateConfig()).apply(stays)


__all__ = ["QualityGate", "QualityGateConfig", "filter_quality"]
