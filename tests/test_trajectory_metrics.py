import math

import polars as pl
import pytest

from semoflow.metrics.semantic_id_trajectory import (
    trajectory_behavioral_metrics,
)
from semoflow.metrics.trajectory.mobility_motifs import mobility_motif_metrics


def test_temporal_coherence_samples_large_tuple_prefix_sets() -> None:
    lookup = pl.DataFrame(
        {
            "placekey_id": list(range(1, 403)),
            "sem_code_0": [g // 16 for g in range(201) for _ in range(2)],
            "sem_code_1": [g % 16 for g in range(201) for _ in range(2)],
        }
    )
    stays = pl.DataFrame(
        {
            "placekey_id": list(range(1, 403)),
            "device_key": ["device"] * 402,
            "hour": [12] * 402,
        }
    )

    result = trajectory_behavioral_metrics(
        stays,
        lookup,
        ["sem_code_0", "sem_code_1"],
        min_visits=1,
    )

    assert result["prefix_temporal_coherence_by_depth"][2] == pytest.approx(0)


def test_motifs_report_undefined_comparison_for_all_padding_output() -> None:
    result = mobility_motif_metrics([[1, 2]], [[0, 0]])

    assert result["real_day_count"] == 1
    assert result["generated_day_count"] == 0
    assert math.isnan(result["mobility_motif_jsd"])
    assert all(
        math.isnan(value)
        for value in result["generated_motif_frequencies"].values()
    )
    assert math.isnan(result["frequent_motif_mean_abs_frequency_error"])
