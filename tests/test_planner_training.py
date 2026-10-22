import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import torch

from semoflow.planner.artifacts import OccurrencePlannerTrainingOptions
from semoflow.planner.dataset_build import build_occurrence_planner_dataset
from semoflow.planner.training import (
    OccurrencePlannerArchitecture,
    train_occurrence_planner,
)


def test_early_stopped_checkpoint_records_completed_epoch(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    segments = tmp_path / "segments.parquet"
    lookup = tmp_path / "lookup.parquet"
    dataset = tmp_path / "dataset.pt"
    start = datetime(2020, 1, 6)
    pl.DataFrame(
        {
            "placekey_id": [[1, 2], [2, 1], [1, 2]],
            "_timestamps": [[start, start + timedelta(hours=1)]] * 3,
            "_split": ["train", "train", "val"],
        }
    ).write_parquet(segments)
    pl.DataFrame({"placekey_id": [1, 2], "sem_code_0": [0, 1]}).write_parquet(
        lookup
    )
    build_occurrence_planner_dataset(
        segments_path=segments,
        lookup_path=lookup,
        output_path=dataset,
        codebook_size=2,
        prefix_depth=1,
        max_occurrences=4,
    )

    result = train_occurrence_planner(
        dataset_path=dataset,
        output_dir=tmp_path / "run",
        architecture=OccurrencePlannerArchitecture(
            model_dim=8, n_heads=2, n_layers=1, dropout=0.0
        ),
        options=OccurrencePlannerTrainingOptions(
            epochs=5,
            batch_size=2,
            learning_rate=1e-30,
            weight_decay=0.0,
            length_weight=1.0,
            gap_weight=1.0,
            prefix_weight=1.0,
            label_smoothing=0.0,
            device="cpu",
            early_stopping_patience=1,
        ),
        seed=0,
    )

    checkpoint = torch.load(result.last_checkpoint, weights_only=True)
    history = json.loads(result.metrics_path.read_text())["epochs"]
    assert len(history) == 2
    assert checkpoint["epoch"] == history[-1]["epoch"] == 2
