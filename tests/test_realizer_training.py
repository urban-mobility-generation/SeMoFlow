from pathlib import Path

import polars as pl
import pytest
import torch

from semoflow.realizer.training import (
    PrefixTimeSuffixLatentFlowRealizerConfig,
    train_prefix_time_suffix_latent_flow_realizer,
)


@pytest.mark.parametrize(
    "scope, expected_mean", [("suffix", 2.0), ("full_sid", 17.0)]
)
def test_realizer_statistics_follow_the_latent_target(
    tmp_path: Path, scope: str, expected_mean: float
) -> None:
    bank = tmp_path / "bank.pt"
    codebooks = tmp_path / "codebooks.pt"
    torch.save(
        {
            "sid_codes": torch.tensor([[[0, 0]], [[1, 1]]] * 2),
            "time_tokens": torch.zeros(4, 1, dtype=torch.long),
            "valid_mask": torch.ones(4, 1, dtype=torch.bool),
            "row_counts": {"train": 2, "val": 2, "test": 0},
        },
        bank,
    )
    torch.save(torch.tensor([[[10.0], [20.0]], [[1.0], [3.0]]]), codebooks)
    output = tmp_path / "run"

    result = train_prefix_time_suffix_latent_flow_realizer(
        PrefixTimeSuffixLatentFlowRealizerConfig(
            bank_path=bank,
            codebooks_path=codebooks,
            output_dir=output,
            prefix_depth=1,
            codebook_size=2,
            model_dim=8,
            condition_depth=1,
            depth=1,
            num_heads=2,
            dropout=0.0,
            batch_size=2,
            epochs=1,
            cpu_threads=1,
            device="cpu",
            latent_target_scope=scope,
        )
    )

    assert result["epochs_completed"] == 1
    statistics = torch.load(output / "tail_stats.pt", weights_only=True)
    torch.testing.assert_close(
        statistics["mean"], torch.tensor([expected_mean])
    )


@pytest.mark.parametrize("width", [3, 8])
def test_catalog_objective_supports_variable_sid_depths(
    tmp_path: Path, width: int
) -> None:
    bank = tmp_path / "bank.pt"
    codebooks = tmp_path / "codebooks.pt"
    catalog = tmp_path / "catalog.parquet"
    torch.save(
        {
            "sid_codes": torch.tensor([[[0] * width], [[1] * width]] * 2),
            "time_tokens": torch.zeros(4, 1, dtype=torch.long),
            "valid_mask": torch.ones(4, 1, dtype=torch.bool),
            "row_counts": {"train": 2, "val": 2, "test": 0},
        },
        bank,
    )
    torch.save(torch.tensor([[[0.0], [1.0]]] * width), codebooks)
    pl.DataFrame(
        {f"sem_code_{i}": [0, 1] for i in range(width)}
    ).write_parquet(catalog)

    result = train_prefix_time_suffix_latent_flow_realizer(
        PrefixTimeSuffixLatentFlowRealizerConfig(
            bank_path=bank,
            codebooks_path=codebooks,
            output_dir=tmp_path / "run",
            prefix_depth=1,
            codebook_size=2,
            model_dim=8,
            condition_depth=1,
            depth=1,
            num_heads=2,
            dropout=0.0,
            batch_size=2,
            epochs=1,
            cpu_threads=1,
            device="cpu",
            catalog_path=catalog,
            catalog_loss_weight=0.1,
            catalog_negative_count=1,
        )
    )

    assert result["num_codebooks"] == width
    assert result["epochs_completed"] == 1
