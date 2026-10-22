from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from semoflow.cli.preprocess import app
from semoflow.config import ExperimentConfig, load_experiment_config
from semoflow.preprocess.foursquare import FOURSQUARE_GLOBAL_CITY_NAMES
from semoflow.utils.splits import assign_split

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO_ROOT / "configs").rglob("*.yaml"))


def test_defaults_use_supported_foursquare_inputs() -> None:
    cfg = ExperimentConfig()

    assert cfg.data.dataset_name == "foursquare"
    assert set(cfg.data.cities) == set(FOURSQUARE_GLOBAL_CITY_NAMES)
    assert cfg.data.foursquare.raw_dir == "foursquare_global_2015"
    with pytest.raises(ValueError, match="must be 'foursquare'"):
        cfg.with_overrides(["data.dataset_name=unsupported"])


@pytest.mark.parametrize(
    "path", CONFIGS, ids=[str(p.relative_to(REPO_ROOT)) for p in CONFIGS]
)
def test_published_configs_load(path: Path) -> None:
    load_experiment_config(path).validate()


def test_preprocess_cli_preserves_checkins(tmp_path: Path) -> None:
    source = tmp_path / "release"
    source.mkdir()
    (source / "dataset_TIST2015_Cities.txt").write_text(
        "Atlanta\t33.749\t-84.388\tUS\tUnited States\tOther\n",
        encoding="utf-8",
    )
    (source / "dataset_TIST2015_POIs.txt").write_text(
        "venue-1\t33.749\t-84.388\tCoffee Shop\tUS\n",
        encoding="utf-8",
    )
    (source / "dataset_TIST2015_Checkins.txt").write_text(
        "".join(
            f"user-1\tvenue-1\tTue Apr 03 18:{minute:02d}:00 +0000 2012\t-240\n"
            for minute in (0, 5, 10)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "processed"

    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(REPO_ROOT / "configs/preprocess/foursquare.yaml"),
            "--set",
            f"data.foursquare.archive_path={source}",
            "--set",
            "data.cities=[atlanta]",
            "--out-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output or repr(result.exception)
    visits = pl.read_parquet(output / "processed_visits.parquet")
    places = pl.read_parquet(output / "places_index.parquet")
    scores = pl.read_parquet(output / "quality_scores.parquet")
    assert visits.height == 3
    assert visits["timestamp"].to_list() == [
        datetime(2012, 4, 3, 14, minute) for minute in (0, 5, 10)
    ]
    assert (
        visits["placekey_id"].to_list() == places["placekey_id"].to_list() * 3
    )
    assert visits["split"].unique().to_list() == [
        assign_split("foursquare:user-1")
    ]
    assert scores.height == 1
    manifest = json.loads(
        (output / "foursquare_source_manifest.json").read_text()
    )
    assert manifest["raw_row_counts"] == {"atlanta": 3}
    assert manifest["event_semantics"] == "checkin"
    resolved = load_experiment_config(output / "config_resolved.yaml")
    assert resolved.data.cities == ["atlanta"]
    assert (output / "provenance.json").is_file()
    assert (output / "run_manifest.json").is_file()
