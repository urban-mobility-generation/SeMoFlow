from pathlib import Path

import polars as pl
import pytest
import torch

from semoflow.encoder.artifacts import (
    load_poi_embedding_artifact,
    write_poi_embedding_artifact,
)
from semoflow.encoder.text.assembly import assemble_text_embeddings


def test_text_assembly_aligns_independently_numbered_catalogs(
    tmp_path: Path,
) -> None:
    source = pl.DataFrame(
        {"placekey": ["A", "B"], "placekey_id": [1, 2], "city_id": [0, 0]}
    )
    source_path = tmp_path / "source.npy"
    write_poi_embedding_artifact(
        source_path,
        torch.tensor([[10.0], [20.0]]),
        source,
        kind="text",
        metadata={"include_address_in_template": False},
    )
    target = pl.DataFrame(
        {"placekey": ["B", "A"], "placekey_id": [1, 2], "city_id": [0, 0]}
    )
    target_path = tmp_path / "places.parquet"
    target.write_parquet(target_path)
    output = tmp_path / "assembled.npy"

    result = assemble_text_embeddings(target_path, {0: source_path}, output)

    artifact = load_poi_embedding_artifact(output)
    assert result.identity == "placekey"
    assert {"placekey", "placekey_id"} <= set(artifact.index.columns)
    torch.testing.assert_close(
        artifact.align_to(target), torch.tensor([[20.0], [10.0]])
    )


def test_cross_catalog_assembly_rejects_numeric_only_sidecars(
    tmp_path: Path,
) -> None:
    source = pl.DataFrame({"placekey_id": [1]})
    source_path = tmp_path / "source.npy"
    write_poi_embedding_artifact(
        source_path,
        torch.tensor([[10.0]]),
        source,
        kind="text",
        metadata={"include_address_in_template": False},
    )
    places = pl.DataFrame(
        {"placekey": ["A"], "placekey_id": [1], "city_id": [0]}
    )
    places_path = tmp_path / "places.parquet"
    places.write_parquet(places_path)

    with pytest.raises(ValueError, match="stable placekey"):
        assemble_text_embeddings(
            places_path, {0: source_path}, tmp_path / "assembled.npy"
        )


def test_numeric_only_artifacts_still_align_within_their_namespace(
    tmp_path: Path,
) -> None:
    artifact = write_poi_embedding_artifact(
        tmp_path / "graph.pt",
        torch.tensor([[10.0], [20.0]]),
        pl.DataFrame({"placekey_id": [1, 2]}),
        kind="graph",
    )

    torch.testing.assert_close(
        artifact.align_to(pl.DataFrame({"placekey_id": [2, 1]})),
        torch.tensor([[20.0], [10.0]]),
    )
