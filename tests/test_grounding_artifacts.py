from pathlib import Path

import pytest

from semoflow.tokenization.decoding.grounding_index import (
    SemanticIDGroundingIndex,
)

ARTIFACTS = sorted(
    (Path(__file__).resolve().parents[1] / "veraset_checkpoints").glob(
        "*/grounding.pkl"
    )
)


@pytest.mark.parametrize(
    "path", ARTIFACTS, ids=[p.parent.name for p in ARTIFACTS]
)
def test_bundled_grounding_index_loads(path: Path) -> None:
    index = SemanticIDGroundingIndex.load(path)

    assert index.n_keys > 0
    assert index.n_pois >= index.n_keys
    key, payloads = next(iter(index.iter_key_payload_groups()))
    assert index.candidate_count(key) == len(payloads)
