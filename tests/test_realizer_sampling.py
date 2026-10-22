import torch

from semoflow.realizer import sampling


def test_nearest_decoder_bounds_candidate_pairs(monkeypatch) -> None:
    codes = torch.stack(
        [torch.ones(1000, dtype=torch.long), torch.arange(1000)], dim=1
    )
    codebooks = torch.zeros(2, 1000, 1)
    codebooks[1, :, 0] = torch.arange(1000)
    catalog = sampling.PrefixTailCatalog.from_codes(
        codes, codebooks, prefix_depth=1
    )
    candidate_pairs: list[int] = []
    original = sampling._candidate_index_block

    def record_pairs(left, right, query_rows):
        indices, legal = original(left, right, query_rows)
        candidate_pairs.append(indices.numel())
        return indices, legal

    monkeypatch.setattr(sampling, "_candidate_index_block", record_pairs)

    result = catalog.decode(
        torch.ones(1, 256, 1, dtype=torch.long),
        torch.zeros(1, 256, 1),
        torch.ones(1, 256, dtype=torch.bool),
        pad_id=-1,
    )

    assert max(candidate_pairs) <= sampling._CATALOG_DECODE_MAX_CANDIDATE_PAIRS
    assert result[..., 0].eq(1).all()
    assert result[..., 1].eq(0).all()


def test_batched_decoding_preserves_mixed_prefixes_and_padding(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sampling, "_CATALOG_DECODE_MAX_CANDIDATE_PAIRS", 16)
    codes = torch.tensor([[0, i] for i in range(8)] + [[1, 0], [1, 1]])
    codebooks = torch.zeros(2, 8, 1)
    codebooks[1, :, 0] = torch.arange(8)
    catalog = sampling.PrefixTailCatalog.from_codes(
        codes, codebooks, prefix_depth=1
    )

    result = catalog.decode(
        torch.tensor([[[0], [0], [0]], [[1], [1], [1]]]),
        torch.tensor([[[5.1], [6.9], [1.6]], [[6.9], [-0.3], [1.6]]]),
        torch.tensor([[True, True, True], [True, True, False]]),
        pad_id=-1,
    )

    torch.testing.assert_close(
        result,
        torch.tensor([[[0, 5], [0, 7], [0, 2]], [[1, 1], [1, 0], [-1, -1]]]),
    )
