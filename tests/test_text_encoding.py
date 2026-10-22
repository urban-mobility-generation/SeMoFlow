import numpy as np
import pytest
import torch

from semoflow.config.schemas.encoder_config import TextEncoderConfig
from semoflow.encoder.text import sentence_t5


@pytest.mark.parametrize(
    "precision, expected_dtype",
    [
        ("float32", torch.float32),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "[0,1]"])
def test_compute_precision_keeps_embedding_outputs_float32(
    monkeypatch, precision: str, expected_dtype, device: str
) -> None:
    class Encoder(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def encode(self, texts, *, precision="float32", **kwargs):
            assert precision == "float32"
            assert self.weight.dtype == expected_dtype
            return np.array([[1.0, 0.0]], dtype=np.float32)

        def start_multi_process_pool(self, **kwargs):
            return {}

        def stop_multi_process_pool(self, pool):
            pass

    monkeypatch.setattr(sentence_t5, "SentenceTransformer", Encoder)

    result = sentence_t5.encode_texts(
        ["Coffee shop"], "test-model", device, 1, precision=precision
    )

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [[1.0, 0.0]])


def test_text_config_rejects_unsupported_compute_precision() -> None:
    with pytest.raises(ValueError, match="precision"):
        TextEncoderConfig(precision="invalid").validate()
