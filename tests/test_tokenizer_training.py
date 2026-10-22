import torch

from semoflow.config.schemas.tokenizer_config import TokenizerConfig
from semoflow.tokenization.rqvae import build_tokenizer_model


def test_tokenizer_loss_is_reconstruction_plus_commitment() -> None:
    torch.manual_seed(0)
    config = TokenizerConfig(
        n_codebooks=2,
        codebook_size=4,
        hidden_dim=8,
        latent_dim=4,
        n_encoder_layers=1,
        beta=2.0,
    )
    model = build_tokenizer_model(config, input_dim=3)

    result = model({"x": torch.randn(8, 3)})

    assert result.losses is not None
    assert result.loss is not None
    torch.testing.assert_close(
        result.loss,
        result.losses["recon_loss"] + 2.0 * result.losses["vq_loss"],
    )
    result.loss.backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
