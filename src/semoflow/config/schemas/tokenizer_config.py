from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    n_codebooks: int = 3
    codebook_size: int = 512
    hidden_dim: int = 256
    latent_dim: int = 64
    n_encoder_layers: int = 4
    epochs: int = 50
    batch_size: int = 1024
    lr: float = 0.001
    beta: float = 1.0
    lambda_graph: float = 0.0
    lambda_prefix_graph: float = 0.0
    sk_epsilons: list[float] | None = None
    device: str = "cuda:0"
    mixed_precision: str = "bf16"
    num_workers: int = 0
    seed: int = 42
    input_embedding_source: str = "pca_concat"
    pca_text_dim: int | None = None
    pca_graph_dim: int | None = None
    geo_relative_dim: int | None = None
    geo_scale: float | None = None
    text_graph_scale: float | None = None

    def validate(self) -> None:
        if self.input_embedding_source not in {"pca_text", "pca_concat"}:
            raise ValueError(
                "tokenizer.input_embedding_source must be 'pca_text' "
                f"or 'pca_concat', got {self.input_embedding_source!r}"
            )
        if self.n_encoder_layers < 1:
            raise ValueError("tokenizer.n_encoder_layers must be >= 1")
        if self.lambda_graph < 0 or self.lambda_prefix_graph < 0:
            raise ValueError(
                "tokenizer.lambda_graph and lambda_prefix_graph must "
                "be non-negative"
            )
        if self.pca_text_dim is not None and self.pca_text_dim <= 0:
            raise ValueError(
                "tokenizer.pca_text_dim must be positive, got "
                f"{self.pca_text_dim!r}"
            )
        if self.geo_relative_dim is not None:
            if self.geo_relative_dim < 2 or (self.geo_relative_dim - 2) % 4:
                raise ValueError(
                    "tokenizer.geo_relative_dim must be 2 (raw dx, dy) "
                    "or 2 + 4F for F sin/cos wavelength pairs, got "
                    f"{self.geo_relative_dim!r}"
                )
            if self.input_embedding_source != "pca_concat":
                raise ValueError(
                    "tokenizer.geo_relative_dim requires "
                    "input_embedding_source='pca_concat', got "
                    f"{self.input_embedding_source!r}"
                )
        if self.geo_scale is not None:
            if self.geo_scale <= 0:
                raise ValueError(
                    "tokenizer.geo_scale must be positive, got "
                    f"{self.geo_scale!r}"
                )
            if self.geo_relative_dim is None:
                raise ValueError(
                    "tokenizer.geo_scale requires geo_relative_dim; "
                    "a scale with no geo block silently does nothing"
                )
        if self.pca_graph_dim is not None:
            if self.pca_graph_dim <= 0:
                raise ValueError(
                    "tokenizer.pca_graph_dim must be positive, got "
                    f"{self.pca_graph_dim!r}"
                )
            if self.input_embedding_source != "pca_concat":
                raise ValueError(
                    "tokenizer.pca_graph_dim requires "
                    "input_embedding_source='pca_concat', got "
                    f"{self.input_embedding_source!r}"
                )
        if self.text_graph_scale is not None and self.text_graph_scale <= 0:
            raise ValueError(
                "tokenizer.text_graph_scale must be positive, got "
                f"{self.text_graph_scale!r}"
            )


__all__ = ["TokenizerConfig"]
