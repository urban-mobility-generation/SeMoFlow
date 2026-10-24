from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    device: str = "cpu"
    mixed_precision: str = "bf16"
    num_workers: int = 0
    seed: int = 42
    preprocessing_dir: str = "outputs/preprocessing"
    places_index_path: str | None = None
    processed_visits_path: str | None = None
    out_dir: str = "outputs/tokenization"
    eval_k: int = 10
    eval_sample_size: int = 5000
    eval_pair_sample_size: int = 2_000_000
    eval_geometry_sample_size: int = 5000
    eval_geometry_lcp_pair_sample_size: int = 10_000
    eval_bootstrap_resamples: int = 200
    eval_bootstrap_confidence: float = 0.95
    skip_artifact_export: bool = False
    skip_semantic_id_eval: bool = False
    text_embeddings_path: str = "outputs/encoder/text/text_embeddings.npy"
    graphsage_artifact_dir: str | None = None
    input_embedding_source: str = "pca_concat"
    pca_text_dim: int | None = None
    pca_graph_dim: int | None = None
    geo_relative_dim: int | None = None
    geo_scale: float | None = None
    text_graph_scale: float | None = None

    def validate(self) -> None:
        if self.input_embedding_source not in {"pca_text", "pca_concat"}:
            raise ValueError(
                "tokenizer_config.input_embedding_source must be 'pca_text' "
                f"or 'pca_concat', got {self.input_embedding_source!r}"
            )
        if self.n_encoder_layers < 1:
            raise ValueError("tokenizer_config.n_encoder_layers must be >= 1")
        if self.lambda_graph < 0 or self.lambda_prefix_graph < 0:
            raise ValueError(
                "tokenizer_config.lambda_graph and lambda_prefix_graph must "
                "be non-negative"
            )
        if self.pca_text_dim is not None and self.pca_text_dim <= 0:
            raise ValueError(
                "tokenizer_config.pca_text_dim must be positive, got "
                f"{self.pca_text_dim!r}"
            )
        if self.geo_relative_dim is not None:
            if self.geo_relative_dim < 2 or (self.geo_relative_dim - 2) % 4:
                raise ValueError(
                    "tokenizer_config.geo_relative_dim must be 2 (raw dx, dy) "
                    "or 2 + 4F for F sin/cos wavelength pairs, got "
                    f"{self.geo_relative_dim!r}"
                )
            if self.input_embedding_source != "pca_concat":
                raise ValueError(
                    "tokenizer_config.geo_relative_dim requires "
                    "input_embedding_source='pca_concat', got "
                    f"{self.input_embedding_source!r}"
                )
        if self.geo_scale is not None:
            if self.geo_scale <= 0:
                raise ValueError(
                    "tokenizer_config.geo_scale must be positive, got "
                    f"{self.geo_scale!r}"
                )
            if self.geo_relative_dim is None:
                raise ValueError(
                    "tokenizer_config.geo_scale requires geo_relative_dim; "
                    "a scale with no geo block silently does nothing"
                )
        if self.pca_graph_dim is not None:
            if self.pca_graph_dim <= 0:
                raise ValueError(
                    "tokenizer_config.pca_graph_dim must be positive, got "
                    f"{self.pca_graph_dim!r}"
                )
            if self.input_embedding_source != "pca_concat":
                raise ValueError(
                    "tokenizer_config.pca_graph_dim requires "
                    "input_embedding_source='pca_concat', got "
                    f"{self.input_embedding_source!r}"
                )
        if self.text_graph_scale is not None and self.text_graph_scale <= 0:
            raise ValueError(
                "tokenizer_config.text_graph_scale must be positive, got "
                f"{self.text_graph_scale!r}"
            )

    def run_dir(self) -> Path:
        return Path(self.out_dir)

    def places_index_output_path(self) -> Path:
        if self.places_index_path is not None:
            return Path(self.places_index_path)
        from semoflow.config.artifacts import PreprocessingArtifacts

        return PreprocessingArtifacts.from_run_dir(
            Path(self.preprocessing_dir)
        ).places_index

    def processed_visits_output_path(self) -> Path:
        if self.processed_visits_path is not None:
            return Path(self.processed_visits_path)
        from semoflow.config.artifacts import PreprocessingArtifacts

        return PreprocessingArtifacts.from_run_dir(
            Path(self.preprocessing_dir)
        ).processed_visits


__all__ = ["TokenizerConfig"]
