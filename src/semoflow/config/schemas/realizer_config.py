from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RealizerDataOptions:
    """How ``semoflow-artifacts build`` derives the realizer bank.

    The bank's SID lookup is the realizer's ``catalog_path``.
    """

    segments_path: Path
    max_events: int = 32


@dataclass(frozen=True, slots=True)
class PrefixTimeSuffixLatentFlowRealizerConfig:
    bank_path: Path
    codebooks_path: Path
    output_dir: Path
    data: RealizerDataOptions | None = None
    initialize_from: Path | None = None
    prefix_depth: int = 3
    # None reads the codebook size from codebooks_path.
    codebook_size: int | None = None
    time_vocab: int = 42
    latent_dim: int | None = None
    model_dim: int = 256
    condition_depth: int = 4
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 2e-4
    lr_scheduler: str = "constant"
    warmup_steps: int = 0
    min_lr: float = 0.0
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    train_max_samples: int | None = None
    val_max_samples: int | None = None
    num_workers: int = 0
    prefetch_factor: int = 2
    cpu_threads: int = 4
    device: str = "cuda"
    seed: int = 0
    compile: bool = False
    mixed_precision: str = "none"
    tf32: bool = False
    fused_optimizer: bool = False
    val_every: int = 1
    lognormal_t: bool = True
    city_vocab: int = 0
    spatial_conditioning: bool = False
    spatial_num_basis: int = 8
    spatial_max_km: float = 50.0
    places_path: Path | None = None
    prefix_embedding_mode: str = "learned"
    tail_representation: str = "summed"
    latent_target_scope: str = "suffix"
    proposal_corruption_rate: float = 0.0
    tail_normalization: str = "diagonal"
    tail_whitening_shrinkage: float = 0.05
    tail_whitening_chunk_size: int = 262144
    catalog_decoder_mode: str = "nearest"
    factorized_sum_consistency_weight: float = 0.0
    catalog_path: Path | None = None
    catalog_loss_weight: float = 0.0
    catalog_negative_count: int = 8
    catalog_temperature: float = 1.0
    catalog_candidate_count: int = 64
    endpoint_loss_weight: float = 0.0
    population_loss_weight: float = 0.0
    catalog_prior_weight: float = 0.25

    def __post_init__(self) -> None:
        if (
            self.initialize_from is not None
            and self.output_dir.resolve()
            == self.initialize_from.resolve().parent
        ):
            raise ValueError(
                "output_dir must differ from the warm-start checkpoint "
                "directory; adaptation would overwrite the parent artifacts"
            )
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.val_every <= 0:
            raise ValueError("val_every must be positive")
        if (
            self.num_workers < 0
            or self.prefetch_factor <= 0
            or self.cpu_threads <= 0
        ):
            raise ValueError("data-loader worker settings are invalid")
        if self.mixed_precision not in {"none", "bf16", "fp16"}:
            raise ValueError(
                "mixed_precision must be 'none', 'bf16', or 'fp16'"
            )
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer hyperparameters are invalid")
        if self.lr_scheduler not in {"constant", "cosine"}:
            raise ValueError("lr_scheduler must be 'constant' or 'cosine'")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 <= self.min_lr <= self.learning_rate:
            raise ValueError("min_lr must be between zero and learning_rate")
        if self.lr_scheduler == "constant" and self.min_lr != 0.0:
            raise ValueError("min_lr requires the cosine LR scheduler")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 1
        ):
            raise ValueError(
                "early_stopping_patience must be positive or None"
            )
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.prefix_embedding_mode not in {
            "learned",
            "hybrid_frozen_codebook",
        }:
            raise ValueError("prefix_embedding_mode is invalid")
        if self.tail_representation not in {"summed", "factorized"}:
            raise ValueError("tail_representation is invalid")
        if self.latent_target_scope not in {"suffix", "full_sid"}:
            raise ValueError("latent_target_scope is invalid")
        if (
            self.latent_target_scope == "full_sid"
            and self.tail_representation != "summed"
        ):
            raise ValueError("full-SID targets require summed representation")
        if not 0.0 <= self.proposal_corruption_rate <= 1.0:
            raise ValueError("proposal_corruption_rate must be in [0, 1]")
        if (
            self.latent_target_scope == "suffix"
            and self.proposal_corruption_rate > 0.0
        ):
            raise ValueError(
                "proposal corruption is valid only for full-SID latent targets"
            )
        if self.tail_normalization not in {"diagonal", "whitened"}:
            raise ValueError("tail_normalization is invalid")
        if (
            self.tail_normalization == "whitened"
            and self.tail_representation != "summed"
        ):
            raise ValueError(
                "whitened tail normalization requires summed tails"
            )
        if not 0.0 <= self.tail_whitening_shrinkage <= 1.0:
            raise ValueError("tail_whitening_shrinkage must be in [0, 1]")
        if self.tail_whitening_chunk_size <= 0:
            raise ValueError("tail_whitening_chunk_size must be positive")
        if self.catalog_decoder_mode not in {"nearest", "calibrated"}:
            raise ValueError("catalog_decoder_mode is invalid")
        if (
            self.catalog_decoder_mode == "calibrated"
            and self.tail_representation == "factorized"
        ):
            raise ValueError(
                "factorized tails are incompatible with calibrated catalog "
                "decoding"
            )
        if self.factorized_sum_consistency_weight < 0.0:
            raise ValueError(
                "factorized_sum_consistency_weight must be non-negative"
            )
        if self.catalog_loss_weight < 0.0:
            raise ValueError("catalog_loss_weight must be non-negative")
        if (
            self.catalog_decoder_mode == "calibrated"
            and self.catalog_loss_weight <= 0.0
        ):
            raise ValueError(
                "catalog_loss_weight must be positive for calibrated catalog "
                "decoding"
            )
        if self.catalog_loss_weight > 0.0 and self.catalog_path is None:
            raise ValueError(
                "catalog_path is required when catalog loss is enabled"
            )
        if self.proposal_corruption_rate > 0.0 and self.catalog_path is None:
            raise ValueError(
                "catalog_path is required when proposal corruption is enabled"
            )
        if (
            self.catalog_decoder_mode == "calibrated"
            and self.catalog_path is None
        ):
            raise ValueError(
                "catalog_path is required for calibrated decoding"
            )
        if self.spatial_conditioning and self.places_path is None:
            raise ValueError(
                "places_path is required when spatial_conditioning is enabled"
            )
        if self.spatial_conditioning and self.catalog_path is None:
            raise ValueError(
                "catalog_path is required when spatial_conditioning is enabled "
                "(prefix-bucket centroids are built from the catalog)"
            )
        if self.spatial_conditioning and self.spatial_num_basis < 1:
            raise ValueError("spatial_num_basis must be positive")
        if self.spatial_conditioning and self.spatial_max_km <= 0.0:
            raise ValueError("spatial_max_km must be positive")
        if self.catalog_negative_count <= 0 or self.catalog_temperature <= 0.0:
            raise ValueError(
                "catalog ranking hyperparameters must be positive"
            )
        if self.catalog_candidate_count <= 0:
            raise ValueError("catalog_candidate_count must be positive")
        if (
            self.endpoint_loss_weight < 0.0
            or self.population_loss_weight < 0.0
        ):
            raise ValueError("auxiliary loss weights must be non-negative")
        if self.catalog_prior_weight < 0.0:
            raise ValueError("catalog_prior_weight must be non-negative")


__all__ = ["PrefixTimeSuffixLatentFlowRealizerConfig", "RealizerDataOptions"]
