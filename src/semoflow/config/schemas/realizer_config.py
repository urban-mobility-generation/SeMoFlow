from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefixTimeSuffixLatentFlowRealizerConfig:
    """Hard-prefix latent-flow realizer over whitened suffix tails, decoded
    by the calibrated legal-catalog posterior."""

    max_events: int = 32
    prefix_depth: int = 3
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
    num_workers: int = 0
    prefetch_factor: int = 2
    cpu_threads: int = 4
    device: str = "cuda:0"
    seed: int = 0
    compile: bool = False
    mixed_precision: str = "none"
    tf32: bool = False
    fused_optimizer: bool = False
    city_vocab: int = 0
    tail_whitening_shrinkage: float = 0.05
    tail_whitening_chunk_size: int = 262144
    catalog_loss_weight: float = 0.1
    catalog_temperature: float = 1.0
    catalog_candidate_count: int = 64
    endpoint_loss_weight: float = 0.0
    population_loss_weight: float = 0.0
    catalog_prior_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.mixed_precision not in {"none", "bf16", "fp16"}:
            raise ValueError(
                "mixed_precision must be 'none', 'bf16', or 'fp16'"
            )
        if self.lr_scheduler not in {"constant", "cosine"}:
            raise ValueError("lr_scheduler must be 'constant' or 'cosine'")
        if not 0.0 <= self.min_lr <= self.learning_rate:
            raise ValueError("min_lr must be between zero and learning_rate")
        if self.lr_scheduler == "constant" and self.min_lr != 0.0:
            raise ValueError("min_lr requires the cosine LR scheduler")
        if self.catalog_loss_weight <= 0.0:
            raise ValueError(
                "catalog_loss_weight must be positive; it trains the "
                "calibrated catalog decoder"
            )
        if not 0.0 <= self.tail_whitening_shrinkage <= 1.0:
            raise ValueError("tail_whitening_shrinkage must be in [0, 1]")


__all__ = ["PrefixTimeSuffixLatentFlowRealizerConfig"]
