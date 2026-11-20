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
    """Hard-prefix latent-flow realizer over whitened suffix tails, decoded
    by the calibrated legal-catalog posterior."""

    bank_path: Path
    codebooks_path: Path
    # SID lookup whose rows are the legal catalog candidates.
    catalog_path: Path
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
    tail_whitening_shrinkage: float = 0.05
    tail_whitening_chunk_size: int = 262144
    catalog_loss_weight: float = 0.1
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
        if self.epochs <= 0 or self.batch_size <= 0 or self.val_every <= 0:
            raise ValueError(
                "epochs, batch_size and val_every must be positive"
            )
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


__all__ = ["PrefixTimeSuffixLatentFlowRealizerConfig", "RealizerDataOptions"]
