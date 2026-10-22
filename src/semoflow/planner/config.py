from __future__ import annotations

import math
from dataclasses import dataclass

from semoflow.planner.gap_codec import (
    validate_gap_codec_contract,
)
from semoflow.planner.mobility_router import (
    ExploreRetrieveRouter,
)


def validate_gap_duration_minutes(
    gap_duration_minutes: tuple[float, ...] | None,
    observation_window_minutes: float | None,
    *,
    gap_vocab: int,
) -> None:
    if (gap_duration_minutes is None) != (observation_window_minutes is None):
        raise ValueError(
            "gap_duration_minutes and observation_window_minutes must be "
            "set together or not at all"
        )
    if gap_duration_minutes is None:
        return
    if len(gap_duration_minutes) != gap_vocab:
        raise ValueError(
            "gap_duration_minutes must have one entry per gap_vocab token"
        )
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in gap_duration_minutes
    ):
        raise ValueError(
            "gap_duration_minutes must be finite and strictly positive"
        )
    assert observation_window_minutes is not None
    if (
        not math.isfinite(observation_window_minutes)
        or observation_window_minutes <= 0.0
    ):
        raise ValueError(
            "observation_window_minutes must be finite and positive"
        )


@dataclass(frozen=True, slots=True)
class OccurrencePlannerConfig:
    codebook_size: int
    gap_vocab: int
    prefix_depth: int = 3
    max_occurrences: int = 128
    model_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    context_dim: int = 0
    context_schema: tuple[str, ...] = ()
    city_conditioning: bool = False
    prefix_decoder: str = "hierarchical"
    mobility_router: str = "none"
    mobility_gate_prior: str = "log_distinct"
    use_tokenizer_latent: bool = False
    use_distance_prior: bool = False
    gap_codec_version: str | None = None
    gap_bucket_edges_minutes: tuple[float, ...] | None = None
    gap_duration_minutes: tuple[float, ...] | None = None
    observation_window_minutes: float | None = None

    def __post_init__(self) -> None:
        if self.codebook_size < 1 or self.gap_vocab < 1:
            raise ValueError("codebook_size and gap_vocab must be positive")
        if self.prefix_depth < 1 or self.max_occurrences < 1:
            raise ValueError(
                "prefix_depth and max_occurrences must be positive"
            )
        if self.model_dim < 1 or self.n_heads < 1 or self.n_layers < 1:
            raise ValueError("model dimensions and layers must be positive")
        if self.model_dim % self.n_heads:
            raise ValueError("model_dim must be divisible by n_heads")
        if (self.model_dim // self.n_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        if self.context_dim < 0:
            raise ValueError("context_dim must be non-negative")
        if len(self.context_schema) != self.context_dim:
            raise ValueError(
                "context_schema must have one name per context feature"
            )
        if len(set(self.context_schema)) != len(self.context_schema):
            raise ValueError("context_schema feature names must be unique")
        if not isinstance(self.city_conditioning, bool):
            raise ValueError("city_conditioning must be boolean")
        if not isinstance(self.use_tokenizer_latent, bool):
            raise ValueError("use_tokenizer_latent must be boolean")
        if not isinstance(self.use_distance_prior, bool):
            raise ValueError("use_distance_prior must be boolean")
        if self.prefix_decoder not in {"hierarchical", "flat"}:
            raise ValueError("prefix_decoder must be 'hierarchical' or 'flat'")
        if self.mobility_router not in {"none", "explore_retrieve"}:
            raise ValueError(
                "mobility_router must be 'none' or 'explore_retrieve'"
            )
        if self.mobility_gate_prior not in ExploreRetrieveRouter.gate_priors:
            raise ValueError(
                "mobility_gate_prior must be one of "
                + ", ".join(ExploreRetrieveRouter.gate_priors)
            )
        validate_gap_duration_minutes(
            self.gap_duration_minutes,
            self.observation_window_minutes,
            gap_vocab=self.gap_vocab,
        )
        validate_gap_codec_contract(
            version=self.gap_codec_version,
            bucket_edges_minutes=self.gap_bucket_edges_minutes,
            representative_minutes=self.gap_duration_minutes,
        )

    @property
    def temporal_state_enabled(self) -> bool:
        return self.gap_duration_minutes is not None
