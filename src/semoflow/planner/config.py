from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OccurrencePlannerConfig:
    codebook_size: int
    gap_vocab: int
    gap_codec_version: str
    gap_bucket_edges_minutes: tuple[float, ...]
    gap_duration_minutes: tuple[float, ...]
    observation_window_minutes: float
    prefix_depth: int = 3
    max_occurrences: int = 128
    model_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    city_conditioning: bool = False
    mobility_gate_prior: str = "log_distinct"
    use_distance_prior: bool = False
