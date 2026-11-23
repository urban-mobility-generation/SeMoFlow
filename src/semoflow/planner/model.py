from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from semoflow.config import PlannerConfig
from semoflow.planner.prefix_decoder import (
    HierarchicalPrefixDecoder,
    LegalPrefixIndex,
    PrefixDistanceContext,
    PrefixDistancePrior,
    sample_categorical_logits,
)
from semoflow.planner.router import (
    ExploreRetrieveRouter,
    explore_retrieve_target_log_probability,
    prefix_target_log_probability,
    retrieval_target_log_probability,
)
from semoflow.planner.transformer import (
    SpatioTemporalRelationBias,
    SpatioTemporalTransformer,
)
from semoflow.utils import load_artifact, save_torch_atomic


@dataclass(frozen=True, slots=True)
class OccurrencePlannerConfig:
    codebook_size: int
    gap_vocab: int
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


@dataclass(frozen=True, slots=True)
class OccurrenceBatch:
    previous_prefix_codes: torch.Tensor
    previous_gap_tokens: torch.Tensor
    previous_event_time_minutes: torch.Tensor
    target_prefix_codes: torch.Tensor
    target_gap_tokens: torch.Tensor
    valid_mask: torch.Tensor
    lengths: torch.Tensor
    city_ids: torch.Tensor

    def _map_tensors(
        self, transform: Callable[[torch.Tensor], torch.Tensor]
    ) -> OccurrenceBatch:
        return OccurrenceBatch(
            **{
                field.name: transform(getattr(self, field.name))
                for field in dataclasses.fields(self)
            }
        )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> OccurrenceBatch:
        return self._map_tensors(
            lambda tensor: tensor.to(device, non_blocking=non_blocking)
        )

    def pin_memory(self) -> OccurrenceBatch:
        return self._map_tensors(torch.Tensor.pin_memory)


@dataclass(frozen=True, slots=True)
class OccurrencePlannerOutput:
    length_logits: torch.Tensor
    gap_logits: torch.Tensor
    prefix_log_probability: torch.Tensor


@dataclass(frozen=True, slots=True)
class OccurrencePlannerLoss:
    total: torch.Tensor
    length: torch.Tensor
    gap: torch.Tensor
    prefix: torch.Tensor


_DAY_MINUTES = 1440.0


_WEEK_MINUTES = 10080.0


_DAILY_HARMONICS = 4


_WEEKLY_HARMONICS = 2


TIME_FEATURE_DIM = 2 * (_DAILY_HARMONICS + _WEEKLY_HARMONICS) + 2


def time_features(
    event_time_minutes: torch.Tensor,
    *,
    observation_window_minutes: float,
) -> torch.Tensor:
    minutes = event_time_minutes.float()
    day_phase = (
        2.0 * math.pi * torch.remainder(minutes, _DAY_MINUTES) / _DAY_MINUTES
    )
    week_phase = (
        2.0 * math.pi * torch.remainder(minutes, _WEEK_MINUTES) / _WEEK_MINUTES
    )
    features: list[torch.Tensor] = []
    for harmonic in range(1, _DAILY_HARMONICS + 1):
        features.append(torch.sin(harmonic * day_phase))
        features.append(torch.cos(harmonic * day_phase))
    for harmonic in range(1, _WEEKLY_HARMONICS + 1):
        features.append(torch.sin(harmonic * week_phase))
        features.append(torch.cos(harmonic * week_phase))
    elapsed_fraction = torch.clamp(
        minutes / observation_window_minutes, 0.0, 1.0
    )
    features.append(elapsed_fraction)
    features.append(1.0 - elapsed_fraction)
    return torch.stack(features, dim=-1)


def _gap_duration_buffers(
    config: OccurrencePlannerConfig,
) -> dict[str, torch.Tensor]:
    edges = torch.tensor(config.gap_bucket_edges_minutes, dtype=torch.float32)
    return {
        "gap_duration_lower_minutes": torch.cat((torch.zeros(1), edges)),
        "gap_representative_minutes": torch.tensor(
            config.gap_duration_minutes, dtype=torch.float32
        ),
    }


class OccurrenceARPlanner(nn.Module):
    def __init__(
        self,
        config: OccurrencePlannerConfig,
        legal_prefixes: torch.Tensor,
        *,
        catalog_city_ids: torch.Tensor,
        city_legal_prefix_mask: torch.Tensor,
        legal_prefix_centroids: torch.Tensor | None = None,
        legal_prefix_centroid_available: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.legality = LegalPrefixIndex(
            legal_prefixes,
            codebook_size=config.codebook_size,
            prefix_depth=config.prefix_depth,
            city_legal_prefix_mask=city_legal_prefix_mask,
        )
        self.register_buffer(
            "catalog_city_ids",
            catalog_city_ids.clone(),
            persistent=False,
        )
        self.register_buffer(
            "city_legal_prefix_mask",
            city_legal_prefix_mask.clone(),
            persistent=False,
        )

        self.distance_prior: PrefixDistancePrior | None = None
        if config.use_distance_prior:
            self.distance_prior = PrefixDistancePrior(
                legality=self.legality,
                centroids=legal_prefix_centroids,
                centroid_available=legal_prefix_centroid_available,
            )

        self.city_embedding = (
            nn.Embedding(catalog_city_ids.numel(), config.model_dim)
            if config.city_conditioning
            else None
        )
        self.prefix_embeddings = nn.ModuleList(
            nn.Embedding(config.codebook_size + 1, config.model_dim)
            for _ in range(config.prefix_depth)
        )
        self.gap_embedding = nn.Embedding(
            config.gap_vocab + 1, config.model_dim
        )
        self.length_embedding = nn.Embedding(
            config.max_occurrences, config.model_dim
        )
        self.bos_embedding = nn.Parameter(torch.zeros(config.model_dim))
        self.length_query = nn.Parameter(torch.zeros(config.model_dim))
        self.time_projection = nn.Sequential(
            nn.Linear(TIME_FEATURE_DIM, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        for name, buffer in _gap_duration_buffers(config).items():
            self.register_buffer(name, buffer, persistent=False)
        self.relation_bias = SpatioTemporalRelationBias(
            n_heads=config.n_heads,
            prefix_depth=config.prefix_depth,
        )
        self.transformer = SpatioTemporalTransformer(
            model_dim=config.model_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
            max_seq_len=config.max_occurrences,
        )
        self.length_head = nn.Linear(config.model_dim, config.max_occurrences)
        self.gap_head = nn.Linear(config.model_dim, config.gap_vocab)
        self.gap_condition = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.prefix_decoder = HierarchicalPrefixDecoder(
            model_dim=config.model_dim,
            codebook_size=config.codebook_size,
            prefix_depth=config.prefix_depth,
            legality=self.legality,
        )
        self.mobility_router = ExploreRetrieveRouter(
            model_dim=config.model_dim,
            gate_prior=config.mobility_gate_prior,
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {}

    def set_extra_state(self, state: object) -> None:
        pass

    def catalog_rows(self, city_ids: torch.Tensor) -> torch.Tensor:
        return torch.searchsorted(self.catalog_city_ids, city_ids.contiguous())

    def _add_city_state(
        self, hidden: torch.Tensor, city_ids: torch.Tensor
    ) -> torch.Tensor:
        if self.city_embedding is None:
            return hidden
        city = self.city_embedding(self.catalog_rows(city_ids))
        return hidden + city.view(
            city.shape[0], *([1] * (hidden.ndim - 2)), city.shape[-1]
        )

    def _time_state_embedding(
        self, event_time_minutes: torch.Tensor
    ) -> torch.Tensor:
        features = time_features(
            event_time_minutes,
            observation_window_minutes=self.config.observation_window_minutes,
        ).to(dtype=self.length_query.dtype)
        return self.time_projection(features)

    def _mask_gap_logits_by_horizon(
        self,
        gap_logits: torch.Tensor,
        earliest_previous_time_minutes: torch.Tensor,
    ) -> torch.Tensor:
        window = self.config.observation_window_minutes
        latest_duration = window - earliest_previous_time_minutes
        lower = self.gap_duration_lower_minutes.float()
        before_horizon = earliest_previous_time_minutes < window
        illegal = (latest_duration[..., None] <= lower) & before_horizon[
            ..., None
        ]
        return gap_logits.masked_fill(illegal, float("-inf"))

    def _earliest_previous_event_time(
        self, previous_gap_tokens: torch.Tensor
    ) -> torch.Tensor:
        safe_tokens = previous_gap_tokens.clamp(min=0)
        lower = self.gap_duration_lower_minutes[safe_tokens].float()
        lower = lower.masked_fill(previous_gap_tokens < 0, 0.0)
        return lower.cumsum(dim=-1)

    def _reconstruct_arrival_time(
        self,
        previous_time_minutes: torch.Tensor,
        gap_tokens: torch.Tensor,
    ) -> torch.Tensor:
        window = self.config.observation_window_minutes
        representatives = self.gap_representative_minutes[gap_tokens].float()
        return (previous_time_minutes + representatives).clamp(max=window)

    def _condition(
        self,
        hidden: torch.Tensor,
        gap_tokens: torch.Tensor,
        arrival_time_minutes: torch.Tensor,
    ) -> torch.Tensor:
        conditioned = hidden + self.gap_condition(
            self.gap_embedding(gap_tokens)
        )
        return conditioned + self._time_state_embedding(arrival_time_minutes)

    def predict_length(self, city_ids: torch.Tensor) -> torch.Tensor:
        query = self.length_query.expand(city_ids.shape[0], -1)
        return self.length_head(self._add_city_state(query, city_ids))

    def _embed_events(
        self, prefix_codes: torch.Tensor, gap_tokens: torch.Tensor
    ) -> torch.Tensor:
        prefix_ids = torch.where(
            prefix_codes >= 0,
            prefix_codes,
            torch.full_like(prefix_codes, self.config.codebook_size),
        )
        gap_ids = torch.where(
            gap_tokens >= 0,
            gap_tokens,
            torch.full_like(gap_tokens, self.config.gap_vocab),
        )
        level_states = [
            embedding(prefix_ids[..., level])
            for level, embedding in enumerate(self.prefix_embeddings)
        ]
        prefix_state = torch.stack(level_states, dim=0).sum(dim=0)
        return prefix_state + self.gap_embedding(gap_ids)

    def _embed_history(
        self,
        previous_prefix_codes: torch.Tensor,
        previous_gap_tokens: torch.Tensor,
        previous_event_time_minutes: torch.Tensor,
        lengths: torch.Tensor,
        city_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._embed_events(previous_prefix_codes, previous_gap_tokens)
        bos = (previous_prefix_codes < 0).all(dim=-1)
        hidden = hidden + bos[..., None] * self.bos_embedding
        hidden = hidden + self.length_embedding(lengths - 1)[:, None, :]
        hidden = self._add_city_state(hidden, city_ids)
        return hidden + self._time_state_embedding(previous_event_time_minutes)

    def _distance_context(
        self, source_prefix_codes: torch.Tensor
    ) -> PrefixDistanceContext | None:
        if self.distance_prior is None:
            return None
        units, valid = self.distance_prior.source_state(source_prefix_codes)
        return PrefixDistanceContext(
            prior=self.distance_prior,
            source_units=units,
            source_valid=valid,
        )

    def _embed_retrieval_memory(
        self,
        previous_prefix_codes: torch.Tensor,
        previous_gap_tokens: torch.Tensor,
        previous_event_time_minutes: torch.Tensor,
    ) -> torch.Tensor:
        memory = self._embed_events(previous_prefix_codes, previous_gap_tokens)
        return memory + self._time_state_embedding(previous_event_time_minutes)

    def _encode_history(self, batch: OccurrenceBatch) -> torch.Tensor:
        previous_prefixes = batch.previous_prefix_codes
        hidden = self._embed_history(
            previous_prefixes,
            batch.previous_gap_tokens,
            batch.previous_event_time_minutes,
            batch.lengths,
            batch.city_ids,
        )
        relation_bias = self.relation_bias(
            previous_prefixes,
            batch.previous_event_time_minutes,
        )
        return self.transformer(
            hidden,
            relation_bias=relation_bias,
            valid_mask=batch.valid_mask,
        )

    def initialize_generation_cache(
        self,
        *,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        config = self.config
        return torch.zeros(
            (
                config.n_layers,
                2,
                batch_size,
                config.n_heads,
                max_length,
                config.model_dim // config.n_heads,
            ),
            device=device,
            dtype=dtype,
        )

    def _sample_prefix(
        self,
        conditioned: torch.Tensor,
        *,
        memory_prefix_codes: torch.Tensor,
        memory_gap_tokens: torch.Tensor,
        memory_event_time_minutes: torch.Tensor,
        catalog_rows: torch.Tensor,
        temperature: float,
        route_temperature: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        memory_states = self._embed_retrieval_memory(
            memory_prefix_codes, memory_gap_tokens, memory_event_time_minutes
        )
        routing = self.mobility_router.forward_step(
            conditioned, memory_states, memory_prefix_codes
        )
        explore_prefix = self.prefix_decoder.sample(
            conditioned,
            temperature=temperature,
            generator=generator,
            catalog_rows=catalog_rows,
            distance=self._distance_context(memory_prefix_codes[:, -1]),
        )
        no_history = ~routing.history_mask.any(dim=-1)
        retrieval_logits = routing.retrieval_log_probability.float().clone()
        retrieval_logits[no_history] = float("-inf")
        retrieval_logits[no_history, 0] = 0.0
        retrieval_choice = sample_categorical_logits(
            retrieval_logits, temperature=temperature, generator=generator
        )
        row = torch.arange(conditioned.shape[0], device=conditioned.device)
        retrieved_prefix = memory_prefix_codes[row, retrieval_choice]
        retrieve_probability = torch.sigmoid(
            routing.retrieve_logit / route_temperature
        )
        route_draw = torch.rand(
            retrieve_probability.shape,
            device=conditioned.device,
            generator=generator,
        )
        use_retrieve = route_draw < retrieve_probability
        return torch.where(
            use_retrieve[:, None], retrieved_prefix, explore_prefix
        )

    @torch.no_grad()
    def sample_step(
        self,
        history_prefix_codes: torch.Tensor,
        history_gap_tokens: torch.Tensor,
        history_event_time_minutes: torch.Tensor,
        *,
        lengths: torch.Tensor,
        city_ids: torch.Tensor,
        step: int,
        active: torch.Tensor,
        cache: torch.Tensor,
        temperature: float,
        route_temperature: float,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefixes = history_prefix_codes[active, : step + 1]
        gaps = history_gap_tokens[active, : step + 1]
        times = history_event_time_minutes[active, : step + 1]
        cities = city_ids[active]
        hidden = self._embed_history(
            prefixes[:, -1:],
            gaps[:, -1:],
            times[:, -1:],
            lengths[active],
            cities,
        )
        hidden = self.transformer.forward_step(
            hidden,
            active=active,
            cache=cache,
            relation_bias=self.relation_bias.step(prefixes, times),
        ).squeeze(1)
        gap_logits = self._mask_gap_logits_by_horizon(
            self.gap_head(hidden),
            self._earliest_previous_event_time(gaps)[:, -1],
        )
        gap_tokens = sample_categorical_logits(
            gap_logits, temperature=temperature, generator=generator
        )
        arrival = self._reconstruct_arrival_time(times[:, -1], gap_tokens)
        prefix = self._sample_prefix(
            self._condition(hidden, gap_tokens, arrival),
            memory_prefix_codes=prefixes,
            memory_gap_tokens=gaps,
            memory_event_time_minutes=times,
            catalog_rows=self.catalog_rows(cities),
            temperature=temperature,
            route_temperature=route_temperature,
            generator=generator,
        )
        return gap_tokens, prefix, arrival

    def forward(self, batch: OccurrenceBatch) -> OccurrencePlannerOutput:
        hidden = self._encode_history(batch)
        gap_logits = self._mask_gap_logits_by_horizon(
            self.gap_head(hidden),
            self._earliest_previous_event_time(batch.previous_gap_tokens),
        )
        arrival_time_minutes = self._reconstruct_arrival_time(
            batch.previous_event_time_minutes, batch.target_gap_tokens
        )
        conditioned = self._condition(
            hidden, batch.target_gap_tokens, arrival_time_minutes
        )

        catalog_rows = self.catalog_rows(batch.city_ids)
        safe_token_ids = (
            self.city_legal_prefix_mask[catalog_rows]
            .to(torch.int64)
            .argmax(dim=-1)
        )
        decoder_target_prefix_codes = torch.where(
            batch.valid_mask[..., None],
            batch.target_prefix_codes,
            self.legality.codes[safe_token_ids][:, None, :],
        )
        prefix_logits = self.prefix_decoder(
            conditioned,
            decoder_target_prefix_codes,
            catalog_rows=catalog_rows,
            distance=self._distance_context(batch.previous_prefix_codes),
        )
        memory_states = self._embed_retrieval_memory(
            batch.previous_prefix_codes,
            batch.previous_gap_tokens,
            batch.previous_event_time_minutes,
        )
        routing = self.mobility_router(
            conditioned, memory_states, batch.previous_prefix_codes
        )
        prefix_log_probability = explore_retrieve_target_log_probability(
            explore_target_log_probability=prefix_target_log_probability(
                prefix_logits, decoder_target_prefix_codes
            ),
            routing=routing,
            retrieval_target_log_probability=retrieval_target_log_probability(
                routing,
                batch.previous_prefix_codes,
                batch.target_prefix_codes,
            ),
        )
        return OccurrencePlannerOutput(
            length_logits=self.predict_length(batch.city_ids),
            gap_logits=gap_logits,
            prefix_log_probability=prefix_log_probability,
        )


def occurrence_planner_loss(
    output: OccurrencePlannerOutput, batch: OccurrenceBatch
) -> OccurrencePlannerLoss:
    valid = batch.valid_mask
    batch_size = batch.lengths.shape[0]
    length_loss = F.cross_entropy(output.length_logits, batch.lengths - 1)
    gap_loss = (
        F.cross_entropy(
            output.gap_logits[valid],
            batch.target_gap_tokens[valid],
            reduction="sum",
        )
        / batch_size
    )
    prefix_loss = -output.prefix_log_probability[valid].sum() / batch_size
    return OccurrencePlannerLoss(
        total=length_loss + gap_loss + prefix_loss,
        length=length_loss,
        gap=gap_loss,
        prefix=prefix_loss,
    )


CHECKPOINT_ARTIFACT_TYPE = "occurrence_planner_checkpoint"


def _cpu(
    tensor: torch.Tensor | None, dtype: torch.dtype
) -> torch.Tensor | None:
    return None if tensor is None else tensor.detach().to("cpu", dtype).clone()


def save_occurrence_planner_checkpoint(
    path: Path,
    *,
    config: OccurrencePlannerConfig,
    legal_prefixes: torch.Tensor,
    catalog_city_ids: torch.Tensor,
    city_legal_prefix_mask: torch.Tensor,
    model_state: dict[str, Any],
    epoch: int,
    validation_metric: float,
    settings: PlannerConfig,
    legal_prefix_centroids: torch.Tensor | None,
    legal_prefix_centroid_available: torch.Tensor | None,
    force: bool,
) -> None:
    payload = {
        "artifact_type": CHECKPOINT_ARTIFACT_TYPE,
        "config": asdict(config),
        "legal_prefixes": _cpu(legal_prefixes, torch.long),
        "catalog_city_ids": _cpu(catalog_city_ids, torch.long),
        "city_legal_prefix_mask": _cpu(city_legal_prefix_mask, torch.bool),
        "legal_prefix_centroids": _cpu(legal_prefix_centroids, torch.float32),
        "legal_prefix_centroid_available": _cpu(
            legal_prefix_centroid_available, torch.bool
        ),
        "model_state": model_state,
        "epoch": int(epoch),
        "validation_metric": float(validation_metric),
        "settings": asdict(settings),
    }
    save_torch_atomic(path, payload, force=force)


_RETIRED_CONFIG = {
    "context_dim": 0,
    "context_schema": (),
    "prefix_decoder": "hierarchical",
    "mobility_router": "explore_retrieve",
    "use_tokenizer_latent": False,
    "gap_codec_version": "positive_duration_v1",
}


def load_occurrence_planner_checkpoint(
    path: Path,
) -> tuple[OccurrenceARPlanner, dict[str, Any]]:
    payload = load_artifact(path, CHECKPOINT_ARTIFACT_TYPE)
    config = dict(payload["config"])
    for key, supported in _RETIRED_CONFIG.items():
        if key in config and config.pop(key) != supported:
            raise ValueError(
                f"unsupported planner checkpoint: {key} != {supported!r}"
            )
    model = OccurrenceARPlanner(
        OccurrencePlannerConfig(**config),
        payload["legal_prefixes"],
        catalog_city_ids=payload["catalog_city_ids"],
        city_legal_prefix_mask=payload["city_legal_prefix_mask"],
        legal_prefix_centroids=payload["legal_prefix_centroids"],
        legal_prefix_centroid_available=payload[
            "legal_prefix_centroid_available"
        ],
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


__all__ = [
    "OccurrenceARPlanner",
    "OccurrenceBatch",
    "OccurrencePlannerConfig",
    "load_occurrence_planner_checkpoint",
    "occurrence_planner_loss",
    "save_occurrence_planner_checkpoint",
]
