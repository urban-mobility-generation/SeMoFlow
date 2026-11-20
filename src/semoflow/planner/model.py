from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from semoflow.planner.config import OccurrencePlannerConfig
from semoflow.planner.decoders import (
    HierarchicalPrefixDecoder,
    PrefixDistanceContext,
    sample_categorical_logits,
)
from semoflow.planner.geography import PrefixDistancePrior
from semoflow.planner.legality import LegalPrefixIndex
from semoflow.planner.mobility_router import (
    ExploreRetrieveRouter,
    explore_retrieve_target_log_probability,
    prefix_target_log_probability,
    retrieval_target_log_probability,
)
from semoflow.planner.spatiotemporal_transformer import (
    SpatioTemporalRelationBias,
    SpatioTemporalTransformer,
)
from semoflow.planner.time_features import (
    TIME_FEATURE_DIM,
    time_features,
)
from semoflow.planner.types import (
    OccurrenceBatch,
    OccurrenceGenerationCache,
    OccurrencePlannerOutput,
    OccurrenceStepSample,
)


def _gap_duration_buffers(
    config: OccurrencePlannerConfig,
) -> dict[str, torch.Tensor]:
    """Per-gap-token duration bounds and representatives, in minutes."""
    edges = torch.tensor(config.gap_bucket_edges_minutes, dtype=torch.float32)
    window = torch.tensor(
        [config.observation_window_minutes], dtype=torch.float32
    )
    return {
        "gap_duration_lower_minutes": torch.cat((torch.zeros(1), edges)),
        "gap_duration_upper_minutes": torch.cat((edges, window)),
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

    # Released checkpoints carry an ``_extra_state`` entry; it holds no
    # information the model needs.
    def get_extra_state(self) -> dict[str, Any]:
        return {}

    def set_extra_state(self, state: object) -> None:
        pass

    def catalog_rows(self, city_ids: torch.Tensor) -> torch.Tensor:
        """Row of each city in the catalog's city-legality mask."""
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

    def condition_on_gap_and_time(
        self,
        hidden: torch.Tensor,
        gap_tokens: torch.Tensor,
        arrival_time_minutes: torch.Tensor,
    ) -> torch.Tensor:
        conditioned = self.condition_on_gap(hidden, gap_tokens)
        return conditioned + self._time_state_embedding(arrival_time_minutes)

    def predict_length(self, city_ids: torch.Tensor) -> torch.Tensor:
        query = self.length_query.expand(city_ids.shape[0], -1)
        return self.length_head(self._add_city_state(query, city_ids))

    def _embed_prefix_history(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        level_states = [
            embedding(prefix_ids[..., level])
            for level, embedding in enumerate(self.prefix_embeddings)
        ]
        return torch.stack(level_states, dim=0).sum(dim=0)

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
        prefix_ids = torch.where(
            previous_prefix_codes >= 0,
            previous_prefix_codes,
            torch.full_like(previous_prefix_codes, self.config.codebook_size),
        )
        gap_ids = torch.where(
            previous_gap_tokens >= 0,
            previous_gap_tokens,
            torch.full_like(previous_gap_tokens, self.config.gap_vocab),
        )
        memory = self._embed_prefix_history(prefix_ids)
        memory = memory + self.gap_embedding(gap_ids)
        return memory + self._time_state_embedding(previous_event_time_minutes)

    def initialize_generation_cache(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> OccurrenceGenerationCache:
        head_dim = self.config.model_dim // self.config.n_heads
        shape = (batch_size, self.config.n_heads, 0, head_dim)
        return OccurrenceGenerationCache(
            keys=tuple(
                torch.empty(shape, device=device, dtype=dtype)
                for _ in range(self.config.n_layers)
            ),
            values=tuple(
                torch.empty(shape, device=device, dtype=dtype)
                for _ in range(self.config.n_layers)
            ),
        )

    def _embed_history_step(
        self,
        batch: OccurrenceBatch,
        *,
        step: int,
        active: torch.Tensor,
    ) -> torch.Tensor:
        previous_prefixes = batch.previous_prefix_codes[
            active, step : step + 1
        ]
        previous_gaps = batch.previous_gap_tokens[active, step : step + 1]
        prefix_ids = torch.where(
            previous_prefixes >= 0,
            previous_prefixes,
            torch.full_like(previous_prefixes, self.config.codebook_size),
        )
        gap_ids = torch.where(
            previous_gaps >= 0,
            previous_gaps,
            torch.full_like(previous_gaps, self.config.gap_vocab),
        )
        hidden = self._embed_prefix_history(prefix_ids)
        hidden = hidden + self.gap_embedding(gap_ids)
        bos = (previous_prefixes < 0).all(dim=-1)
        hidden = hidden + bos[..., None] * self.bos_embedding
        hidden = (
            hidden
            + self.length_embedding(batch.lengths[active] - 1)[:, None, :]
        )
        hidden = self._add_city_state(hidden, batch.city_ids[active])
        previous_time = batch.previous_event_time_minutes[
            active, step : step + 1
        ]
        return hidden + self._time_state_embedding(previous_time)

    @torch.no_grad()
    def _encode_history_step(
        self,
        batch: OccurrenceBatch,
        *,
        step: int,
        active: torch.Tensor,
        cache: OccurrenceGenerationCache,
    ) -> tuple[torch.Tensor, OccurrenceGenerationCache]:
        batch_size = batch.previous_prefix_codes.shape[0]
        hidden = self._embed_history_step(batch, step=step, active=active)
        new_keys: list[torch.Tensor] = []
        new_values: list[torch.Tensor] = []
        head_dim = self.config.model_dim // self.config.n_heads
        relation_bias = self.relation_bias.step(
            batch.previous_prefix_codes[active],
            batch.previous_event_time_minutes[active],
            step=step,
        )
        for layer_index, layer in enumerate(self.transformer.layers):
            past_keys = cache.keys[layer_index]
            past_values = cache.values[layer_index]
            hidden, current_key, current_value = layer.forward_step(
                hidden,
                step=step,
                past_key=past_keys[active],
                past_value=past_values[active],
                relation_bias=relation_bias,
            )
            key_slot = torch.zeros(
                (batch_size, self.config.n_heads, 1, head_dim),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            value_slot = torch.zeros_like(key_slot)
            key_slot[active] = current_key
            value_slot[active] = current_value
            new_keys.append(torch.cat((past_keys, key_slot), dim=2))
            new_values.append(torch.cat((past_values, value_slot), dim=2))
        hidden = self.transformer.final_norm(hidden)
        return hidden.squeeze(1), OccurrenceGenerationCache(
            keys=tuple(new_keys), values=tuple(new_values)
        )

    def _encode_history(self, batch: OccurrenceBatch) -> torch.Tensor:
        previous_prefixes = batch.previous_prefix_codes
        prefix_ids = torch.where(
            previous_prefixes >= 0,
            previous_prefixes,
            torch.full_like(previous_prefixes, self.config.codebook_size),
        )
        gap_ids = torch.where(
            batch.previous_gap_tokens >= 0,
            batch.previous_gap_tokens,
            torch.full_like(batch.previous_gap_tokens, self.config.gap_vocab),
        )
        hidden = self._embed_prefix_history(prefix_ids)
        hidden = hidden + self.gap_embedding(gap_ids)
        bos = (previous_prefixes < 0).all(dim=-1)
        hidden = hidden + bos[..., None] * self.bos_embedding
        hidden = hidden + self.length_embedding(batch.lengths - 1)[:, None, :]
        hidden = self._add_city_state(hidden, batch.city_ids)
        hidden = hidden + self._time_state_embedding(
            batch.previous_event_time_minutes
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

    def condition_on_gap(
        self, hidden: torch.Tensor, gap_tokens: torch.Tensor
    ) -> torch.Tensor:
        return hidden + self.gap_condition(self.gap_embedding(gap_tokens))

    def _sample_routed_prefix(
        self,
        conditioned: torch.Tensor,
        *,
        memory_prefix_codes: torch.Tensor,
        memory_gap_tokens: torch.Tensor,
        memory_event_time_minutes: torch.Tensor,
        catalog_rows: torch.Tensor,
        distance: PrefixDistanceContext | None,
        prefix_temperature: float,
        route_temperature: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Sample a prefix by exploring the decoder or retrieving memory."""
        memory_states = self._embed_retrieval_memory(
            memory_prefix_codes,
            memory_gap_tokens,
            memory_event_time_minutes,
        )
        routing = self.mobility_router.forward_step(
            conditioned,
            memory_states,
            memory_prefix_codes,
        )
        explore_prefix = self.prefix_decoder.sample(
            conditioned,
            temperature=prefix_temperature,
            generator=generator,
            catalog_rows=catalog_rows,
            distance=distance,
        )

        history_available = routing.history_mask.any(dim=-1)
        safe_retrieval_logits = (
            routing.retrieval_log_probability.float().clone()
        )
        if bool((~history_available).any()):
            safe_retrieval_logits[~history_available] = float("-inf")
            safe_retrieval_logits[~history_available, 0] = 0.0
        retrieval_choice = sample_categorical_logits(
            safe_retrieval_logits,
            temperature=prefix_temperature,
            generator=generator,
        )
        row = torch.arange(
            conditioned.shape[0], device=memory_prefix_codes.device
        )
        retrieved_prefix = memory_prefix_codes[row, retrieval_choice]

        route_probability_policy = torch.sigmoid(
            routing.retrieve_logit / route_temperature
        ).masked_fill(~history_available, 0.0)
        route_draw = torch.rand(
            route_probability_policy.shape,
            device=conditioned.device,
            generator=generator,
        )
        use_retrieve = (
            route_draw < route_probability_policy
        ) & history_available
        return torch.where(
            use_retrieve[:, None], retrieved_prefix, explore_prefix
        )

    def _sample_from_hidden(
        self,
        hidden: torch.Tensor,
        *,
        previous_time_minutes: torch.Tensor,
        memory_prefix_codes: torch.Tensor,
        memory_gap_tokens: torch.Tensor,
        memory_event_time_minutes: torch.Tensor,
        catalog_rows: torch.Tensor,
        gap_temperature: float,
        prefix_temperature: float,
        route_temperature: float,
        generator: torch.Generator | None,
    ) -> OccurrenceStepSample:
        distance = self._distance_context(memory_prefix_codes[:, -1])
        earliest_previous_time = self._earliest_previous_event_time(
            memory_gap_tokens
        )[:, -1]
        gap_logits = self._mask_gap_logits_by_horizon(
            self.gap_head(hidden), earliest_previous_time
        )
        gap_tokens = sample_categorical_logits(
            gap_logits,
            temperature=gap_temperature,
            generator=generator,
        )
        arrival_time_minutes = self._reconstruct_arrival_time(
            previous_time_minutes, gap_tokens
        )
        conditioned = self.condition_on_gap_and_time(
            hidden, gap_tokens, arrival_time_minutes
        )
        prefix = self._sample_routed_prefix(
            conditioned,
            memory_prefix_codes=memory_prefix_codes,
            memory_gap_tokens=memory_gap_tokens,
            memory_event_time_minutes=memory_event_time_minutes,
            catalog_rows=catalog_rows,
            distance=distance,
            prefix_temperature=prefix_temperature,
            route_temperature=route_temperature,
            generator=generator,
        )
        return OccurrenceStepSample(
            gap_tokens=gap_tokens,
            prefix_codes=prefix,
            arrival_time_minutes=arrival_time_minutes,
        )

    @torch.no_grad()
    def sample_next_occurrence(
        self,
        batch: OccurrenceBatch,
        *,
        step: int,
        active: torch.Tensor,
        cache: OccurrenceGenerationCache,
        gap_temperature: float,
        prefix_temperature: float,
        route_temperature: float,
        generator: torch.Generator | None,
    ) -> tuple[OccurrenceStepSample, OccurrenceGenerationCache]:
        hidden, cache = self._encode_history_step(
            batch, step=step, active=active, cache=cache
        )
        sample = self._sample_from_hidden(
            hidden,
            previous_time_minutes=batch.previous_event_time_minutes[
                active, step
            ],
            memory_prefix_codes=batch.previous_prefix_codes[
                active, : step + 1
            ],
            memory_gap_tokens=batch.previous_gap_tokens[active, : step + 1],
            memory_event_time_minutes=batch.previous_event_time_minutes[
                active, : step + 1
            ],
            catalog_rows=self.catalog_rows(batch.city_ids[active]),
            gap_temperature=gap_temperature,
            prefix_temperature=prefix_temperature,
            route_temperature=route_temperature,
            generator=generator,
        )
        return sample, cache

    def forward(self, batch: OccurrenceBatch) -> OccurrencePlannerOutput:
        hidden = self._encode_history(batch)
        gap_logits = self._mask_gap_logits_by_horizon(
            self.gap_head(hidden),
            self._earliest_previous_event_time(batch.previous_gap_tokens),
        )
        arrival_time_minutes = self._reconstruct_arrival_time(
            batch.previous_event_time_minutes, batch.target_gap_tokens
        )
        conditioned = self.condition_on_gap_and_time(
            hidden, batch.target_gap_tokens, arrival_time_minutes
        )

        # Padded steps decode the first prefix legal in the row's city, so
        # every decoder level has a legal target.
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
        prefix_logits, prefix_targets = self.prefix_decoder(
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
                prefix_logits, prefix_targets
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
