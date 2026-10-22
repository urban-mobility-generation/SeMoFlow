from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from semoflow.planner.config import OccurrencePlannerConfig
from semoflow.planner.decoders import (
    FlatLegalPrefixDecoder,
    HierarchicalPrefixDecoder,
    PrefixDecoder,
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

_CHECKPOINT_SCHEMA_VERSION = 8


def _tensor_fingerprint(tensor: torch.Tensor | None) -> str | None:
    if tensor is None:
        return None
    value = tensor.detach().to("cpu").contiguous()
    header = f"prefix-metadata-v1:{tuple(value.shape)}:{value.dtype}:".encode()
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def _city_legality_fingerprint(
    city_ids: torch.Tensor,
    city_mask: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"city-legality-v1:")
    digest.update(
        city_ids.detach().to("cpu", torch.long).contiguous().numpy().tobytes()
    )
    digest.update(
        city_mask.detach()
        .to("cpu", torch.uint8)
        .contiguous()
        .numpy()
        .tobytes()
    )
    return digest.hexdigest()


class OccurrenceARPlanner(nn.Module):
    def __init__(
        self,
        config: OccurrencePlannerConfig,
        legal_prefixes: torch.Tensor,
        *,
        catalog_city_ids: torch.Tensor | None = None,
        city_legal_prefix_mask: torch.Tensor | None = None,
        prefix_codebooks: torch.Tensor | None = None,
        legal_prefix_centroids: torch.Tensor | None = None,
        legal_prefix_centroid_available: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if (catalog_city_ids is None) != (city_legal_prefix_mask is None):
            raise ValueError(
                "catalog_city_ids and city_legal_prefix_mask must be set together"
            )
        if catalog_city_ids is None:
            catalog_city_ids = torch.tensor([0], dtype=torch.long)
            city_legal_prefix_mask = torch.ones(
                (1, legal_prefixes.shape[0]), dtype=torch.bool
            )
        catalog_city_ids = torch.as_tensor(catalog_city_ids, dtype=torch.long)
        city_legal_prefix_mask = torch.as_tensor(
            city_legal_prefix_mask, dtype=torch.bool
        )
        if catalog_city_ids.ndim != 1 or catalog_city_ids.numel() < 1:
            raise ValueError("catalog_city_ids must be a non-empty vector")
        if not torch.equal(
            catalog_city_ids, torch.sort(catalog_city_ids).values
        ):
            raise ValueError("catalog_city_ids must be sorted")
        if torch.unique(catalog_city_ids).numel() != catalog_city_ids.numel():
            raise ValueError("catalog_city_ids must be unique")
        if city_legal_prefix_mask.shape != (
            catalog_city_ids.numel(),
            legal_prefixes.shape[0],
        ):
            raise ValueError(
                "city_legal_prefix_mask must have shape [cities, vocabulary]"
            )
        if not bool(city_legal_prefix_mask.any(dim=-1).all()):
            raise ValueError("each city must permit at least one prefix")
        if catalog_city_ids.numel() == 1 and not bool(
            city_legal_prefix_mask.all()
        ):
            raise ValueError(
                "a single-city catalog must contain the global vocabulary"
            )
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

        if config.use_tokenizer_latent:
            if prefix_codebooks is None:
                raise ValueError(
                    "use_tokenizer_latent requires prefix_codebooks"
                )
            codebooks = torch.as_tensor(
                prefix_codebooks, dtype=torch.float32
            ).detach()
            if codebooks.ndim != 3 or codebooks.shape[:2] != (
                config.prefix_depth,
                config.codebook_size,
            ):
                raise ValueError(
                    "prefix_codebooks must have shape [prefix_depth, K, D]"
                )
            if not bool(torch.isfinite(codebooks).all()):
                raise ValueError("prefix_codebooks must be finite")
            self.register_buffer(
                "prefix_codebooks", codebooks.clone(), persistent=False
            )
            self.tokenizer_latent_projection = nn.Linear(
                int(codebooks.shape[-1]), config.model_dim, bias=False
            )
            nn.init.zeros_(self.tokenizer_latent_projection.weight)
        else:
            if prefix_codebooks is not None:
                raise ValueError(
                    "prefix_codebooks supplied without use_tokenizer_latent"
                )
            self.register_buffer("prefix_codebooks", None, persistent=False)
            self.tokenizer_latent_projection = None

        self.distance_prior: PrefixDistancePrior | None = None
        if config.use_distance_prior:
            if (
                legal_prefix_centroids is None
                or legal_prefix_centroid_available is None
            ):
                raise ValueError(
                    "use_distance_prior requires legal-prefix centroids"
                )
            self.distance_prior = PrefixDistancePrior(
                legality=self.legality,
                centroids=legal_prefix_centroids,
                centroid_available=legal_prefix_centroid_available,
            )
        elif (
            legal_prefix_centroids is not None
            or legal_prefix_centroid_available is not None
        ):
            raise ValueError(
                "prefix centroids supplied without use_distance_prior"
            )

        self._codebooks_fingerprint = _tensor_fingerprint(
            self.prefix_codebooks
        )
        self._geography_fingerprint = (
            None
            if self.distance_prior is None
            else _tensor_fingerprint(
                torch.cat(
                    (
                        self.distance_prior.leaf_units.flatten(),
                        self.distance_prior.leaf_available.float(),
                    )
                )
            )
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
        self.context_projection = (
            nn.Linear(config.context_dim, config.model_dim, bias=False)
            if config.context_dim
            else None
        )
        self.time_projection = (
            nn.Sequential(
                nn.Linear(TIME_FEATURE_DIM, config.model_dim),
                nn.GELU(),
                nn.Linear(config.model_dim, config.model_dim),
            )
            if config.temporal_state_enabled
            else None
        )
        if config.temporal_state_enabled:
            assert config.gap_duration_minutes is not None
            assert config.gap_bucket_edges_minutes is not None
            assert config.observation_window_minutes is not None
            edges = torch.tensor(
                config.gap_bucket_edges_minutes, dtype=torch.float32
            )
            lower = torch.cat((torch.zeros(1), edges))
            upper = torch.cat(
                (
                    edges,
                    torch.tensor(
                        [config.observation_window_minutes],
                        dtype=torch.float32,
                    ),
                )
            )
            self.register_buffer(
                "gap_duration_lower_minutes", lower, persistent=False
            )
            self.register_buffer(
                "gap_duration_upper_minutes", upper, persistent=False
            )
            self.register_buffer(
                "gap_representative_minutes",
                torch.tensor(config.gap_duration_minutes, dtype=torch.float32),
                persistent=False,
            )
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
        self.prefix_decoder: PrefixDecoder
        if config.prefix_decoder == "hierarchical":
            self.prefix_decoder = HierarchicalPrefixDecoder(
                model_dim=config.model_dim,
                codebook_size=config.codebook_size,
                prefix_depth=config.prefix_depth,
                legality=self.legality,
            )
        else:
            self.prefix_decoder = FlatLegalPrefixDecoder(
                model_dim=config.model_dim,
                legality=self.legality,
            )
        self.mobility_router = (
            ExploreRetrieveRouter(
                model_dim=config.model_dim,
                gate_prior=config.mobility_gate_prior,
            )
            if config.mobility_router == "explore_retrieve"
            else None
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "prefix_codebooks_fingerprint": self._codebooks_fingerprint,
            "prefix_geography_fingerprint": self._geography_fingerprint,
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "config": asdict(self.config),
            "legal_prefix_count": self.legality.vocabulary_size,
            "catalog_fingerprint": self.legality.catalog_fingerprint,
            "catalog_order_fingerprint": (
                self.legality.catalog_order_fingerprint
            ),
            "catalog_city_ids": self.catalog_city_ids.detach().cpu().tolist(),
            "city_legality_fingerprint": _city_legality_fingerprint(
                self.catalog_city_ids,
                self.city_legal_prefix_mask,
            ),
        }

    def set_extra_state(self, state: object) -> None:
        pass

    def _validate_checkpoint_metadata(
        self,
        state: object,
        *,
        allow_catalog_mismatch: bool,
    ) -> None:
        if not isinstance(state, dict):
            raise RuntimeError("planner checkpoint metadata is missing")
        metadata = cast("dict[str, Any]", state)
        required = {
            "schema_version",
            "config",
            "legal_prefix_count",
            "catalog_fingerprint",
            "catalog_order_fingerprint",
            "catalog_city_ids",
            "city_legality_fingerprint",
            "prefix_codebooks_fingerprint",
            "prefix_geography_fingerprint",
        }
        if set(metadata) != required:
            raise RuntimeError(
                "planner checkpoint metadata has invalid fields"
            )
        if metadata["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(
                "planner checkpoint schema version is unsupported"
            )
        if metadata["config"] != asdict(self.config):
            raise RuntimeError(
                "planner checkpoint architecture is incompatible"
            )
        catalog_mismatch = (
            metadata["legal_prefix_count"] != self.legality.vocabulary_size
            or metadata["catalog_fingerprint"]
            != self.legality.catalog_fingerprint
            or metadata["catalog_city_ids"]
            != self.catalog_city_ids.detach().cpu().tolist()
            or metadata["city_legality_fingerprint"]
            != _city_legality_fingerprint(
                self.catalog_city_ids,
                self.city_legal_prefix_mask,
            )
            or metadata["prefix_codebooks_fingerprint"]
            != self._codebooks_fingerprint
            or metadata["prefix_geography_fingerprint"]
            != self._geography_fingerprint
        )
        order_mismatch = (
            metadata["catalog_order_fingerprint"]
            != self.legality.catalog_order_fingerprint
        )
        if not catalog_mismatch and (
            self.config.prefix_decoder != "flat" or not order_mismatch
        ):
            return
        if allow_catalog_mismatch:
            if self.config.prefix_decoder == "flat":
                raise RuntimeError(
                    "catalog mismatch is unsafe for the flat prefix decoder"
                )
            return
        raise RuntimeError(
            "planner checkpoint catalog fingerprint is incompatible"
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        self._validate_checkpoint_metadata(
            state_dict.get("_extra_state"), allow_catalog_mismatch=False
        )
        return super().load_state_dict(
            state_dict, strict=strict, assign=assign
        )

    def load_neural_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        allow_catalog_mismatch: bool = False,
        strict: bool = True,
        assign: bool = False,
    ):
        self._validate_checkpoint_metadata(
            state_dict.get("_extra_state"),
            allow_catalog_mismatch=allow_catalog_mismatch,
        )
        return nn.Module.load_state_dict(
            self, state_dict, strict=strict, assign=assign
        )

    def _catalog_rows_for_city_ids(
        self,
        city_ids: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        if city_ids is None:
            if self.catalog_city_ids.numel() != 1:
                raise ValueError(
                    "city_ids are required for a multi-city planner"
                )
            return torch.zeros(
                batch_size,
                dtype=torch.long,
                device=self.catalog_city_ids.device,
            )
        if city_ids.shape != (batch_size,) or city_ids.dtype != torch.long:
            raise ValueError(
                "city_ids must be a long tensor with shape [batch]"
            )
        if city_ids.device != self.catalog_city_ids.device:
            raise ValueError("city_ids and planner must share one device")
        city_ids = city_ids.contiguous()
        if self.catalog_city_ids.numel() == 1:
            if not torch.compiler.is_compiling() and not bool(
                (city_ids == self.catalog_city_ids[0]).all()
            ):
                unknown = torch.unique(city_ids).detach().cpu().tolist()
                raise ValueError(f"unknown planner city_ids: {unknown}")
            return torch.zeros_like(city_ids)
        positions = torch.searchsorted(self.catalog_city_ids, city_ids)
        safe = positions.clamp(max=self.catalog_city_ids.numel() - 1)
        found = self.catalog_city_ids[safe] == city_ids
        if not torch.compiler.is_compiling() and not bool(found.all()):
            unknown = torch.unique(city_ids[~found]).detach().cpu().tolist()
            raise ValueError(f"unknown planner city_ids: {unknown}")
        return safe

    def _validate_batch(
        self, batch: OccurrenceBatch, *, validate_targets: bool = True
    ) -> None:
        batch.validate(self.config, validate_temporal_targets=validate_targets)
        if batch.lengths.device != self.length_query.device:
            raise ValueError(
                "occurrence batch and planner must share one device"
            )
        catalog_rows = self._catalog_rows_for_city_ids(
            batch.city_ids,
            batch_size=batch.lengths.shape[0],
        )
        if validate_targets:
            valid_targets = batch.target_prefix_codes[batch.valid_mask]
            if not bool(self.legality.contains(valid_targets).all()):
                raise ValueError("prefix targets contain an illegal tuple")
            target_token_ids = self.legality.token_ids(valid_targets)
            target_batch_rows = batch.valid_mask.nonzero(as_tuple=True)[0]
            if not bool(
                self.city_legal_prefix_mask[
                    catalog_rows[target_batch_rows], target_token_ids
                ].all()
            ):
                raise ValueError("prefix target is illegal for its city")
        history_mask = batch.valid_mask & (
            batch.previous_prefix_codes >= 0
        ).all(dim=-1)
        valid_history = batch.previous_prefix_codes[history_mask]
        if valid_history.numel():
            if not bool(self.legality.contains(valid_history).all()):
                raise ValueError("prefix history contains an illegal tuple")
            history_token_ids = self.legality.token_ids(valid_history)
            history_batch_rows = history_mask.nonzero(as_tuple=True)[0]
            if not bool(
                self.city_legal_prefix_mask[
                    catalog_rows[history_batch_rows], history_token_ids
                ].all()
            ):
                raise ValueError("prefix history is illegal for its city")

    def _context_state(
        self,
        context_features: torch.Tensor | None,
        city_ids: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        dtype = self.length_query.dtype
        if self.context_projection is None:
            if context_features is not None:
                raise ValueError("context supplied to a context-free planner")
            context = torch.zeros(
                (batch_size, self.config.model_dim),
                device=device,
                dtype=dtype,
            )
        else:
            if context_features is None:
                raise ValueError("context_features are required")
            if context_features.shape != (batch_size, self.config.context_dim):
                raise ValueError("context_features have the wrong shape")
            context = self.context_projection(
                context_features.to(device=device, dtype=dtype)
            )
        if self.city_embedding is not None:
            catalog_rows = self._catalog_rows_for_city_ids(
                city_ids, batch_size=batch_size
            )
            context = context + self.city_embedding(catalog_rows)
        return context

    def _time_state_embedding(
        self, event_time_minutes: torch.Tensor
    ) -> torch.Tensor:
        if self.time_projection is None:
            raise RuntimeError("this planner has no time channel")
        dtype = self.length_query.dtype
        window = self.config.observation_window_minutes
        assert window is not None
        features = time_features(
            event_time_minutes, observation_window_minutes=window
        ).to(dtype=dtype)
        return self.time_projection(features)

    def _mask_gap_logits_by_horizon(
        self,
        gap_logits: torch.Tensor,
        earliest_previous_time_minutes: torch.Tensor,
    ) -> torch.Tensor:
        window = self.config.observation_window_minutes
        assert window is not None
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
        if previous_time_minutes.shape != gap_tokens.shape:
            raise ValueError("gap tokens must align with previous event time")
        window = self.config.observation_window_minutes
        if window is None:
            raise RuntimeError("this planner has no temporal state")
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

    def predict_length(
        self,
        *,
        batch_size: int,
        context_features: torch.Tensor | None = None,
        city_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = self.length_query.device
        context = self._context_state(
            context_features,
            city_ids,
            batch_size=batch_size,
            device=device,
        )
        return self.length_head(self.length_query[None, :] + context)

    def _embed_prefix_history(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        if prefix_ids.ndim != 3:
            raise ValueError("prefix ids must have shape [B, S, P]")
        if prefix_ids.shape[-1] != self.config.prefix_depth:
            raise ValueError("prefix depth differs from config")
        level_states = [
            embedding(prefix_ids[..., level])
            for level, embedding in enumerate(self.prefix_embeddings)
        ]
        hidden = torch.stack(level_states, dim=0).sum(dim=0)
        if self.tokenizer_latent_projection is None:
            return hidden
        assert self.prefix_codebooks is not None
        known = (prefix_ids < self.config.codebook_size).all(dim=-1)
        safe = prefix_ids.clamp(max=self.config.codebook_size - 1)
        latent = torch.stack(
            [
                self.prefix_codebooks[level][safe[..., level]]
                for level in range(self.config.prefix_depth)
            ],
            dim=0,
        ).sum(dim=0)
        latent = latent * known[..., None].to(latent.dtype)
        projected = self.tokenizer_latent_projection(
            latent.to(dtype=hidden.dtype)
        )
        return hidden + projected

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
        previous_event_time_minutes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if previous_prefix_codes.ndim != 3:
            raise ValueError(
                "retrieval prefix memory must have shape [B, M, P]"
            )
        batch_size, memory_steps, depth = previous_prefix_codes.shape
        if depth != self.config.prefix_depth:
            raise ValueError(
                "retrieval memory prefix depth differs from config"
            )
        if previous_gap_tokens.shape != (batch_size, memory_steps):
            raise ValueError("retrieval gaps do not align with prefixes")
        if self.time_projection is None:
            if previous_event_time_minutes is not None:
                raise ValueError(
                    "retrieval event times supplied without a time channel"
                )
        else:
            if previous_event_time_minutes is None:
                raise ValueError(
                    "retrieval event times are required by the time channel"
                )
            if previous_event_time_minutes.shape != (
                batch_size,
                memory_steps,
            ):
                raise ValueError(
                    "retrieval event times do not align with prefixes"
                )

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
        if self.time_projection is not None:
            assert previous_event_time_minutes is not None
            memory = memory + self._time_state_embedding(
                previous_event_time_minutes
            )
        return memory

    def initialize_generation_cache(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> OccurrenceGenerationCache:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
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
        context = self._context_state(
            None
            if batch.context_features is None
            else batch.context_features[active],
            None if batch.city_ids is None else batch.city_ids[active],
            batch_size=int(active.sum()),
            device=hidden.device,
        )
        hidden = hidden + context[:, None, :]
        if self.time_projection is not None:
            assert batch.previous_event_time_minutes is not None
            previous_time = batch.previous_event_time_minutes[
                active, step : step + 1
            ]
            hidden = hidden + self._time_state_embedding(previous_time)
        return hidden

    @torch.no_grad()
    def _encode_history_step(
        self,
        batch: OccurrenceBatch,
        *,
        step: int,
        active: torch.Tensor,
        cache: OccurrenceGenerationCache,
    ) -> tuple[torch.Tensor, OccurrenceGenerationCache]:
        if self.training:
            raise RuntimeError(
                "incremental generation cache requires eval mode"
            )
        batch_size = batch.previous_prefix_codes.shape[0]
        if not 0 <= step < batch.previous_prefix_codes.shape[1]:
            raise ValueError("step is outside the batch sequence")
        if active.dtype != torch.bool or active.shape != (batch_size,):
            raise ValueError(
                "active must be a boolean tensor with shape [batch]"
            )
        if not torch.equal(active, batch.lengths > step):
            raise ValueError("active rows must match the declared lengths")
        if len(cache.keys) != self.config.n_layers or len(cache.values) != len(
            cache.keys
        ):
            raise ValueError("generation cache has the wrong layer count")
        if cache.step != step:
            raise ValueError("generation cache step is out of sequence")
        hidden = self._embed_history_step(batch, step=step, active=active)
        new_keys: list[torch.Tensor] = []
        new_values: list[torch.Tensor] = []
        head_dim = self.config.model_dim // self.config.n_heads
        expected_cache_shape = (
            batch_size,
            self.config.n_heads,
            step,
            head_dim,
        )
        relation_bias = self.relation_bias.step(
            batch.previous_prefix_codes[active],
            (
                None
                if batch.previous_event_time_minutes is None
                else batch.previous_event_time_minutes[active]
            ),
            step=step,
        )
        for layer_index, layer in enumerate(self.transformer.layers):
            past_keys = cache.keys[layer_index]
            past_values = cache.values[layer_index]
            if (
                past_keys.shape != expected_cache_shape
                or past_values.shape != expected_cache_shape
                or past_keys.device != hidden.device
                or past_values.device != hidden.device
                or past_keys.dtype != hidden.dtype
                or past_values.dtype != hidden.dtype
            ):
                raise ValueError(
                    "generation cache shape, device, or dtype differs"
                )
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
        if previous_prefixes.ndim != 3:
            raise ValueError("previous prefixes must have shape [B, S, P]")
        batch_size, steps, depth = previous_prefixes.shape
        if depth != self.config.prefix_depth:
            raise ValueError("history prefix depth differs from config")
        if steps > self.config.max_occurrences:
            raise ValueError("history exceeds max_occurrences")
        if batch.previous_gap_tokens.shape != (batch_size, steps):
            raise ValueError("previous gaps do not align with prefixes")
        if batch.valid_mask.shape != (batch_size, steps):
            raise ValueError("valid mask does not align with history")
        if batch.lengths.shape != (batch_size,):
            raise ValueError("lengths must have shape [batch]")
        if not torch.compiler.is_compiling() and (
            bool((batch.lengths < 1).any())
            or bool((batch.lengths > self.config.max_occurrences).any())
        ):
            raise ValueError("length is outside the configured vocabulary")

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
        context = self._context_state(
            batch.context_features,
            batch.city_ids,
            batch_size=batch_size,
            device=hidden.device,
        )
        hidden = hidden + context[:, None, :]
        if self.time_projection is not None:
            assert batch.previous_event_time_minutes is not None
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
        if gap_tokens.shape != hidden.shape[:-1]:
            raise ValueError("gap tokens must align with hidden states")
        if not torch.compiler.is_compiling() and (
            bool((gap_tokens < 0).any())
            or bool((gap_tokens >= self.config.gap_vocab).any())
        ):
            raise ValueError("gap token is outside the vocabulary")
        return hidden + self.gap_condition(self.gap_embedding(gap_tokens))

    def _step_explore_log_probability(
        self,
        conditioned: torch.Tensor,
        target_prefix: torch.Tensor,
        *,
        temperature: float,
        catalog_rows: torch.Tensor | None = None,
        distance: PrefixDistanceContext | None = None,
    ) -> torch.Tensor:
        logits, targets = self.prefix_decoder(
            conditioned,
            target_prefix,
            catalog_rows=catalog_rows,
            distance=distance,
        )
        scaled_logits = logits.float() / temperature
        if self.config.prefix_decoder == "hierarchical":
            return prefix_target_log_probability(
                scaled_logits[:, None, ...],
                targets[:, None, ...],
            ).squeeze(1)
        return prefix_target_log_probability(
            scaled_logits[:, None, ...],
            targets[:, None],
        ).squeeze(1)

    @staticmethod
    def _step_retrieval_target_log_probability(
        log_probability: torch.Tensor,
        history_mask: torch.Tensor,
        memory_prefix_codes: torch.Tensor,
        target_prefix: torch.Tensor,
    ) -> torch.Tensor:
        matches = (memory_prefix_codes == target_prefix[:, None, :]).all(
            dim=-1
        )
        matches = matches & history_mask
        return torch.logsumexp(
            log_probability.masked_fill(~matches, float("-inf")),
            dim=-1,
        )

    def _sample_from_hidden(
        self,
        hidden: torch.Tensor,
        *,
        previous_time_minutes: torch.Tensor | None = None,
        memory_prefix_codes: torch.Tensor | None = None,
        memory_gap_tokens: torch.Tensor | None = None,
        memory_event_time_minutes: torch.Tensor | None = None,
        catalog_rows: torch.Tensor | None = None,
        gap_temperature: float,
        prefix_temperature: float,
        route_temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> OccurrenceStepSample:
        if route_temperature <= 0.0:
            raise ValueError("route_temperature must be positive")
        distance = (
            None
            if memory_prefix_codes is None
            else self._distance_context(memory_prefix_codes[:, -1])
        )
        gap_logits = self.gap_head(hidden)
        if self.time_projection is not None:
            assert previous_time_minutes is not None
            assert memory_gap_tokens is not None
            earliest_previous_time = self._earliest_previous_event_time(
                memory_gap_tokens
            )[:, -1]
            gap_logits = self._mask_gap_logits_by_horizon(
                gap_logits,
                earliest_previous_time,
            )
        gap_sample = sample_categorical_logits(
            gap_logits,
            temperature=gap_temperature,
            generator=generator,
        )
        arrival_time_minutes: torch.Tensor | None = None
        if self.time_projection is not None:
            assert previous_time_minutes is not None
            arrival_time_minutes = self._reconstruct_arrival_time(
                previous_time_minutes,
                gap_sample.choices,
            )
            conditioned = self.condition_on_gap_and_time(
                hidden,
                gap_sample.choices,
                arrival_time_minutes,
            )
        else:
            conditioned = self.condition_on_gap(hidden, gap_sample.choices)

        if self.mobility_router is None:
            (
                prefix,
                prefix_model_log_probability,
                prefix_sampling_log_probability,
            ) = self.prefix_decoder.sample(
                conditioned,
                temperature=prefix_temperature,
                generator=generator,
                catalog_rows=catalog_rows,
                distance=distance,
            )
        else:
            if memory_prefix_codes is None or memory_gap_tokens is None:
                raise ValueError(
                    "mobility routing requires visible prefix and gap memory"
                )
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

            (
                explore_prefix,
                _,
                _,
            ) = self.prefix_decoder.sample(
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
            retrieval_sample = sample_categorical_logits(
                safe_retrieval_logits,
                temperature=prefix_temperature,
                generator=generator,
            )
            row = torch.arange(
                hidden.shape[0], device=memory_prefix_codes.device
            )
            retrieved_prefix = memory_prefix_codes[
                row, retrieval_sample.choices
            ]

            route_probability_policy = torch.sigmoid(
                routing.retrieve_logit / route_temperature
            ).masked_fill(~history_available, 0.0)
            route_draw = torch.rand(
                route_probability_policy.shape,
                device=hidden.device,
                generator=generator,
            )
            use_retrieve = (
                route_draw < route_probability_policy
            ) & history_available
            prefix = torch.where(
                use_retrieve[:, None],
                retrieved_prefix,
                explore_prefix,
            )

            explore_model_log_probability = self._step_explore_log_probability(
                conditioned,
                prefix,
                temperature=1.0,
                catalog_rows=catalog_rows,
                distance=distance,
            )
            explore_policy_log_probability = (
                self._step_explore_log_probability(
                    conditioned,
                    prefix,
                    temperature=prefix_temperature,
                    catalog_rows=catalog_rows,
                    distance=distance,
                )
            )
            retrieval_model_log_probability = (
                self._step_retrieval_target_log_probability(
                    routing.retrieval_log_probability,
                    routing.history_mask,
                    memory_prefix_codes,
                    prefix,
                )
            )
            retrieval_policy_distribution = F.log_softmax(
                safe_retrieval_logits / prefix_temperature,
                dim=-1,
            )
            retrieval_policy_log_probability = (
                self._step_retrieval_target_log_probability(
                    retrieval_policy_distribution,
                    routing.history_mask,
                    memory_prefix_codes,
                    prefix,
                )
            )
            prefix_model_log_probability = (
                explore_retrieve_target_log_probability(
                    explore_target_log_probability=(
                        explore_model_log_probability
                    ),
                    routing=routing,
                    retrieval_target_log_probability=(
                        retrieval_model_log_probability
                    ),
                )
            )
            policy_routing = routing.__class__(
                retrieve_logit=routing.retrieve_logit / route_temperature,
                retrieve_probability=route_probability_policy,
                mechanism_weights=routing.mechanism_weights,
                retrieval_log_probability=routing.retrieval_log_probability,
                history_mask=routing.history_mask,
                distinct_prefix_count=routing.distinct_prefix_count,
            )
            prefix_sampling_log_probability = (
                explore_retrieve_target_log_probability(
                    explore_target_log_probability=(
                        explore_policy_log_probability
                    ),
                    routing=policy_routing,
                    retrieval_target_log_probability=(
                        retrieval_policy_log_probability
                    ),
                )
            )

        return OccurrenceStepSample(
            gap_tokens=gap_sample.choices,
            prefix_codes=prefix,
            model_gap_log_probability=gap_sample.model_log_probability,
            model_prefix_log_probability=prefix_model_log_probability,
            sampling_gap_log_probability=gap_sample.sampling_log_probability,
            sampling_prefix_log_probability=prefix_sampling_log_probability,
            model_duration_log_probability=None,
            sampling_duration_log_probability=None,
            arrival_time_minutes=arrival_time_minutes,
        )

    @torch.no_grad()
    def _sample_next_occurrence_cached(
        self,
        batch: OccurrenceBatch,
        *,
        step: int,
        active: torch.Tensor,
        cache: OccurrenceGenerationCache,
        gap_temperature: float,
        prefix_temperature: float,
        route_temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[OccurrenceStepSample, OccurrenceGenerationCache]:
        hidden, cache = self._encode_history_step(
            batch, step=step, active=active, cache=cache
        )
        previous_time = (
            None
            if batch.previous_event_time_minutes is None
            else batch.previous_event_time_minutes[active, step]
        )
        catalog_rows = self._catalog_rows_for_city_ids(
            None if batch.city_ids is None else batch.city_ids[active],
            batch_size=hidden.shape[0],
        )
        decoder_catalog_rows = (
            catalog_rows if self.catalog_city_ids.numel() > 1 else None
        )
        return (
            self._sample_from_hidden(
                hidden,
                previous_time_minutes=previous_time,
                memory_prefix_codes=batch.previous_prefix_codes[
                    active, : step + 1
                ],
                memory_gap_tokens=batch.previous_gap_tokens[
                    active, : step + 1
                ],
                memory_event_time_minutes=(
                    None
                    if batch.previous_event_time_minutes is None
                    else batch.previous_event_time_minutes[active, : step + 1]
                ),
                catalog_rows=decoder_catalog_rows,
                gap_temperature=gap_temperature,
                prefix_temperature=prefix_temperature,
                route_temperature=route_temperature,
                generator=generator,
            ),
            cache,
        )

    def forward(self, batch: OccurrenceBatch) -> OccurrencePlannerOutput:
        if not torch.compiler.is_compiling():
            self._validate_batch(batch)
        hidden = self._encode_history(batch)
        gap_logits = self.gap_head(hidden)
        arrival_time_minutes: torch.Tensor | None = None
        if self.time_projection is not None:
            assert batch.previous_event_time_minutes is not None
            earliest_previous_time = self._earliest_previous_event_time(
                batch.previous_gap_tokens
            )
            gap_logits = self._mask_gap_logits_by_horizon(
                gap_logits,
                earliest_previous_time,
            )
            arrival_time_minutes = self._reconstruct_arrival_time(
                batch.previous_event_time_minutes,
                batch.target_gap_tokens,
            )
            conditioned = self.condition_on_gap_and_time(
                hidden,
                batch.target_gap_tokens,
                arrival_time_minutes,
            )
        else:
            conditioned = self.condition_on_gap(
                hidden, batch.target_gap_tokens
            )

        catalog_rows = self._catalog_rows_for_city_ids(
            batch.city_ids,
            batch_size=conditioned.shape[0],
        )
        use_city_masks = self.catalog_city_ids.numel() > 1
        row_catalog_mask = self.city_legal_prefix_mask[catalog_rows]
        safe_token_ids = row_catalog_mask.to(torch.int64).argmax(dim=-1)
        safe_prefixes = self.legality.codes[safe_token_ids]
        decoder_target_prefix_codes = torch.where(
            batch.valid_mask[..., None],
            batch.target_prefix_codes,
            safe_prefixes[:, None, :],
        )
        prefix_logits, prefix_targets = self.prefix_decoder(
            conditioned,
            decoder_target_prefix_codes,
            catalog_rows=(catalog_rows if use_city_masks else None),
            distance=self._distance_context(batch.previous_prefix_codes),
        )
        prefix_log_probability: torch.Tensor | None = None
        retrieve_probability: torch.Tensor | None = None
        retrieve_responsibility: torch.Tensor | None = None
        retrieval_mechanism_weights: torch.Tensor | None = None
        if self.mobility_router is not None:
            memory_states = self._embed_retrieval_memory(
                batch.previous_prefix_codes,
                batch.previous_gap_tokens,
                batch.previous_event_time_minutes,
            )
            routing = self.mobility_router(
                conditioned,
                memory_states,
                batch.previous_prefix_codes,
            )
            explore_log_probability = prefix_target_log_probability(
                prefix_logits, prefix_targets
            )
            retrieve_log_probability = retrieval_target_log_probability(
                routing,
                batch.previous_prefix_codes,
                batch.target_prefix_codes,
            )
            prefix_log_probability = explore_retrieve_target_log_probability(
                explore_target_log_probability=explore_log_probability,
                routing=routing,
                retrieval_target_log_probability=retrieve_log_probability,
            )
            retrieve_log_joint = (
                F.logsigmoid(routing.retrieve_logit) + retrieve_log_probability
            )
            retrieve_responsibility = torch.exp(
                retrieve_log_joint - prefix_log_probability
            ).masked_fill(~batch.valid_mask, 0.0)
            retrieve_responsibility = torch.nan_to_num(
                retrieve_responsibility,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            retrieve_probability = routing.retrieve_probability
            retrieval_mechanism_weights = routing.mechanism_weights

        length_logits = self.predict_length(
            batch_size=hidden.shape[0],
            context_features=batch.context_features,
            city_ids=batch.city_ids,
        )
        return OccurrencePlannerOutput(
            hidden=hidden,
            length_logits=length_logits,
            gap_logits=gap_logits,
            prefix_logits=prefix_logits,
            prefix_targets=prefix_targets,
            duration_log_probability=None,
            prefix_log_probability=prefix_log_probability,
            retrieve_probability=retrieve_probability,
            retrieve_responsibility=retrieve_responsibility,
            retrieval_mechanism_weights=retrieval_mechanism_weights,
        )
