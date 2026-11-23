from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_NEG_INF = float("-inf")
_N_RETRIEVAL_EXPERTS = 3


@dataclass(frozen=True, slots=True)
class ExploreRetrieveRoutingOutput:
    retrieve_logit: torch.Tensor
    retrieval_log_probability: torch.Tensor
    history_mask: torch.Tensor


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _masked_log_softmax(
    scores: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    any_valid = mask.any(dim=-1, keepdim=True)
    safe_mask = mask.clone()
    safe_mask[..., 0] |= ~any_valid.squeeze(-1)
    safe_scores = scores.masked_fill(~safe_mask, _NEG_INF)
    result = F.log_softmax(safe_scores.float(), dim=-1)
    result = result.masked_fill(~mask, _NEG_INF)
    return result.masked_fill(~any_valid, _NEG_INF)


def _causal_history_mask(prefix_codes: torch.Tensor) -> torch.Tensor:
    valid_memory = (prefix_codes >= 0).all(dim=-1)
    steps = prefix_codes.shape[1]
    indices = torch.arange(steps, device=prefix_codes.device)
    causal = indices[None, :, None] >= indices[None, None, :]
    return causal & valid_memory[:, None, :]


def _same_prefix(
    prefix_codes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = (prefix_codes >= 0).all(dim=-1)
    same = (prefix_codes[:, :, None, :] == prefix_codes[:, None, :, :]).all(
        dim=-1
    )
    return valid, same & valid[:, :, None] & valid[:, None, :]


def _prefix_statistics(
    prefix_codes: torch.Tensor,
    history_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid, same = _same_prefix(prefix_codes)

    cumulative = same.cumsum(dim=-1, dtype=torch.int32)
    frequency_count = cumulative.transpose(1, 2).to(torch.float32)
    count_at_occurrence = cumulative.diagonal(dim1=1, dim2=2)
    latest_mask = history_mask & (
        frequency_count == count_at_occurrence[:, None, :]
    )

    steps = prefix_codes.shape[1]
    indices = torch.arange(steps, device=prefix_codes.device)
    earlier = indices[None, None, :] < indices[None, :, None]
    seen_earlier = (same & earlier).any(dim=-1)
    first_occurrence = valid & ~seen_earlier
    distinct_count = first_occurrence.to(torch.long).cumsum(dim=-1)
    return frequency_count, latest_mask, distinct_count


def _step_prefix_statistics(
    prefix_codes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid, same = _same_prefix(prefix_codes)
    frequency_count = same.sum(dim=-1, dtype=torch.int32).to(torch.float32)

    memory_steps = prefix_codes.shape[1]
    indices = torch.arange(memory_steps, device=prefix_codes.device)
    later = indices[None, None, :] > indices[None, :, None]
    has_later_same = (same & later).any(dim=-1)
    latest_mask = valid & ~has_later_same
    distinct_count = latest_mask.sum(dim=-1).to(torch.long)
    return frequency_count, latest_mask, distinct_count


def _mix_expert_log_probabilities(
    expert_log_probabilities: Sequence[torch.Tensor],
    mechanism_log_weights: torch.Tensor,
    history_mask: torch.Tensor,
    history_available: torch.Tensor,
) -> torch.Tensor:
    mixed: torch.Tensor | None = None
    for index, expert in enumerate(expert_log_probabilities):
        term = expert.masked_fill(~history_available[..., None], 0.0)
        term = term + mechanism_log_weights[..., index, None]
        term = term.masked_fill(~history_mask, 0.0)
        mixed = term if mixed is None else torch.logaddexp(mixed, term)
    assert mixed is not None
    return mixed.masked_fill(~history_mask, _NEG_INF)


class _EprGatePrior(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.explore_rho_logit = nn.Parameter(torch.tensor(0.0))
        self.explore_gamma_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0))
        )

    @property
    def explore_rho(self) -> torch.Tensor:
        return torch.sigmoid(self.explore_rho_logit)

    @property
    def explore_gamma(self) -> torch.Tensor:
        return F.softplus(self.explore_gamma_raw)

    def forward(self, distinct_prefix_count: torch.Tensor) -> torch.Tensor:
        distinct = distinct_prefix_count.float().clamp_min(1.0)
        q_explore = self.explore_rho.float() * distinct.pow(
            -self.explore_gamma.float()
        )
        eps = torch.finfo(q_explore.dtype).eps
        q_explore = q_explore.clamp(min=eps, max=1.0 - eps)
        return torch.logit(1.0 - q_explore)


class _LogDistinctGatePrior(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.coefficients = nn.Parameter(torch.zeros(3))

    def forward(self, distinct_prefix_count: torch.Tensor) -> torch.Tensor:
        log_distinct = distinct_prefix_count.float().clamp_min(1.0).log()
        bias, linear, quadratic = self.coefficients.float().unbind()
        return bias + linear * log_distinct + quadratic * log_distinct.square()


class ExploreRetrieveRouter(nn.Module):
    def __init__(self, *, model_dim: int, gate_prior: str) -> None:
        super().__init__()
        self.model_dim = int(model_dim)

        self.query_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.key_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.mechanism_router = nn.Linear(model_dim, _N_RETRIEVAL_EXPERTS)

        self.router_residual = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        nn.init.zeros_(self.mechanism_router.weight)
        nn.init.zeros_(self.mechanism_router.bias)
        nn.init.zeros_(self.router_residual[-1].weight)
        nn.init.zeros_(self.router_residual[-1].bias)

        self.frequency_log_exponent = nn.Parameter(torch.zeros(()))

        self.gate_prior_module = (
            _EprGatePrior() if gate_prior == "epr" else _LogDistinctGatePrior()
        )

    @property
    def frequency_exponent(self) -> torch.Tensor:
        return self.frequency_log_exponent.exp()

    def _neural_expert(
        self,
        query_states: torch.Tensor,
        memory_states: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.query_projection(query_states)
        keys = self.key_projection(memory_states)
        scores = torch.matmul(queries, keys.transpose(-1, -2))
        scores = scores / math.sqrt(self.model_dim)
        return _masked_log_softmax(scores, history_mask)

    def _frequency_expert(
        self,
        frequency_count: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        beta = self.frequency_exponent.to(frequency_count.dtype)
        scores = beta * torch.log(frequency_count.clamp_min(1.0))
        return _masked_log_softmax(scores, history_mask)

    def _last_expert(self, history_mask: torch.Tensor) -> torch.Tensor:
        memory_steps = history_mask.shape[-1]
        indices = torch.arange(
            memory_steps, device=history_mask.device, dtype=torch.int32
        )
        candidate_indices = torch.where(
            history_mask, indices, torch.full_like(indices, -1)
        )
        last_index = candidate_indices.max(dim=-1).values
        last = history_mask & (indices == last_index[..., None])
        log_probability = torch.full(
            history_mask.shape,
            _NEG_INF,
            dtype=torch.float32,
            device=history_mask.device,
        )
        return log_probability.masked_fill(last, 0.0)

    def _router_logit(
        self,
        query_states: torch.Tensor,
        memory_states: torch.Tensor,
        retrieval_log_probability: torch.Tensor,
        distinct_prefix_count: torch.Tensor,
        history_available: torch.Tensor,
    ) -> torch.Tensor:
        retrieval_probability = retrieval_log_probability.exp().to(
            memory_states.dtype
        )
        if retrieval_probability.ndim == 3:
            memory_summary = torch.einsum(
                "btj,bjd->btd", retrieval_probability, memory_states
            )
        else:
            memory_summary = torch.einsum(
                "bj,bjd->bd", retrieval_probability, memory_states
            )
        residual_features = torch.cat(
            (
                query_states,
                memory_summary,
                query_states * memory_summary,
            ),
            dim=-1,
        )
        residual = self.router_residual(residual_features).squeeze(-1).float()
        residual = residual + self.gate_prior_module(distinct_prefix_count)
        return residual.masked_fill(~history_available, _NEG_INF)

    def _route(
        self,
        query_states: torch.Tensor,
        memory_states: torch.Tensor,
        expert_log_probabilities: tuple[torch.Tensor, ...],
        history_mask: torch.Tensor,
        distinct_prefix_count: torch.Tensor,
    ) -> ExploreRetrieveRoutingOutput:
        mechanism_log_weights = F.log_softmax(
            self.mechanism_router(query_states).float(), dim=-1
        )
        history_available = history_mask.any(dim=-1)
        retrieval_log_probability = _mix_expert_log_probabilities(
            expert_log_probabilities,
            mechanism_log_weights,
            history_mask,
            history_available,
        )
        retrieve_logit = self._router_logit(
            query_states,
            memory_states,
            retrieval_log_probability,
            distinct_prefix_count,
            history_available,
        )
        return ExploreRetrieveRoutingOutput(
            retrieve_logit=retrieve_logit,
            retrieval_log_probability=retrieval_log_probability,
            history_mask=history_mask,
        )

    def forward_step(
        self,
        query_state: torch.Tensor,
        memory_states: torch.Tensor,
        memory_prefix_codes: torch.Tensor,
    ) -> ExploreRetrieveRoutingOutput:

        frequency_count, history_mask, distinct_prefix_count = (
            _step_prefix_statistics(memory_prefix_codes)
        )
        neural_log_probability = self._neural_expert(
            query_state[:, None, :], memory_states, history_mask[:, None, :]
        ).squeeze(1)
        return self._route(
            query_state,
            memory_states,
            (
                neural_log_probability,
                self._frequency_expert(frequency_count, history_mask),
                self._last_expert(history_mask),
            ),
            history_mask,
            distinct_prefix_count,
        )

    def forward(
        self,
        query_states: torch.Tensor,
        memory_states: torch.Tensor,
        memory_prefix_codes: torch.Tensor,
    ) -> ExploreRetrieveRoutingOutput:

        causal_history_mask = _causal_history_mask(memory_prefix_codes)
        (
            frequency_count,
            history_mask,
            distinct_prefix_count,
        ) = _prefix_statistics(memory_prefix_codes, causal_history_mask)
        return self._route(
            query_states,
            memory_states,
            (
                self._neural_expert(query_states, memory_states, history_mask),
                self._frequency_expert(frequency_count, history_mask),
                self._last_expert(history_mask),
            ),
            history_mask,
            distinct_prefix_count,
        )


def retrieval_target_log_probability(
    routing: ExploreRetrieveRoutingOutput,
    memory_prefix_codes: torch.Tensor,
    target_prefix_codes: torch.Tensor,
) -> torch.Tensor:
    matches = (
        target_prefix_codes[:, :, None, :]
        == memory_prefix_codes[:, None, :, :]
    ).all(dim=-1)
    matches = matches & routing.history_mask
    has_match = matches.any(dim=-1)
    target_terms = routing.retrieval_log_probability.masked_fill(
        ~matches, _NEG_INF
    )
    safe_terms = torch.where(
        has_match[..., None],
        target_terms,
        torch.zeros_like(target_terms),
    )
    result = torch.logsumexp(safe_terms, dim=-1)
    return result.masked_fill(~has_match, _NEG_INF)


def explore_retrieve_target_log_probability(
    *,
    explore_target_log_probability: torch.Tensor,
    routing: ExploreRetrieveRoutingOutput,
    retrieval_target_log_probability: torch.Tensor,
) -> torch.Tensor:
    log_retrieve = F.logsigmoid(routing.retrieve_logit)
    log_explore = F.logsigmoid(-routing.retrieve_logit)
    no_history = ~routing.history_mask.any(dim=-1)
    log_explore = log_explore.masked_fill(no_history, 0.0)
    log_retrieve = log_retrieve.masked_fill(no_history, _NEG_INF)
    explore_term = log_explore + explore_target_log_probability.float()
    retrieve_term = log_retrieve + retrieval_target_log_probability.float()
    retrieval_available = torch.isfinite(retrieval_target_log_probability)
    safe_retrieve_term = torch.where(
        retrieval_available,
        retrieve_term,
        torch.zeros_like(retrieve_term),
    )
    mixed = torch.logaddexp(explore_term, safe_retrieve_term)
    return torch.where(retrieval_available, mixed, explore_term)


def prefix_target_log_probability(
    prefix_logits: torch.Tensor,
    prefix_targets: torch.Tensor,
) -> torch.Tensor:
    selected = (
        F.log_softmax(prefix_logits.float(), dim=-1)
        .gather(-1, prefix_targets[..., None])
        .squeeze(-1)
    )
    return selected.sum(dim=-1)


__all__ = [
    "ExploreRetrieveRouter",
    "explore_retrieve_target_log_probability",
    "prefix_target_log_probability",
    "retrieval_target_log_probability",
]
