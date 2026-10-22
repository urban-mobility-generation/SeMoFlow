from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from semoflow.planner.geography import PrefixDistancePrior
from semoflow.planner.legality import LegalPrefixIndex

_ILLEGAL_LOGIT = float("-inf")


@dataclass(frozen=True, slots=True)
class PrefixDistanceContext:
    prior: PrefixDistancePrior
    source_units: torch.Tensor
    source_valid: torch.Tensor

    def level_bias(self, level: int, parents: torch.Tensor) -> torch.Tensor:
        return self.prior.level_bias(
            level, parents, self.source_units, self.source_valid
        )

    def flat_bias(self) -> torch.Tensor:
        return self.prior.flat_bias(self.source_units, self.source_valid)

    def flatten(self) -> PrefixDistanceContext:
        return PrefixDistanceContext(
            prior=self.prior,
            source_units=self.source_units.reshape(-1, 3),
            source_valid=self.source_valid.reshape(-1),
        )


@dataclass(frozen=True, slots=True)
class CategoricalSample:
    choices: torch.Tensor
    model_log_probability: torch.Tensor
    sampling_log_probability: torch.Tensor


def sample_categorical_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    generator: torch.Generator | None,
) -> CategoricalSample:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    score_logits = logits.float()
    model_log_probabilities = torch.log_softmax(score_logits, dim=-1)
    sampling_log_probabilities = torch.log_softmax(
        score_logits / temperature, dim=-1
    )
    choices = torch.multinomial(
        sampling_log_probabilities.exp(), 1, generator=generator
    ).squeeze(-1)
    model_selected = model_log_probabilities.gather(
        -1, choices[:, None]
    ).squeeze(-1)
    sampling_selected = sampling_log_probabilities.gather(
        -1, choices[:, None]
    ).squeeze(-1)
    return CategoricalSample(
        choices=choices,
        model_log_probability=model_selected,
        sampling_log_probability=sampling_selected,
    )


class PrefixDecoder(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        hidden: torch.Tensor,
        target_codes: torch.Tensor,
        *,
        catalog_rows: torch.Tensor | None = None,
        distance: PrefixDistanceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def sample(
        self,
        hidden: torch.Tensor,
        *,
        temperature: float,
        generator: torch.Generator | None,
        catalog_rows: torch.Tensor | None = None,
        distance: PrefixDistanceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class _CompileSafeGRUCell(nn.GRUCell):
    def forward(
        self,
        input: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)
        if hx is None:
            hx = torch.zeros(
                input.shape[0],
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
        elif not is_batched:
            hx = hx.unsqueeze(0)

        input_reset, input_update, input_new = F.linear(
            input, self.weight_ih, self.bias_ih
        ).chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = F.linear(
            hx, self.weight_hh, self.bias_hh
        ).chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        new = torch.tanh(input_new + reset * hidden_new)
        output = new + update * (hx - new)
        return output if is_batched else output.squeeze(0)


class HierarchicalPrefixDecoder(PrefixDecoder):
    def __init__(
        self,
        *,
        model_dim: int,
        codebook_size: int,
        prefix_depth: int,
        legality: LegalPrefixIndex,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.prefix_depth = prefix_depth
        object.__setattr__(self, "legality", legality)
        self.context = nn.Linear(model_dim, model_dim)
        self.cell = _CompileSafeGRUCell(model_dim, model_dim)
        self.code_embedding = nn.Embedding(codebook_size, model_dim)
        self.level_embedding = nn.Embedding(prefix_depth, model_dim)
        self.start_embedding = nn.Parameter(torch.zeros(model_dim))
        self.heads = nn.ModuleList(
            nn.Linear(model_dim, codebook_size) for _ in range(prefix_depth)
        )

    def _step(
        self,
        state: torch.Tensor,
        previous_codes: torch.Tensor | None,
        level: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoder_input = (
            self.start_embedding.expand(state.shape[0], -1)
            if previous_codes is None
            else self.code_embedding(previous_codes)
        )
        decoder_input = decoder_input + self.level_embedding.weight[level]
        state = self.cell(decoder_input, state)
        return state, self.heads[level](state)

    def forward(
        self,
        hidden: torch.Tensor,
        target_codes: torch.Tensor,
        *,
        catalog_rows: torch.Tensor | None = None,
        distance: PrefixDistanceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = hidden.shape[:-1]
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_targets = target_codes.reshape(-1, self.prefix_depth)
        flat_catalog_rows: torch.Tensor | None = None
        if catalog_rows is not None:
            if catalog_rows.shape != shape[:1]:
                raise ValueError("catalog_rows must have shape [batch]")
            expanded_rows = catalog_rows.reshape(
                shape[0], *([1] * (len(shape) - 1))
            ).expand(*shape)
            flat_catalog_rows = expanded_rows.reshape(-1)
        flat_distance = None if distance is None else distance.flatten()
        state = self.context(flat_hidden)
        logits: list[torch.Tensor] = []
        previous: torch.Tensor | None = None
        for level in range(self.prefix_depth):
            state, current = self._step(state, previous, level)
            valid = (
                self.legality.next_mask(level, flat_targets[:, :level])
                if flat_catalog_rows is None
                else self.legality.next_mask_for_catalog_rows(
                    level, flat_targets[:, :level], flat_catalog_rows
                )
            )
            if flat_distance is not None:
                current = current + flat_distance.level_bias(
                    level, flat_targets[:, :level]
                )
            logits.append(current.masked_fill(~valid, _ILLEGAL_LOGIT))
            previous = flat_targets[:, level]
        stacked = torch.stack(logits, dim=1).reshape(
            *shape, self.prefix_depth, -1
        )
        return stacked, target_codes

    def sample(
        self,
        hidden: torch.Tensor,
        *,
        temperature: float,
        generator: torch.Generator | None,
        catalog_rows: torch.Tensor | None = None,
        distance: PrefixDistanceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if catalog_rows is not None and catalog_rows.shape != hidden.shape[:1]:
            raise ValueError("catalog_rows must have shape [batch]")
        state = self.context(hidden)
        codes: list[torch.Tensor] = []
        model_log_probability = torch.zeros(
            hidden.shape[0], dtype=torch.float32, device=hidden.device
        )
        sampling_log_probability = torch.zeros(
            hidden.shape[0], dtype=torch.float32, device=hidden.device
        )
        previous: torch.Tensor | None = None
        for level in range(self.prefix_depth):
            state, logits = self._step(state, previous, level)
            parents = (
                torch.empty(
                    (hidden.shape[0], 0),
                    dtype=torch.long,
                    device=hidden.device,
                )
                if not codes
                else torch.stack(codes, dim=-1)
            )
            valid = (
                self.legality.next_mask(level, parents)
                if catalog_rows is None
                else self.legality.next_mask_for_catalog_rows(
                    level, parents, catalog_rows
                )
            )
            if distance is not None:
                logits = logits + distance.level_bias(level, parents)
            sampled = sample_categorical_logits(
                logits.masked_fill(~valid, _ILLEGAL_LOGIT),
                temperature=temperature,
                generator=generator,
            )
            choice = sampled.choices
            codes.append(choice)
            model_log_probability = (
                model_log_probability + sampled.model_log_probability
            )
            sampling_log_probability = (
                sampling_log_probability + sampled.sampling_log_probability
            )
            previous = choice
        return (
            torch.stack(codes, dim=-1),
            model_log_probability,
            sampling_log_probability,
        )


class FlatLegalPrefixDecoder(PrefixDecoder):
    def __init__(self, *, model_dim: int, legality: LegalPrefixIndex) -> None:
        super().__init__()
        object.__setattr__(self, "legality", legality)
        self.head = nn.Linear(model_dim, legality.vocabulary_size)

    def forward(
        self,
        hidden: torch.Tensor,
        target_codes: torch.Tensor,
        *,
        catalog_rows: torch.Tensor | None = None,
        distance: PrefixDistanceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.head(hidden)
        if distance is not None:
            logits = logits + distance.flat_bias()
        if catalog_rows is not None:
            if catalog_rows.shape != hidden.shape[:1]:
                raise ValueError("catalog_rows must have shape [batch]")
            catalog_mask = self.legality.city_legal_prefix_mask[catalog_rows]
            if logits.ndim > 2:
                catalog_mask = catalog_mask[:, None, :]
            logits = logits.masked_fill(~catalog_mask, _ILLEGAL_LOGIT)
        return logits, self.legality.token_ids(target_codes)

    def sample(
        self,
        hidden: torch.Tensor,
        *,
        temperature: float,
        generator: torch.Generator | None,
        catalog_rows: torch.Tensor | None = None,
        distance: PrefixDistanceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.head(hidden)
        if distance is not None:
            logits = logits + distance.flat_bias()
        if catalog_rows is not None:
            if catalog_rows.shape != hidden.shape[:1]:
                raise ValueError("catalog_rows must have shape [batch]")
            catalog_mask = self.legality.city_legal_prefix_mask[catalog_rows]
            logits = logits.masked_fill(~catalog_mask, _ILLEGAL_LOGIT)
        sampled = sample_categorical_logits(
            logits,
            temperature=temperature,
            generator=generator,
        )
        return (
            self.legality.codes[sampled.choices],
            sampled.model_log_probability,
            sampled.sampling_log_probability,
        )
