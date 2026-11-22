from __future__ import annotations

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

    def flatten(self) -> PrefixDistanceContext:
        return PrefixDistanceContext(
            prior=self.prior,
            source_units=self.source_units.reshape(-1, 3),
            source_valid=self.source_valid.reshape(-1),
        )


def sample_categorical_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    probabilities = torch.log_softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(
        probabilities.exp(), 1, generator=generator
    ).squeeze(-1)


class _CompileSafeGRUCell(nn.GRUCell):
    """``nn.GRUCell`` written out in tensor ops, which compile fully."""

    def forward(self, input: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        input_reset, input_update, input_new = F.linear(
            input, self.weight_ih, self.bias_ih
        ).chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = F.linear(
            hx, self.weight_hh, self.bias_hh
        ).chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        new = torch.tanh(input_new + reset * hidden_new)
        return new + update * (hx - new)


class HierarchicalPrefixDecoder(nn.Module):
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

    def _level_logits(
        self,
        state: torch.Tensor,
        level: int,
        parents: torch.Tensor,
        catalog_rows: torch.Tensor,
        distance: PrefixDistanceContext | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the GRU past ``parents``; legal logits for ``level``."""
        decoder_input = (
            self.start_embedding.expand(state.shape[0], -1)
            if level == 0
            else self.code_embedding(parents[:, level - 1])
        )
        decoder_input = decoder_input + self.level_embedding.weight[level]
        state = self.cell(decoder_input, state)
        logits = self.heads[level](state)
        valid = self.legality.next_mask(level, parents, catalog_rows)
        if distance is not None:
            logits = logits + distance.level_bias(level, parents)
        return state, logits.masked_fill(~valid, _ILLEGAL_LOGIT)

    def forward(
        self,
        hidden: torch.Tensor,
        target_codes: torch.Tensor,
        *,
        catalog_rows: torch.Tensor,
        distance: PrefixDistanceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = hidden.shape[:-1]
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_targets = target_codes.reshape(-1, self.prefix_depth)
        flat_catalog_rows = (
            catalog_rows.reshape(shape[0], *([1] * (len(shape) - 1)))
            .expand(*shape)
            .reshape(-1)
        )
        flat_distance = None if distance is None else distance.flatten()
        state = self.context(flat_hidden)
        logits: list[torch.Tensor] = []
        for level in range(self.prefix_depth):
            state, level_logits = self._level_logits(
                state,
                level,
                flat_targets[:, :level],
                flat_catalog_rows,
                flat_distance,
            )
            logits.append(level_logits)
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
        catalog_rows: torch.Tensor,
        distance: PrefixDistanceContext | None = None,
    ) -> torch.Tensor:
        state = self.context(hidden)
        codes = torch.empty(
            (hidden.shape[0], 0), dtype=torch.long, device=hidden.device
        )
        for level in range(self.prefix_depth):
            state, logits = self._level_logits(
                state, level, codes, catalog_rows, distance
            )
            choice = sample_categorical_logits(
                logits, temperature=temperature, generator=generator
            )
            codes = torch.cat((codes, choice[:, None]), dim=-1)
        return codes
