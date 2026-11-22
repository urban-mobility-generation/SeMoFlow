from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from semoflow.realizer.modules import (
    CircadianTokenEncoder,
    TimestepEmbedder,
)


@dataclass(frozen=True, slots=True)
class LatentSuffixConditioning:
    plan: torch.Tensor
    valid_mask: torch.BoolTensor


MODEL_TYPE = "prefix_time_suffix_latent_flow_realizer"


def _transformer_encoder(
    model_dim: int,
    *,
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
    num_layers: int,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=model_dim,
        nhead=num_heads,
        dim_feedforward=int(model_dim * mlp_ratio),
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer, num_layers=num_layers, enable_nested_tensor=False
    )


class PrefixTimeSuffixLatentFlowRealizer(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len: int,
        prefix_depth: int,
        num_codebooks: int,
        codebook_size: int,
        time_vocab: int,
        latent_dim: int,
        model_dim: int,
        condition_depth: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        prefix_plan_codebooks: torch.Tensor,
        city_vocab: int,
    ) -> None:
        """Hard-prefix suffix flow over summed, whitened suffix-tail latents.

        Prefix codes are embedded by learned tables plus gated projections
        of the frozen tokenizer codebooks (``prefix_plan_codebooks``); the
        calibrated catalog head scores legal same-prefix candidates.
        """
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.prefix_depth = int(prefix_depth)
        self.num_codebooks = int(num_codebooks)
        self.suffix_depth = self.num_codebooks - self.prefix_depth
        self.codebook_size = int(codebook_size)
        self.time_vocab = int(time_vocab)
        self.latent_dim = int(latent_dim)
        self.model_dim = int(model_dim)
        self.city_vocab = int(city_vocab)
        if self.city_vocab > 0:
            self.city_embedding = nn.Embedding(self.city_vocab, model_dim)
            nn.init.zeros_(self.city_embedding.weight)
        self.prefix_pad_id = self.codebook_size
        self.pad_id = self.codebook_size + 1

        self.prefix_embeddings = nn.ModuleList(
            nn.Embedding(self.codebook_size + 1, self.model_dim)
            for _ in range(self.prefix_depth)
        )
        self.prefix_projection = nn.Linear(self.model_dim, self.model_dim)
        self.register_buffer(
            "prefix_plan_codebooks",
            prefix_plan_codebooks.detach().cpu().float().contiguous(),
        )
        prefix_embedding_dim = int(prefix_plan_codebooks.shape[-1])
        self.prefix_plan_norms = nn.ModuleList(
            nn.LayerNorm(prefix_embedding_dim)
            for _ in range(self.prefix_depth)
        )
        self.prefix_plan_projections = nn.ModuleList(
            nn.Linear(prefix_embedding_dim, self.model_dim)
            for _ in range(self.prefix_depth)
        )
        self.prefix_plan_gates = nn.Parameter(torch.zeros(self.prefix_depth))
        self.time_encoder = CircadianTokenEncoder(
            self.time_vocab, self.model_dim
        )
        self.event_position_embedding = nn.Embedding(
            self.max_seq_len, self.model_dim
        )

        self.condition_encoder = _transformer_encoder(
            self.model_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_layers=condition_depth,
        )

        self.input_projection = nn.Linear(self.latent_dim, self.model_dim)
        self.condition_projection = nn.Linear(self.model_dim, self.model_dim)
        self.timestep_embedding = TimestepEmbedder(self.model_dim)
        self.flow_position_embedding = nn.Embedding(
            self.max_seq_len, self.model_dim
        )
        self.flow_trunk = _transformer_encoder(
            self.model_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_layers=depth,
        )
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.output_projection = nn.Linear(self.model_dim, self.latent_dim)
        candidate_input_dim = latent_dim * (1 + self.suffix_depth)
        self.catalog_query = nn.Sequential(
            nn.Linear(self.model_dim + latent_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.catalog_candidate = nn.Sequential(
            nn.LayerNorm(candidate_input_dim),
            nn.Linear(candidate_input_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.catalog_score_norm = nn.LayerNorm(self.model_dim)

    def encode_conditioning(
        self,
        prefix_codes: torch.Tensor,
        time_tokens: torch.Tensor,
        valid_mask: torch.BoolTensor,
        *,
        city_id: torch.LongTensor | None = None,
    ) -> LatentSuffixConditioning:
        valid = valid_mask.bool()
        prefix = torch.where(
            valid[..., None],
            prefix_codes.long(),
            torch.full_like(prefix_codes, self.prefix_pad_id),
        )
        prefix_state = sum(
            embedding(prefix[..., level])
            for level, embedding in enumerate(self.prefix_embeddings)
        )
        prefix_plan_state = torch.zeros_like(prefix_state)
        for level in range(self.prefix_depth):
            prefix_plan = self.prefix_plan_codebooks[level][
                prefix[..., level].clamp_max(self.codebook_size - 1)
            ].masked_fill(~valid[..., None], 0.0)
            prefix_plan_state = prefix_plan_state + self.prefix_plan_gates[
                level
            ] * self.prefix_plan_projections[level](
                self.prefix_plan_norms[level](prefix_plan)
            )
        prefix_state = prefix_state + prefix_plan_state
        positions = torch.arange(prefix.size(1), device=prefix.device)
        plan = (
            self.prefix_projection(prefix_state)
            + self.time_encoder(time_tokens.long())
            + self.event_position_embedding(positions)[None, :, :]
        )
        if self.city_vocab > 0:
            assert city_id is not None
            plan = plan + self.city_embedding(city_id.long())[:, None, :]
        plan = self.condition_encoder(
            plan,
            src_key_padding_mask=~valid,
        ).masked_fill(~valid[..., None], 0.0)
        return LatentSuffixConditioning(plan=plan, valid_mask=valid)

    def forward(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        conditioning: LatentSuffixConditioning,
        *,
        valid_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        positions = torch.arange(zt.size(1), device=zt.device)
        x = (
            self.input_projection(zt)
            + self.condition_projection(conditioning.plan)
            + self.flow_position_embedding(positions)[None, :, :]
            + self.timestep_embedding(t).unsqueeze(1)
        )
        x = self.flow_trunk(x, src_key_padding_mask=~valid_mask)
        prediction = self.output_projection(self.output_norm(x))
        return prediction.masked_fill(~valid_mask[..., None], 0.0)

    def catalog_query_features(
        self,
        conditioning: LatentSuffixConditioning,
        normalized_tail: torch.Tensor,
        valid_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        plan = conditioning.plan[valid_mask]
        query = self.catalog_query(
            torch.cat((plan.float(), normalized_tail.float()), dim=-1)
        )
        return self.catalog_score_norm(query)

    def catalog_candidate_features(
        self,
        candidate_tail: torch.Tensor,
        candidate_level_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        candidate_input = torch.cat(
            (
                candidate_tail.float(),
                candidate_level_embeddings.float().flatten(start_dim=-2),
            ),
            dim=-1,
        )
        candidate = self.catalog_candidate(candidate_input)
        return self.catalog_score_norm(candidate)


__all__ = [
    "MODEL_TYPE",
    "LatentSuffixConditioning",
    "PrefixTimeSuffixLatentFlowRealizer",
]
