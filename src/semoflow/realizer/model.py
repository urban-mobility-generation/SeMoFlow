from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256) -> None:
        super().__init__()
        half = freq_dim // 2
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half) / max(half - 1, 1)
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        return self.proj(torch.cat([angles.sin(), angles.cos()], dim=1))


class CircadianTokenEncoder(nn.Module):
    def __init__(
        self,
        time_vocab: int,
        output_dim: int,
        *,
        n_harmonics: int = 3,
        n_parts_of_day: int = 6,
    ) -> None:
        super().__init__()
        self.time_vocab = time_vocab
        self.n_harmonics = n_harmonics
        self.n_parts_of_day = n_parts_of_day
        self.n_weekdays = math.ceil(time_vocab / n_parts_of_day)
        self.weekday_embedding = nn.Embedding(self.n_weekdays, output_dim)
        self.part_embedding = nn.Embedding(n_parts_of_day, output_dim)
        self.projection = nn.Sequential(
            nn.Linear(4 * n_harmonics, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        valid = (tokens >= 0) & (tokens < self.time_vocab)
        safe = tokens.clamp(min=0, max=self.time_vocab - 1)
        weekday = (safe // self.n_parts_of_day).clamp(max=self.n_weekdays - 1)
        part = safe % self.n_parts_of_day
        weekly_phase = 2.0 * math.pi * safe.float() / self.time_vocab
        daily_phase = 2.0 * math.pi * part.float() / self.n_parts_of_day
        harmonic_ids = torch.arange(
            1, self.n_harmonics + 1, device=tokens.device, dtype=tokens.dtype
        )
        weekly = weekly_phase[..., None] * harmonic_ids
        daily = daily_phase[..., None] * harmonic_ids
        periodic = torch.cat(
            [weekly.sin(), weekly.cos(), daily.sin(), daily.cos()], dim=-1
        )
        encoded = (
            self.weekday_embedding(weekday)
            + self.part_embedding(part)
            + self.projection(periodic)
        )
        return encoded.masked_fill(~valid[..., None], 0.0)


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
        city_id: torch.LongTensor | None,
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


def load_realizer_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[PrefixTimeSuffixLatentFlowRealizer, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    state = checkpoint["model_state_dict"]
    codebooks = checkpoint["codebooks"].float().contiguous()
    prefix_depth = int(checkpoint["prefix_depth"])
    model = PrefixTimeSuffixLatentFlowRealizer(
        max_seq_len=int(state["event_position_embedding.weight"].size(0)),
        prefix_depth=prefix_depth,
        num_codebooks=int(checkpoint["num_codebooks"]),
        codebook_size=int(checkpoint["codebook_size"]),
        time_vocab=int(checkpoint["time_vocab"]),
        latent_dim=int(checkpoint["latent_dim"]),
        model_dim=int(config["model_dim"]),
        condition_depth=int(config["condition_depth"]),
        depth=int(config["depth"]),
        num_heads=int(config["num_heads"]),
        mlp_ratio=float(config["mlp_ratio"]),
        dropout=float(config["dropout"]),
        prefix_plan_codebooks=codebooks[:prefix_depth],
        city_vocab=int(config["city_vocab"]),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def realizer_max_events(checkpoint: Path) -> int:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    position = payload["model_state_dict"]["event_position_embedding.weight"]
    return int(position.shape[0])


__all__ = [
    "MODEL_TYPE",
    "LatentSuffixConditioning",
    "PrefixTimeSuffixLatentFlowRealizer",
    "load_realizer_checkpoint",
    "realizer_max_events",
]
