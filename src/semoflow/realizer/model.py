from __future__ import annotations

import math
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


def model_type(latent_target_scope: str) -> str:
    if latent_target_scope == "full_sid":
        return "prefix_time_full_sid_latent_flow_realizer"
    return "prefix_time_suffix_latent_flow_realizer"


_ARCHITECTURE_CHOICES = {
    "prefix_embedding_mode": ("learned", "hybrid_frozen_codebook"),
    "tail_representation": ("summed", "factorized"),
    "latent_target_scope": ("suffix", "full_sid"),
    "catalog_decoder_mode": ("nearest", "calibrated"),
}


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
        dim_feedforward=max(1, int(model_dim * mlp_ratio)),
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
        prefix_depth: int = 3,
        num_codebooks: int = 6,
        codebook_size: int = 64,
        time_vocab: int = 42,
        latent_dim: int = 128,
        model_dim: int = 256,
        condition_depth: int = 4,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        prefix_embedding_mode: str = "learned",
        prefix_plan_codebooks: torch.Tensor | None = None,
        tail_representation: str = "summed",
        latent_target_scope: str = "suffix",
        catalog_decoder_mode: str = "nearest",
        city_vocab: int = 0,
        spatial_conditioning: bool = False,
        spatial_num_basis: int = 8,
        spatial_max_km: float = 50.0,
    ) -> None:
        super().__init__()
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if not 0 < prefix_depth < num_codebooks:
            raise ValueError(
                "prefix_depth must be strictly between zero and num_codebooks"
            )
        if codebook_size < 2 or time_vocab < 1 or latent_dim < 1:
            raise ValueError(
                "codebook, time, and latent dimensions are invalid"
            )
        if model_dim < 1 or model_dim % num_heads:
            raise ValueError(
                "model_dim must be positive and divisible by num_heads"
            )
        if condition_depth < 1 or depth < 1 or mlp_ratio <= 0.0:
            raise ValueError(
                "transformer depths and mlp_ratio must be positive"
            )
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        for name, value in (
            ("prefix_embedding_mode", prefix_embedding_mode),
            ("tail_representation", tail_representation),
            ("latent_target_scope", latent_target_scope),
            ("catalog_decoder_mode", catalog_decoder_mode),
        ):
            choices = _ARCHITECTURE_CHOICES[name]
            if value not in choices:
                raise ValueError(
                    f"{name} must be " + " or ".join(map(repr, choices))
                )
        if (
            latent_target_scope == "full_sid"
            and tail_representation != "summed"
        ):
            raise ValueError(
                "full-SID latent targets require tail_representation='summed'"
            )
        if (
            tail_representation == "factorized"
            and catalog_decoder_mode == "calibrated"
        ):
            raise ValueError(
                "factorized tails are incompatible with calibrated catalog "
                "decoding"
            )
        if prefix_embedding_mode == "hybrid_frozen_codebook":
            if prefix_plan_codebooks is None:
                raise ValueError(
                    "hybrid_frozen_codebook requires prefix_plan_codebooks"
                )
            if prefix_plan_codebooks.ndim != 3 or (
                prefix_plan_codebooks.shape[:2]
                != (prefix_depth, codebook_size)
            ):
                raise ValueError(
                    "prefix_plan_codebooks must have shape "
                    "[prefix_depth, codebook_size, embedding_dim]"
                )
        elif prefix_plan_codebooks is not None:
            raise ValueError(
                "prefix_plan_codebooks is only valid for "
                "hybrid_frozen_codebook"
            )
        if city_vocab < 0:
            raise ValueError("city_vocab must be non-negative")
        if spatial_conditioning and spatial_num_basis < 1:
            raise ValueError("spatial_num_basis must be positive")
        if spatial_conditioning and spatial_max_km <= 0.0:
            raise ValueError("spatial_max_km must be positive")

        self.max_seq_len = int(max_seq_len)
        self.prefix_depth = int(prefix_depth)
        self.num_codebooks = int(num_codebooks)
        self.suffix_depth = self.num_codebooks - self.prefix_depth
        self.codebook_size = int(codebook_size)
        self.time_vocab = int(time_vocab)
        self.latent_dim = int(latent_dim)
        self.model_dim = int(model_dim)
        self.prefix_embedding_mode = prefix_embedding_mode
        self.tail_representation = tail_representation
        self.latent_target_scope = latent_target_scope
        self.target_level_start = (
            self.prefix_depth if latent_target_scope == "suffix" else 0
        )
        self.target_depth = self.num_codebooks - self.target_level_start
        self.catalog_decoder_mode = catalog_decoder_mode
        self.city_vocab = int(city_vocab)
        if self.city_vocab > 0:
            self.city_embedding = nn.Embedding(self.city_vocab, model_dim)
            nn.init.zeros_(self.city_embedding.weight)
        self.spatial_conditioning = bool(spatial_conditioning)
        self.spatial_num_basis = int(spatial_num_basis)
        self.spatial_max_km = float(spatial_max_km)
        if self.spatial_conditioning:
            self.spatial_head = nn.Linear(self.spatial_num_basis, model_dim)
            nn.init.zeros_(self.spatial_head.weight)
            nn.init.zeros_(self.spatial_head.bias)
            centers = torch.linspace(
                0.0, math.log1p(self.spatial_max_km), self.spatial_num_basis
            )
            width = (
                float(centers[1] - centers[0])
                if self.spatial_num_basis > 1
                else 1.0
            )
            self.register_buffer("spatial_centers", centers, persistent=False)
            self.spatial_basis_width = max(width, 1e-3)
        self.flow_latent_dim = (
            self.latent_dim * self.suffix_depth
            if tail_representation == "factorized"
            else self.latent_dim
        )
        self.prefix_pad_id = self.codebook_size
        self.pad_id = self.codebook_size + 1

        self.prefix_embeddings = nn.ModuleList(
            nn.Embedding(self.codebook_size + 1, self.model_dim)
            for _ in range(self.prefix_depth)
        )
        self.prefix_projection = nn.Linear(self.model_dim, self.model_dim)
        if prefix_embedding_mode == "hybrid_frozen_codebook":
            assert prefix_plan_codebooks is not None
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
            self.prefix_plan_gates = nn.Parameter(
                torch.zeros(self.prefix_depth)
            )
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

        self.input_projection = nn.Linear(self.flow_latent_dim, self.model_dim)
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
        self.output_projection = nn.Linear(
            self.model_dim, self.flow_latent_dim
        )
        if catalog_decoder_mode == "calibrated":
            candidate_input_dim = latent_dim * (1 + self.target_depth)
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

    def _validate_condition_inputs(
        self,
        prefix_codes: torch.Tensor,
        time_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if prefix_codes.ndim != 3:
            raise ValueError("prefix_codes must have shape [B, L, P]")
        if time_tokens.ndim != 2:
            raise ValueError("time_tokens must have shape [B, L]")
        if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with shape [B, L]")
        if prefix_codes.shape[:2] != valid_mask.shape:
            raise ValueError("prefix_codes and valid_mask must share [B, L]")
        if time_tokens.shape != valid_mask.shape:
            raise ValueError("time_tokens and valid_mask must share [B, L]")
        if prefix_codes.shape[-1] != self.prefix_depth:
            raise ValueError("prefix_codes width does not match prefix_depth")
        if valid_mask.size(1) > self.max_seq_len:
            raise ValueError(
                f"sequence length {valid_mask.size(1)} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )
        if valid_mask.numel() and not valid_mask.any(dim=1).all():
            raise ValueError("each batch row must contain a valid event")
        active_prefix = prefix_codes[valid_mask]
        if active_prefix.numel() and (
            active_prefix.min() < 0
            or active_prefix.max() >= self.codebook_size
        ):
            raise ValueError(
                "prefix_codes contain values outside the codebook"
            )
        active_times = time_tokens[valid_mask]
        if active_times.numel() and (
            active_times.min() < 0 or active_times.max() >= self.time_vocab
        ):
            raise ValueError("time_tokens contain values outside time_vocab")

    def encode_conditioning(
        self,
        prefix_codes: torch.Tensor,
        time_tokens: torch.Tensor,
        valid_mask: torch.BoolTensor,
        *,
        city_id: torch.LongTensor | None = None,
        hop_distance_km: torch.Tensor | None = None,
    ) -> LatentSuffixConditioning:
        if prefix_codes.device != time_tokens.device:
            raise ValueError(
                "prefix_codes and time_tokens must share a device"
            )
        if prefix_codes.device != valid_mask.device:
            raise ValueError("prefix_codes and valid_mask must share a device")
        self._validate_condition_inputs(prefix_codes, time_tokens, valid_mask)
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
        if self.prefix_embedding_mode == "hybrid_frozen_codebook":
            prefix_plan_state = torch.zeros_like(prefix_state)
            for level in range(self.prefix_depth):
                prefix_plan_codes = prefix[..., level].clamp_max(
                    self.codebook_size - 1
                )
                prefix_plan = self.prefix_plan_codebooks[level][
                    prefix_plan_codes
                ]
                prefix_plan = prefix_plan.masked_fill(~valid[..., None], 0.0)
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
            if city_id is None:
                raise ValueError(
                    "city_id is required when city_vocab is positive"
                )
            city = city_id.long()
            if city.ndim != 1 or city.shape[0] != prefix.shape[0]:
                raise ValueError("city_id must have shape [B]")
            if int(city.min()) < 0 or int(city.max()) >= self.city_vocab:
                raise ValueError("city_id is outside its vocabulary")
            plan = plan + self.city_embedding(city)[:, None, :]
        elif city_id is not None:
            raise ValueError(
                "city_id was supplied but the model has city_vocab=0"
            )
        if self.spatial_conditioning:
            if hop_distance_km is None:
                raise ValueError(
                    "hop_distance_km is required when spatial_conditioning "
                    "is enabled"
                )
            hop = hop_distance_km.to(plan.device, dtype=plan.dtype)
            if hop.shape != valid.shape:
                raise ValueError("hop_distance_km must have shape [B, L]")
            feature = torch.log1p(hop.clamp_min(0.0))
            rbf = torch.exp(
                -0.5
                * (
                    (feature[..., None] - self.spatial_centers)
                    / self.spatial_basis_width
                )
                ** 2
            )
            plan = plan + self.spatial_head(rbf)
        elif hop_distance_km is not None:
            raise ValueError(
                "hop_distance_km was supplied but spatial_conditioning is "
                "disabled"
            )
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
        valid_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        if zt.ndim != 3:
            raise ValueError("zt must have shape [B, L, latent_dim]")
        if zt.shape[-1] != self.flow_latent_dim:
            raise ValueError(
                "zt width does not match flow_latent_dim="
                f"{self.flow_latent_dim}"
            )
        if t.ndim != 1 or t.size(0) != zt.size(0):
            raise ValueError("t must have shape [B]")
        if conditioning.plan.shape[:2] != zt.shape[:2]:
            raise ValueError("conditioning plan must align with zt")
        valid = conditioning.valid_mask if valid_mask is None else valid_mask
        if valid.shape != zt.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with shape [B, L]")
        if valid.device != zt.device or conditioning.plan.device != zt.device:
            raise ValueError(
                "zt, conditioning, and valid_mask must share a device"
            )

        positions = torch.arange(zt.size(1), device=zt.device)
        x = (
            self.input_projection(zt)
            + self.condition_projection(conditioning.plan)
            + self.flow_position_embedding(positions)[None, :, :]
            + self.timestep_embedding(t).unsqueeze(1)
        )
        x = self.flow_trunk(
            x,
            src_key_padding_mask=~valid,
        )
        prediction = self.output_projection(self.output_norm(x))
        return prediction.masked_fill(~valid[..., None], 0.0)

    def catalog_query_features(
        self,
        conditioning: LatentSuffixConditioning,
        normalized_tail: torch.Tensor,
        valid_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        if self.catalog_decoder_mode != "calibrated":
            raise RuntimeError(
                "catalog query features require calibrated decoding"
            )
        if normalized_tail.ndim != 2:
            raise ValueError("normalized_tail must have shape [N, D]")
        if valid_mask.shape != conditioning.valid_mask.shape:
            raise ValueError("valid_mask must align with conditioning")
        plan = conditioning.plan[valid_mask]
        if plan.shape[0] != normalized_tail.shape[0]:
            raise ValueError("active conditioning and tail batches must align")
        query = self.catalog_query(
            torch.cat((plan.float(), normalized_tail.float()), dim=-1)
        )
        return self.catalog_score_norm(query)

    def catalog_candidate_features(
        self,
        candidate_tail: torch.Tensor,
        candidate_level_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if self.catalog_decoder_mode != "calibrated":
            raise RuntimeError(
                "catalog candidate features require calibrated decoding"
            )
        if candidate_tail.ndim < 2 or candidate_level_embeddings.ndim < 3:
            raise ValueError("catalog candidate tensors have invalid ranks")
        if candidate_tail.shape[:-1] != candidate_level_embeddings.shape[:-2]:
            raise ValueError("catalog candidate leading dimensions must align")
        candidate_input = torch.cat(
            (
                candidate_tail.float(),
                candidate_level_embeddings.float().flatten(start_dim=-2),
            ),
            dim=-1,
        )
        candidate = self.catalog_candidate(candidate_input)
        return self.catalog_score_norm(candidate)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


__all__ = [
    "LatentSuffixConditioning",
    "PrefixTimeSuffixLatentFlowRealizer",
    "model_type",
]
