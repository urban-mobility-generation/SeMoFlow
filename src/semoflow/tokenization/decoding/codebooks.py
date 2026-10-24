from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from semoflow.tokenization.token_space import sid_code_names

if TYPE_CHECKING:
    from semoflow.tokenization.rqvae import RQVAE


@dataclass(frozen=True)
class SIDCodebookBundle:
    codebooks: torch.Tensor
    code_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.codebooks.ndim != 3:
            raise ValueError(
                "codebooks must have shape [n_codebooks, codebook_size, dim], "
                f"got {tuple(self.codebooks.shape)}"
            )
        if len(self.code_fields) != int(self.codebooks.shape[0]):
            raise ValueError(
                "code_fields length must match codebook count: "
                f"{len(self.code_fields)} vs {int(self.codebooks.shape[0])}"
            )

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: RQVAE,
        *,
        code_fields: Sequence[str] | None = None,
    ) -> SIDCodebookBundle:
        codebooks = tokenizer.get_codebooks().detach().cpu().float()
        fields = (
            tuple(code_fields)
            if code_fields is not None
            else sid_code_names(int(codebooks.shape[0]))
        )
        return cls(codebooks=codebooks, code_fields=fields)

    @property
    def n_codebooks(self) -> int:
        return int(self.codebooks.shape[0])

    @property
    def codebook_size(self) -> int:
        return int(self.codebooks.shape[1])

    @property
    def embedding_dim(self) -> int:
        return int(self.codebooks.shape[2])


MAX_STANDALONE_CODEBOOK_ARTIFACT_BYTES = 512 * 1024 * 1024


def _canonical_codebooks(codebooks: torch.Tensor) -> torch.Tensor:
    if not isinstance(codebooks, torch.Tensor):
        raise TypeError("codebooks must be a torch.Tensor")
    return codebooks.detach().cpu().float().contiguous()


def codebook_bundle_metadata(bundle: SIDCodebookBundle) -> dict[str, Any]:
    codebooks = _canonical_codebooks(bundle.codebooks)
    return {
        "shape": [int(value) for value in codebooks.shape],
        "dtype": str(codebooks.dtype),
        "code_fields": list(bundle.code_fields),
    }


def validate_codebook_compatibility(
    expected_codebooks: torch.Tensor,
    expected_code_fields: Sequence[str],
    actual: SIDCodebookBundle,
) -> None:
    expected = _canonical_codebooks(expected_codebooks)
    actual_codebooks = _canonical_codebooks(actual.codebooks)
    expected_fields = tuple(expected_code_fields)
    actual_fields = tuple(actual.code_fields)
    if expected.shape != actual_codebooks.shape:
        raise ValueError(
            "codebook shape mismatch: "
            f"checkpoint={tuple(expected.shape)}, supplied={tuple(actual_codebooks.shape)}"
        )
    if expected_fields != actual_fields:
        raise ValueError(
            "codebook field mismatch: "
            f"checkpoint={expected_fields}, supplied={actual_fields}"
        )
    if not torch.equal(expected, actual_codebooks):
        raise ValueError(
            "codebook values do not match the training checkpoint; "
            "the supplied artifact uses a different latent coordinate system"
        )


def codes_to_embeddings_from_codebooks(
    codes: torch.Tensor,
    codebooks: torch.Tensor,
) -> torch.Tensor:
    if codebooks.ndim != 3:
        raise ValueError(
            f"codebooks must have shape [M,K,D], got {tuple(codebooks.shape)}"
        )

    codes = codes.long()
    codebooks = codebooks.to(codes.device)

    if codes.shape[-1] != codebooks.shape[0]:
        raise ValueError(
            f"codes last dim must equal n_codebooks={codebooks.shape[0]}, "
            f"got shape={tuple(codes.shape)}"
        )

    if codes.numel() == 0:
        return torch.empty(
            *codes.shape[:-1],
            codebooks.shape[-1],
            device=codes.device,
            dtype=codebooks.dtype,
        )

    min_code = int(codes.min().item())
    max_code = int(codes.max().item())
    if min_code < 0 or max_code >= codebooks.shape[1]:
        raise ValueError(
            f"semantic code out of range for codebook_size={codebooks.shape[1]}: "
            f"min={min_code}, max={max_code}"
        )

    original_shape = codes.shape[:-1]
    n_codebooks = int(codebooks.shape[0])
    flat_codes = codes.reshape(-1, n_codebooks)
    level_idx = torch.arange(n_codebooks, device=codes.device)
    selected = codebooks[level_idx[None, :], flat_codes]
    return selected.sum(dim=1).reshape(*original_shape, codebooks.shape[-1])


def extract_codebook_bundle(
    artifact_path: str | Path,
    *,
    max_artifact_bytes: int | None = None,
) -> SIDCodebookBundle:
    path = Path(artifact_path)
    if max_artifact_bytes is not None:
        if max_artifact_bytes <= 0:
            raise ValueError("max_artifact_bytes must be positive")
        artifact_size = path.stat().st_size
        if artifact_size > max_artifact_bytes:
            raise ValueError(
                f"codebook artifact {path} is {artifact_size} bytes; "
                "use a standalone codebook artifact rather than a latent bank "
                f"(limit={max_artifact_bytes} bytes)"
            )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    code_fields: Sequence[str] | None = None
    if isinstance(payload, SIDCodebookBundle):
        code_fields = payload.code_fields
        codebooks = payload.codebooks

    elif isinstance(payload, torch.Tensor):
        codebooks = payload.detach().cpu().float()
    elif isinstance(payload, dict):
        raw = payload.get("codebooks")
        if raw is None:
            raw = payload.get("codebook_bundle")
        if isinstance(raw, SIDCodebookBundle):
            codebooks = raw.codebooks.detach().cpu().float()
            code_fields = raw.code_fields
        elif isinstance(raw, torch.Tensor):
            codebooks = raw.detach().cpu().float()
        else:
            state = payload.get("state_dict") or payload.get(
                "model_state_dict"
            )
            if not isinstance(state, dict):
                raise ValueError(
                    f"artifact at {path} does not contain exported codebooks"
                )
            codebooks = _codebooks_from_state_dict(state)
    else:
        raise TypeError(
            "codebook artifact must be a torch Tensor, SIDCodebookBundle, "
            f"or dict payload; got {type(payload).__name__}"
        )

    fields = (
        tuple(code_fields)
        if code_fields is not None
        else sid_code_names(int(codebooks.shape[0]))
    )
    return SIDCodebookBundle(
        codebooks=_canonical_codebooks(codebooks),
        code_fields=fields,
    )


def _codebooks_from_state_dict(state: dict[str, Any]) -> torch.Tensor:
    residual_keys = sorted(
        (key for key in state if key.startswith("quantizer.codebooks.")),
        key=lambda key: int(key.split(".")[2]),
    )
    tensors = [state[key].detach().cpu().float() for key in residual_keys]
    if tensors:
        return torch.stack(tensors, dim=0).contiguous()
    single = state.get("quantizer.codebook")
    if isinstance(single, torch.Tensor):
        return single.detach().cpu().float().unsqueeze(0).contiguous()
    raise ValueError("state_dict does not contain quantizer codebooks")


__all__ = [
    "MAX_STANDALONE_CODEBOOK_ARTIFACT_BYTES",
    "SIDCodebookBundle",
    "codebook_bundle_metadata",
    "codes_to_embeddings_from_codebooks",
    "extract_codebook_bundle",
    "validate_codebook_compatibility",
]
