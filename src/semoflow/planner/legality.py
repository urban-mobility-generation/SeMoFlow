from __future__ import annotations

import hashlib

import torch
import torch.nn as nn


class LegalPrefixIndex(nn.Module):
    def __init__(
        self,
        legal_prefixes: torch.Tensor,
        *,
        codebook_size: int,
        prefix_depth: int,
        city_legal_prefix_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        codes = torch.as_tensor(legal_prefixes, dtype=torch.long)
        if codes.ndim != 2 or codes.shape[1] != prefix_depth:
            raise ValueError(
                "legal_prefixes must have shape [vocabulary, prefix_depth]"
            )
        if codes.shape[0] < 1:
            raise ValueError("legal_prefixes must not be empty")
        if bool((codes < 0).any()) or bool((codes >= codebook_size).any()):
            raise ValueError("legal prefix code is outside the codebook")
        if torch.unique(codes, dim=0).shape[0] != codes.shape[0]:
            raise ValueError("legal_prefixes must be unique")
        self.codebook_size = int(codebook_size)
        self.prefix_depth = int(prefix_depth)
        self._catalog_fingerprint = self._fingerprint(codes, canonical=True)
        self._catalog_order_fingerprint = self._fingerprint(
            codes, canonical=False
        )
        self.register_buffer("codes", codes.clone(), persistent=False)

        if city_legal_prefix_mask is None:
            city_mask = torch.ones((1, codes.shape[0]), dtype=torch.bool)
        else:
            city_mask = torch.as_tensor(
                city_legal_prefix_mask, dtype=torch.bool
            )
        if city_mask.ndim != 2 or city_mask.shape[1] != codes.shape[0]:
            raise ValueError(
                "city_legal_prefix_mask must have shape [cities, vocabulary]"
            )
        if city_mask.shape[0] < 1 or not bool(city_mask.any(dim=-1).all()):
            raise ValueError("each city must permit at least one prefix")
        self.register_buffer(
            "city_legal_prefix_mask", city_mask.clone(), persistent=False
        )

        flat = self._flatten(codes)
        sorted_flat_ids, token_ids = torch.sort(flat)
        self.register_buffer(
            "sorted_flat_ids", sorted_flat_ids, persistent=False
        )
        self.register_buffer(
            "token_ids_by_sorted_flat", token_ids, persistent=False
        )

        for level in range(prefix_depth):
            if level == 0:
                parent_ids = torch.zeros(codes.shape[0], dtype=torch.long)
            else:
                parent_ids = self._flatten(codes[:, :level])
            unique_parent_ids = torch.unique(parent_ids, sorted=True)
            parent_rows = torch.searchsorted(unique_parent_ids, parent_ids)
            masks = torch.zeros(
                (unique_parent_ids.shape[0], codebook_size), dtype=torch.bool
            )
            masks[parent_rows, codes[:, level]] = True
            self.register_buffer(
                f"level_{level}_parent_ids",
                unique_parent_ids,
                persistent=False,
            )
            self.register_buffer(
                f"level_{level}_masks", masks, persistent=False
            )
            city_masks = torch.zeros(
                (
                    city_mask.shape[0],
                    unique_parent_ids.shape[0],
                    codebook_size,
                ),
                dtype=torch.bool,
            )
            city_rows = torch.arange(city_mask.shape[0])[:, None].expand(
                -1, codes.shape[0]
            )
            expanded_parent_rows = parent_rows[None, :].expand(
                city_mask.shape[0], -1
            )
            expanded_children = codes[:, level][None, :].expand(
                city_mask.shape[0], -1
            )
            city_masks[
                city_rows[city_mask],
                expanded_parent_rows[city_mask],
                expanded_children[city_mask],
            ] = True
            self.register_buffer(
                f"level_{level}_city_masks",
                city_masks,
                persistent=False,
            )

    def _fingerprint(self, codes: torch.Tensor, *, canonical: bool) -> str:
        fingerprint_codes = codes
        if canonical:
            order = torch.argsort(self._flatten(codes), stable=True)
            fingerprint_codes = codes[order]
        header = (
            f"v1:{self.codebook_size}:{self.prefix_depth}:{codes.shape[0]}:"
        ).encode()
        payload = fingerprint_codes.cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(header + payload).hexdigest()

    @property
    def catalog_fingerprint(self) -> str:
        return self._catalog_fingerprint

    @property
    def catalog_order_fingerprint(self) -> str:
        return self._catalog_order_fingerprint

    @property
    def vocabulary_size(self) -> int:
        return int(self.codes.shape[0])

    def _flatten(self, codes: torch.Tensor) -> torch.Tensor:
        depth = codes.shape[-1]
        weights = torch.tensor(
            [
                self.codebook_size ** (depth - level - 1)
                for level in range(depth)
            ],
            dtype=torch.long,
            device=codes.device,
        )
        return (codes * weights).sum(dim=-1)

    def resolve_token_ids(
        self, codes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if codes.shape[-1] != self.prefix_depth:
            raise ValueError("codes must end with prefix_depth")
        in_range = ((codes >= 0) & (codes < self.codebook_size)).all(dim=-1)
        safe = codes.clamp(min=0, max=self.codebook_size - 1)
        flat = self._flatten(safe)
        positions = torch.searchsorted(self.sorted_flat_ids, flat)
        safe_positions = positions.clamp(max=self.vocabulary_size - 1)
        found = self.sorted_flat_ids[safe_positions] == flat
        return (
            self.token_ids_by_sorted_flat[safe_positions],
            in_range & found,
        )

    def contains(self, codes: torch.Tensor) -> torch.Tensor:
        return self.resolve_token_ids(codes)[1]

    def token_ids(self, codes: torch.Tensor) -> torch.Tensor:
        resolved, contained = self.resolve_token_ids(codes)
        if not bool(contained.all()):
            raise ValueError("prefix targets contain an illegal tuple")
        return resolved

    def next_mask_for_catalog_rows(
        self,
        level: int,
        parents: torch.Tensor,
        catalog_rows: torch.Tensor,
    ) -> torch.Tensor:
        if not 0 <= level < self.prefix_depth:
            raise ValueError("level is outside prefix_depth")
        if catalog_rows.ndim != 1 or catalog_rows.dtype != torch.long:
            raise ValueError("catalog_rows must be a long vector")
        if parents.shape[0] != catalog_rows.shape[0]:
            raise ValueError(
                "parents and catalog_rows batch dimensions differ"
            )
        if parents.shape[-1] != level:
            raise ValueError("parents must contain exactly `level` codes")
        if (
            catalog_rows.device != self.codes.device
            or parents.device != self.codes.device
        ):
            raise ValueError(
                "catalog_rows, parents, and legality index must share a device"
            )
        city_masks = getattr(self, f"level_{level}_city_masks")
        if level == 0:
            return city_masks[catalog_rows, 0]
        valid = ((parents >= 0) & (parents < self.codebook_size)).all(dim=-1)
        safe = parents.clamp(min=0, max=self.codebook_size - 1)
        flat = self._flatten(safe)
        parent_ids = getattr(self, f"level_{level}_parent_ids")
        positions = torch.searchsorted(parent_ids, flat)
        safe_positions = positions.clamp(max=parent_ids.shape[0] - 1)
        found = parent_ids[safe_positions] == flat
        result = city_masks[catalog_rows, safe_positions]
        return result & valid[..., None] & found[..., None]

    def next_mask(self, level: int, parents: torch.Tensor) -> torch.Tensor:
        if not 0 <= level < self.prefix_depth:
            raise ValueError("level is outside prefix_depth")
        if parents.shape[-1] != level:
            raise ValueError("parents must contain exactly `level` codes")
        masks = getattr(self, f"level_{level}_masks")
        if level == 0:
            return masks[0].expand(*parents.shape[:-1], -1)
        valid = ((parents >= 0) & (parents < self.codebook_size)).all(dim=-1)
        safe = parents.clamp(min=0, max=self.codebook_size - 1)
        flat = self._flatten(safe)
        parent_ids = getattr(self, f"level_{level}_parent_ids")
        positions = torch.searchsorted(parent_ids, flat)
        safe_positions = positions.clamp(max=parent_ids.shape[0] - 1)
        found = parent_ids[safe_positions] == flat
        result = masks[safe_positions]
        return result & valid[..., None] & found[..., None]
