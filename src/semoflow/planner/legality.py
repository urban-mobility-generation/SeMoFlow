from __future__ import annotations

import torch
import torch.nn as nn


class LegalPrefixIndex(nn.Module):
    def __init__(
        self,
        legal_prefixes: torch.Tensor,
        *,
        codebook_size: int,
        prefix_depth: int,
        city_legal_prefix_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        codes = torch.as_tensor(legal_prefixes, dtype=torch.long)
        self.codebook_size = int(codebook_size)
        self.prefix_depth = int(prefix_depth)
        self.register_buffer("codes", codes.clone(), persistent=False)

        city_mask = torch.as_tensor(city_legal_prefix_mask, dtype=torch.bool)

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
            self.register_buffer(
                f"level_{level}_parent_ids",
                unique_parent_ids,
                persistent=False,
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

    def next_mask(
        self,
        level: int,
        parents: torch.Tensor,
        catalog_rows: torch.Tensor,
    ) -> torch.Tensor:
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
