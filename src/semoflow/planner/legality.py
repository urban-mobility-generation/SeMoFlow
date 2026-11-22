from __future__ import annotations

import torch
import torch.nn as nn

from semoflow.tokenization.sids import find_sorted, packed_code_keys


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
        codes = legal_prefixes.long()
        self.codebook_size = int(codebook_size)
        self.prefix_depth = int(prefix_depth)
        self.register_buffer("codes", codes.clone(), persistent=False)

        city_mask = city_legal_prefix_mask.bool()

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
        return packed_code_keys(codes, codebook_size=self.codebook_size)

    def _find(
        self, sorted_ids: torch.Tensor, codes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each code row's position in ``sorted_ids`` and whether it is
        there; rows with out-of-range codes are never found."""
        in_range = ((codes >= 0) & (codes < self.codebook_size)).all(dim=-1)
        safe = codes.clamp(min=0, max=self.codebook_size - 1)
        positions, found = find_sorted(sorted_ids, self._flatten(safe))
        return positions, in_range & found

    def resolve_token_ids(
        self, codes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions, found = self._find(self.sorted_flat_ids, codes)
        return self.token_ids_by_sorted_flat[positions], found

    def parent_rows(
        self, level: int, parents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each ``level``-code parent's row in the level's tables, and
        whether the parent is legal."""
        if level == 0:
            rows = torch.zeros(
                parents.shape[0], dtype=torch.long, device=parents.device
            )
            return rows, torch.ones_like(rows, dtype=torch.bool)
        return self._find(getattr(self, f"level_{level}_parent_ids"), parents)

    def next_mask(
        self,
        level: int,
        parents: torch.Tensor,
        catalog_rows: torch.Tensor,
    ) -> torch.Tensor:
        city_masks = getattr(self, f"level_{level}_city_masks")
        rows, found = self.parent_rows(level, parents)
        return city_masks[catalog_rows, rows] & found[..., None]
