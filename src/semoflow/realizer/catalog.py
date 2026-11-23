from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch

from semoflow.poi.sids import find_sorted, packed_code_keys, sid_code_columns

_COUNT_CHUNK = 1_048_576


def _to(instance, device: torch.device | str):
    return dataclasses.replace(
        instance,
        **{
            field.name: value.to(device)
            for field in dataclasses.fields(instance)
            if isinstance(value := getattr(instance, field.name), torch.Tensor)
            or isinstance(value, SuffixCatalog)
        },
    )


@dataclass(frozen=True, slots=True)
class SuffixCatalog:
    codes: torch.Tensor
    code_keys: torch.Tensor
    prefix_keys: torch.Tensor
    counts: torch.Tensor
    log_prior: torch.Tensor
    codebooks: torch.Tensor
    prefix_depth: int
    city: torch.Tensor | None

    @classmethod
    def from_codes(
        cls,
        catalog_codes: torch.Tensor,
        codebooks: torch.Tensor,
        *,
        prefix_depth: int,
        observed_codes: torch.Tensor,
        count_device: torch.device | str,
        city: torch.Tensor | None,
    ) -> SuffixCatalog:
        size = int(codebooks.shape[1])
        codes, inverse = torch.unique(
            catalog_codes.long().cpu(), dim=0, return_inverse=True
        )
        code_keys = packed_code_keys(codes, codebook_size=size)
        sorted_keys = code_keys.to(count_device)
        counts = torch.zeros(
            codes.shape[0], dtype=torch.long, device=count_device
        )
        for chunk in observed_codes.split(_COUNT_CHUNK):
            keys = packed_code_keys(
                chunk.to(count_device, dtype=torch.long), codebook_size=size
            )
            index, found = find_sorted(sorted_keys, keys)
            counts.scatter_add_(0, index[found], torch.ones_like(index[found]))
        if city is not None:
            rows = torch.arange(inverse.numel())
            first = torch.full((codes.shape[0],), inverse.numel())
            first.scatter_reduce_(0, inverse, rows, reduce="amin")
            city = city.long().cpu()[first]
        return cls(
            codes=codes,
            code_keys=code_keys,
            prefix_keys=packed_code_keys(
                codes[:, :prefix_depth], codebook_size=size
            ),
            counts=counts.cpu(),
            log_prior=torch.log1p(counts.cpu().float()),
            codebooks=codebooks.float().cpu().contiguous(),
            prefix_depth=prefix_depth,
            city=city,
        )

    @classmethod
    def from_lookup(
        cls,
        lookup_path: Path,
        codebooks: torch.Tensor,
        *,
        prefix_depth: int,
        observed_codes: torch.Tensor,
        count_device: torch.device | str = "cpu",
        with_city: bool = False,
    ) -> SuffixCatalog:
        lookup = pl.read_parquet(lookup_path)
        codes = lookup.select(sid_code_columns(lookup.columns)).to_numpy()
        city = lookup["city_id"].to_numpy().astype("int64")
        return cls.from_codes(
            torch.from_numpy(codes.astype("int64")),
            codebooks,
            prefix_depth=prefix_depth,
            observed_codes=observed_codes,
            count_device=count_device,
            city=torch.from_numpy(city) if with_city else None,
        )

    def to(self, device: torch.device | str) -> SuffixCatalog:
        return _to(self, device)

    def _prefix_key(self, prefixes: torch.Tensor) -> torch.Tensor:
        return packed_code_keys(
            prefixes[..., : self.prefix_depth].to(self.prefix_keys.device),
            codebook_size=int(self.codebooks.shape[1]),
        )

    def supports(self, prefixes: torch.Tensor) -> torch.Tensor:
        return find_sorted(self.prefix_keys, self._prefix_key(prefixes))[1]

    def prefix_ranges(
        self, prefixes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self._prefix_key(prefixes)
        left = torch.searchsorted(self.prefix_keys, keys, side="left")
        right = torch.searchsorted(self.prefix_keys, keys, side="right")
        missing = left == right
        if missing.any():
            prefix = prefixes[torch.where(missing)[0][0]].tolist()
            raise ValueError(
                f"catalog has no legal SID continuation for prefix {prefix}"
            )
        return left, right

    def log_prior_for_codes(self, codes: torch.Tensor) -> torch.Tensor:
        keys = packed_code_keys(
            codes, codebook_size=int(self.codebooks.shape[1])
        )
        index, found = find_sorted(self.code_keys, keys)
        return torch.where(found, self.log_prior[index], 0.0)

    def candidate_table(self, max_candidates: int) -> CandidateTable:
        prefixes, counts = torch.unique_consecutive(
            self.prefix_keys, return_counts=True
        )
        frequency = self.counts.tolist()
        table = torch.zeros(
            (prefixes.numel(), max_candidates), dtype=torch.long
        )
        mask = torch.zeros(table.shape, dtype=torch.bool)
        start = 0
        for index, count in enumerate(counts.tolist()):
            rows = sorted(
                range(start, start + count), key=lambda row: -frequency[row]
            )
            if len(rows) > max_candidates:
                frequent = max_candidates // 2
                remainder = rows[frequent:]
                stride = max(1, len(remainder) // max_candidates)
                rows = (rows[:frequent] + remainder[::stride])[:max_candidates]
            table[index, : len(rows)] = torch.tensor(rows)
            mask[index, : len(rows)] = True
            start += count
        return CandidateTable(
            catalog=self,
            prefix_keys=prefixes,
            codes=self.codes[table].masked_fill(~mask[..., None], 0),
            log_prior=self.log_prior[table].masked_fill(~mask, float("-inf")),
            mask=mask,
        )


@dataclass(frozen=True, slots=True)
class CandidateTable:
    catalog: SuffixCatalog
    prefix_keys: torch.Tensor
    codes: torch.Tensor
    log_prior: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device | str) -> CandidateTable:
        return _to(self, device)

    def prefix_index(self, prefixes: torch.Tensor) -> torch.Tensor:
        index, found = find_sorted(
            self.prefix_keys, self.catalog._prefix_key(prefixes)
        )
        if not found.all():
            raise ValueError(
                "catalog has no legal SID continuation for prefix "
                f"{prefixes[~found][0].tolist()}"
            )
        return index


__all__ = [
    "CandidateTable",
    "SuffixCatalog",
]
