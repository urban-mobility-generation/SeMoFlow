from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Sampler


class TimeAwareSIDDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self,
        sid_codes: torch.Tensor,
        time_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if sid_codes.ndim != 3:
            raise ValueError("sid_codes must have shape [N, L, M]")
        if time_tokens.shape != valid_mask.shape:
            raise ValueError("time_tokens and valid_mask must align")
        if valid_mask.shape != sid_codes.shape[:2]:
            raise ValueError("valid_mask must align with sid_codes")
        self.sid_codes = sid_codes
        self.time_tokens = time_tokens
        self.valid_mask = valid_mask.bool()

    def __len__(self) -> int:
        return int(self.sid_codes.size(0))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.sid_codes[index],
            self.time_tokens[index],
            self.valid_mask[index],
        )


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: torch.Tensor,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        tensor_batches: bool = False,
    ) -> None:
        if lengths.ndim != 1 or lengths.numel() == 0:
            raise ValueError("lengths must be a non-empty vector")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.tensor_batches = bool(tensor_batches)
        self._epoch = 0
        lengths = lengths.long().cpu()
        if bool((lengths < 1).any()):
            raise ValueError("length-bucket rows must contain an event")
        self._buckets = {
            int(length): torch.where(lengths == length)[0]
            for length in torch.unique(lengths, sorted=True)
        }

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1
        batches: list[torch.Tensor] = []
        for indices in self._buckets.values():
            if self.shuffle:
                order = torch.randperm(indices.numel(), generator=generator)
                ordered = indices[order]
            else:
                ordered = indices
            batches.extend(ordered.split(self.batch_size))
        if self.shuffle:
            order = torch.randperm(len(batches), generator=generator)
            batches = [batches[int(i)] for i in order.tolist()]
        if self.tensor_batches:
            yield from batches
        else:
            yield from (batch.tolist() for batch in batches)

    def __len__(self) -> int:
        return sum(
            (indices.numel() + self.batch_size - 1) // self.batch_size
            for indices in self._buckets.values()
        )


def collate_time_aware_rows(
    rows: list[tuple[torch.Tensor, ...]],
) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("cannot collate an empty SID batch")
    codes = torch.stack([row[0] for row in rows]).long()
    times = torch.stack([row[1] for row in rows]).long()
    valid = torch.stack([row[2] for row in rows]).bool()
    max_len = max(1, int(valid.sum(dim=1).max().item()))
    return {
        "sid_codes": codes[:, :max_len],
        "time_tokens": times[:, :max_len],
        "valid_mask": valid[:, :max_len],
    }


@dataclass(frozen=True, slots=True)
class ProposalCorruptionIndex:
    prefixes: torch.LongTensor
    keys: torch.LongTensor
    level1_start: torch.LongTensor
    level1_end: torch.LongTensor
    level2_start: torch.LongTensor
    level2_end: torch.LongTensor
    codebook_size: int

    def to(self, device: torch.device | str) -> ProposalCorruptionIndex:
        return ProposalCorruptionIndex(
            prefixes=self.prefixes.to(device),
            keys=self.keys.to(device),
            level1_start=self.level1_start.to(device),
            level1_end=self.level1_end.to(device),
            level2_start=self.level2_start.to(device),
            level2_end=self.level2_end.to(device),
            codebook_size=self.codebook_size,
        )


def build_proposal_corruption_index(
    catalog_codes: torch.Tensor,
    *,
    codebook_size: int,
) -> ProposalCorruptionIndex:
    if catalog_codes.ndim != 2 or catalog_codes.shape[1] != 6:
        raise ValueError("catalog codes must have shape [N, 6]")
    if codebook_size < 2:
        raise ValueError("codebook_size must be at least two")
    prefixes = torch.unique(catalog_codes[:, :3].long(), dim=0)
    if prefixes.shape[0] < 2:
        raise ValueError(
            "structured proposal corruption requires multiple legal prefixes"
        )
    keys = (
        prefixes[:, 0] * codebook_size + prefixes[:, 1]
    ) * codebook_size + prefixes[:, 2]
    order = torch.argsort(keys, stable=True)
    prefixes = prefixes[order].contiguous()
    keys = keys[order].contiguous()

    def _ranges(group_keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, counts = torch.unique_consecutive(group_keys, return_counts=True)
        ends = counts.cumsum(dim=0)
        starts = ends - counts
        return (
            torch.repeat_interleave(starts, counts).long(),
            torch.repeat_interleave(ends, counts).long(),
        )

    level1_start, level1_end = _ranges(prefixes[:, 0])
    level2_keys = prefixes[:, 0] * codebook_size + prefixes[:, 1]
    level2_start, level2_end = _ranges(level2_keys)
    return ProposalCorruptionIndex(
        prefixes=prefixes,
        keys=keys,
        level1_start=level1_start,
        level1_end=level1_end,
        level2_start=level2_start,
        level2_end=level2_end,
        codebook_size=codebook_size,
    )


def corrupt_proposal_prefixes_hierarchical(
    clean_codes: torch.Tensor,
    valid_mask: torch.BoolTensor,
    index: ProposalCorruptionIndex,
    *,
    corruption_rate: float,
    generator: torch.Generator | None = None,
    validate_membership: bool = True,
) -> tuple[torch.LongTensor, torch.BoolTensor, torch.LongTensor]:
    if not 0.0 <= corruption_rate <= 1.0:
        raise ValueError("corruption_rate must be in [0, 1]")
    if clean_codes.ndim != 3 or clean_codes.shape[-1] != 6:
        raise ValueError("clean_codes must have shape [B, L, 6]")
    if valid_mask.shape != clean_codes.shape[:2]:
        raise ValueError("valid_mask must align with clean_codes")
    proposal = clean_codes[..., :3].long().clone()
    depth = torch.zeros_like(valid_mask, dtype=torch.long)
    if corruption_rate == 0.0:
        return proposal, torch.zeros_like(valid_mask), depth

    if index.prefixes.device != proposal.device:
        raise ValueError(
            "proposal corruption index and clean_codes must share a device"
        )
    codebook_size = index.codebook_size
    prefix_keys = (
        proposal[..., 0] * codebook_size + proposal[..., 1]
    ) * codebook_size + proposal[..., 2]
    positions = torch.searchsorted(index.keys, prefix_keys)
    safe_positions = positions.clamp_max(index.keys.numel() - 1)
    if validate_membership:
        found = index.keys[safe_positions].eq(prefix_keys)
        if not bool(found[valid_mask].all()):
            raise ValueError(
                "active clean prefixes are missing from the corruption catalog"
            )

    level1_start = index.level1_start[safe_positions]
    level1_end = index.level1_end[safe_positions]
    level2_start = index.level2_start[safe_positions]
    level2_end = index.level2_end[safe_positions]
    sibling_count = (level2_end - level2_start - 1).clamp_min(0)
    subtree_count = (
        (level1_end - level1_start) - (level2_end - level2_start)
    ).clamp_min(0)
    region_count = (
        index.prefixes.shape[0] - (level1_end - level1_start)
    ).clamp_min(0)
    available = torch.stack(
        (region_count > 0, subtree_count > 0, sibling_count > 0),
        dim=-1,
    )
    any_available = available.any(dim=-1)
    corrupt = (
        (
            torch.rand(
                valid_mask.shape,
                device=proposal.device,
                generator=generator,
            )
            < corruption_rate
        )
        & valid_mask
        & any_available
    )

    choice_scores = torch.rand(
        (*valid_mask.shape, 3),
        device=proposal.device,
        generator=generator,
    ).masked_fill(~available, -1.0)
    choice = choice_scores.argmax(dim=-1)
    depth = torch.where(corrupt, choice + 1, depth)

    uniform = torch.rand(
        valid_mask.shape,
        device=proposal.device,
        generator=generator,
    )

    sibling_offset = (
        uniform * sibling_count.clamp_min(1).to(uniform.dtype)
    ).long()
    sibling_base = level2_start + sibling_offset
    sibling_index = sibling_base + (sibling_base >= safe_positions).long()

    subtree_before = level2_start - level1_start
    subtree_offset = (
        uniform * subtree_count.clamp_min(1).to(uniform.dtype)
    ).long()
    subtree_index = torch.where(
        subtree_offset < subtree_before,
        level1_start + subtree_offset,
        level2_end + (subtree_offset - subtree_before),
    )

    region_before = level1_start
    region_offset = (
        uniform * region_count.clamp_min(1).to(uniform.dtype)
    ).long()
    region_index = torch.where(
        region_offset < region_before,
        region_offset,
        level1_end + (region_offset - region_before),
    )

    candidate_index = torch.where(
        choice == 0,
        region_index,
        torch.where(choice == 1, subtree_index, sibling_index),
    ).clamp(0, index.prefixes.shape[0] - 1)
    candidate = index.prefixes[candidate_index]
    proposal = torch.where(corrupt[..., None], candidate, proposal)
    return proposal, corrupt, depth


__all__ = [
    "LengthBucketBatchSampler",
    "ProposalCorruptionIndex",
    "TimeAwareSIDDataset",
    "build_proposal_corruption_index",
    "collate_time_aware_rows",
    "corrupt_proposal_prefixes_hierarchical",
]
