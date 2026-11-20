from __future__ import annotations

import torch
from torch.utils.data import Dataset, Sampler


class TimeAwareSIDDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self,
        sid_codes: torch.Tensor,
        time_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
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
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.tensor_batches = bool(tensor_batches)
        self._epoch = 0
        lengths = lengths.long().cpu()
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
    codes = torch.stack([row[0] for row in rows]).long()
    times = torch.stack([row[1] for row in rows]).long()
    valid = torch.stack([row[2] for row in rows]).bool()
    max_len = max(1, int(valid.sum(dim=1).max().item()))
    return {
        "sid_codes": codes[:, :max_len],
        "time_tokens": times[:, :max_len],
        "valid_mask": valid[:, :max_len],
    }


__all__ = [
    "LengthBucketBatchSampler",
    "TimeAwareSIDDataset",
    "collate_time_aware_rows",
]
