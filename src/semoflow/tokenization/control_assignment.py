from __future__ import annotations

import numpy as np

__all__ = ["frequency_codes", "suffix_codes"]


def frequency_codes(
    values: list[str], *, codebook_size: int
) -> tuple[np.ndarray, float]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) <= codebook_size:
        table = {value: index for index, (value, _) in enumerate(ordered)}
        pooled = 0
    else:
        table = {
            value: index
            for index, (value, _) in enumerate(ordered[: codebook_size - 1])
        }
        pooled = sum(count for _, count in ordered[codebook_size - 1 :])
    codes = np.asarray(
        [table.get(value, codebook_size - 1) for value in values],
        dtype=np.int32,
    )
    return codes, pooled / len(values) if values else 0.0


def _split_members(
    features: np.ndarray,
    members: np.ndarray,
    codes: np.ndarray,
    *,
    level: int,
    codebook_size: int,
    suffix_levels: int,
    seed: int,
) -> int:
    size = len(members)
    if size == 1:
        return 0
    if level >= suffix_levels:
        return size - 1
    if size <= codebook_size:
        codes[members, level] = np.arange(size, dtype=np.int32)
        return 0

    from sklearn.cluster import MiniBatchKMeans

    labels = MiniBatchKMeans(
        n_clusters=codebook_size,
        random_state=seed,
        n_init=3,
        batch_size=1024,
    ).fit_predict(features[members])
    codes[members, level] = labels.astype(np.int32)
    collisions = 0
    for cluster in np.unique(labels):
        collisions += _split_members(
            features,
            members[labels == cluster],
            codes,
            level=level + 1,
            codebook_size=codebook_size,
            suffix_levels=suffix_levels,
            seed=seed,
        )
    return collisions


def suffix_codes(
    features: np.ndarray,
    groups: np.ndarray,
    *,
    codebook_size: int,
    suffix_levels: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    codes = np.zeros((len(groups), suffix_levels), dtype=np.int32)
    collisions = 0
    order = np.argsort(groups, kind="stable")
    boundaries = np.flatnonzero(np.diff(groups[order])) + 1
    for members in np.split(order, boundaries):
        collisions += _split_members(
            features,
            members,
            codes,
            level=0,
            codebook_size=codebook_size,
            suffix_levels=suffix_levels,
            seed=seed,
        )
    return codes, collisions
