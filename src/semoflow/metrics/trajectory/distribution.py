from __future__ import annotations

import math

import torch

from semoflow.config.constants import PAD_ID


def marginal_distribution(
    tokens: torch.Tensor,
    vocab_size: int,
    *,
    pad_id: int = PAD_ID,
) -> torch.Tensor:
    flat = tokens.flatten()
    mask = flat != pad_id
    valid = flat[mask]
    if valid.numel() == 0:
        return torch.zeros(
            vocab_size, dtype=torch.float32, device=tokens.device
        )
    counts = torch.bincount(valid, minlength=vocab_size).float()
    counts = counts[:vocab_size]
    return counts / counts.sum()


def jsd(p: torch.Tensor, q: torch.Tensor) -> float:
    if p.shape != q.shape:
        raise ValueError("JSD inputs must have the same shape")
    if not torch.isfinite(p).all() or not torch.isfinite(q).all():
        raise ValueError("JSD inputs must contain only finite values")
    if torch.any(p < 0) or torch.any(q < 0):
        raise ValueError("JSD inputs must be non-negative")
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")
    p = p / p_sum
    q = q / q_sum
    m = (p + q) / 2
    positive_p = p > 0
    positive_q = q > 0
    kl_pm = (p[positive_p] * (p[positive_p] / m[positive_p]).log()).sum()
    kl_qm = (q[positive_q] * (q[positive_q] / m[positive_q]).log()).sum()
    value = 0.5 * kl_pm + 0.5 * kl_qm
    return float(torch.clamp(value, 0.0, math.log(2.0)).item())


def transition_matrix(
    tokens: torch.Tensor,
    vocab_size: int,
    *,
    pad_id: int = PAD_ID,
) -> torch.Tensor:
    counts = torch.zeros(
        vocab_size * vocab_size, dtype=torch.float32, device=tokens.device
    )
    src = tokens[:, :-1]
    dst = tokens[:, 1:]
    mask = (src != pad_id) & (dst != pad_id)
    flat_src = src[mask]
    flat_dst = dst[mask]
    if flat_src.numel() > 0:
        idx = flat_src.long() * vocab_size + flat_dst.long()
        counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    counts = counts.view(vocab_size, vocab_size)
    row_sums = counts.sum(dim=1, keepdim=True)
    zero_rows = (row_sums == 0).squeeze(1)
    counts[zero_rows] = 1.0 / vocab_size
    row_sums[zero_rows] = 1.0
    return counts / row_sums


def transition_distance(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    vocab_size: int,
    *,
    pad_id: int = PAD_ID,
) -> float:
    t_real = transition_matrix(tokens_real, vocab_size, pad_id=pad_id)
    t_gen = transition_matrix(tokens_gen, vocab_size, pad_id=pad_id)
    return float(torch.norm(t_real - t_gen, p="fro").item())


def visit_count_jsd(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    *,
    pad_id: int = 0,
) -> float:
    max_len = max(tokens_real.size(1), tokens_gen.size(1))
    counts_real = (tokens_real != pad_id).sum(dim=1)
    counts_gen = (tokens_gen != pad_id).sum(dim=1)
    p = torch.bincount(counts_real, minlength=max_len + 1).float()
    q = torch.bincount(counts_gen, minlength=max_len + 1).float()
    return jsd(p, q)


def od_pair_jsd(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    vocab_size: int,
    *,
    pad_id: int = 0,
    max_vocab: int = 500,
) -> float | None:
    if vocab_size > max_vocab:
        return None

    def _od_counts(tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens != pad_id
        valid = mask.any(dim=1)
        if not valid.any():
            return torch.zeros(
                vocab_size * vocab_size,
                dtype=torch.float32,
                device=tokens.device,
            )
        t = tokens[valid]
        m = mask[valid]
        first_idx = (m.float().cumsum(dim=1) == 1).float().argmax(dim=1)
        last_idx = (
            m.size(1)
            - 1
            - (m.flip(1).float().cumsum(dim=1) == 1).float().argmax(dim=1)
        )
        n = t.size(0)
        arange = torch.arange(n, device=tokens.device)
        origin = t[arange, first_idx]
        dest = t[arange, last_idx]
        pair_ids = origin * vocab_size + dest
        return torch.bincount(
            pair_ids, minlength=vocab_size * vocab_size
        ).float()

    p = _od_counts(tokens_real)
    q = _od_counts(tokens_gen)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        return None
    p = p / p_sum
    q = q / q_sum
    return jsd(p, q)


def token_change_count_jsd(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    *,
    pad_id: int = 0,
) -> float:
    def _change_counts(tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.long()
        both_valid = (t[:, :-1] != pad_id) & (t[:, 1:] != pad_id)
        changed = (t[:, :-1] != t[:, 1:]) & both_valid
        counts = changed.sum(dim=1)
        max_changes = tokens.size(1) - 1
        return torch.bincount(counts, minlength=max_changes + 1).float()

    p = _change_counts(tokens_real)
    q = _change_counts(tokens_gen)
    return jsd(p, q)


def token_sequence_min_distance(
    tokens_gen: torch.Tensor,
    tokens_train: torch.Tensor,
    *,
    pad_id: int = 0,
    max_train: int = 200,
    max_gen: int = 200,
) -> float:
    r = tokens_gen[:max_gen]
    t = tokens_train[:max_train]
    mismatch = (r.unsqueeze(1) != t.unsqueeze(0)).float()
    valid = ((r.unsqueeze(1) != pad_id) | (t.unsqueeze(0) != pad_id)).float()
    hamming = mismatch.sum(-1) / valid.sum(-1).clamp(min=1)
    min_dist = hamming.min(dim=1).values
    return float(min_dist.mean().item())


def unique_location_count_jsd(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    *,
    pad_id: int = PAD_ID,
) -> float:
    max_len = max(tokens_real.size(1), tokens_gen.size(1))

    def _counts(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.numel() == 0:
            return torch.zeros(max_len + 1, dtype=torch.float32)
        values = tokens.long()
        sentinel = int(values.max().item()) + 1
        if sentinel == pad_id:
            sentinel += 1
        valid = values != pad_id
        sortable = torch.where(
            valid, values, torch.full_like(values, sentinel)
        )
        sorted_values = sortable.sort(dim=1).values
        sorted_valid = sorted_values != sentinel
        is_first = torch.ones(
            sorted_values.size(0), 1, dtype=torch.bool, device=values.device
        )
        changed = torch.cat(
            [is_first, sorted_values[:, 1:] != sorted_values[:, :-1]],
            dim=1,
        )
        unique_per_row = (sorted_valid & changed).sum(dim=1)
        return torch.bincount(unique_per_row, minlength=max_len + 1).float()

    p = _counts(tokens_real)
    q = _counts(tokens_gen)
    p = p / p.sum().clamp_min(1.0)
    q = q / q.sum().clamp_min(1.0)
    return jsd(p, q)


def location_time_joint_jsd(
    location_real: torch.Tensor,
    hour_real: torch.Tensor,
    location_gen: torch.Tensor,
    hour_gen: torch.Tensor,
    location_vocab: int,
    hour_vocab: int,
    *,
    pad_id: int = PAD_ID,
    max_location_vocab: int = 500,
) -> float | None:
    if location_vocab > max_location_vocab:
        return None
    joint_size = location_vocab * hour_vocab

    def _joint_counts(loc: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        valid = (loc != pad_id) & (hr != pad_id)
        loc_v = loc[valid].long()
        hr_v = hr[valid].long()
        ids = loc_v * hour_vocab + hr_v
        counts = torch.bincount(ids, minlength=joint_size).float()
        total = counts.sum()
        return counts / total if total > 0 else counts

    p = _joint_counts(location_real.flatten(), hour_real.flatten())
    q = _joint_counts(location_gen.flatten(), hour_gen.flatten())
    return jsd(p, q)


def ngram_counts_hashed(
    tokens: torch.Tensor,
    vocab_size: int,
    *,
    n: int,
    pad_id: int = PAD_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.numel() == 0 or tokens.size(1) < n:
        return (
            torch.empty(0, dtype=torch.int64, device=tokens.device),
            torch.empty(0, dtype=torch.int64, device=tokens.device),
        )
    tokens = tokens.long()
    windows = tokens.unfold(1, n, 1)
    valid = (tokens != pad_id).unfold(1, n, 1).all(dim=-1)
    kept = windows[valid]
    if kept.numel() == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=tokens.device),
            torch.empty(0, dtype=torch.int64, device=tokens.device),
        )
    weights = torch.tensor(
        [vocab_size ** (n - 1 - j) for j in range(n)],
        dtype=torch.int64,
        device=tokens.device,
    )
    hashes = (kept * weights).sum(dim=-1)
    return torch.unique(hashes, return_counts=True)


def activity_ngram_jsd(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    vocab_size: int,
    *,
    n: int = 3,
    pad_id: int = PAD_ID,
    max_vocab: int = 500,
    real_counts: tuple[torch.Tensor, torch.Tensor] | None = None,
    gen_counts: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> float | None:
    if vocab_size > max_vocab:
        return None
    r_uniq, r_cnt = real_counts or ngram_counts_hashed(
        tokens_real, vocab_size, n=n, pad_id=pad_id
    )
    g_uniq, g_cnt = gen_counts or ngram_counts_hashed(
        tokens_gen, vocab_size, n=n, pad_id=pad_id
    )
    all_hashes = torch.cat([r_uniq, g_uniq])
    keys, inverse = torch.unique(all_hashes, return_inverse=True)
    p = torch.zeros(keys.numel(), dtype=torch.float32, device=keys.device)
    q = torch.zeros(keys.numel(), dtype=torch.float32, device=keys.device)
    p.scatter_add_(0, inverse[: r_uniq.numel()], r_cnt.float())
    q.scatter_add_(0, inverse[r_uniq.numel() :], g_cnt.float())
    return jsd(p, q)


def distinct_ngram_ratio(
    tokens: torch.Tensor,
    vocab_size: int,
    *,
    n: int = 2,
    pad_id: int = PAD_ID,
    max_vocab: int = 5000,
    counts: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> float | None:
    if vocab_size > max_vocab:
        return None
    uniq, cnt = counts or ngram_counts_hashed(
        tokens, vocab_size, n=n, pad_id=pad_id
    )
    total = int(cnt.sum().item())
    if total == 0:
        return None
    return uniq.numel() / total


def self_ngram_overlap(
    tokens: torch.Tensor,
    vocab_size: int,
    *,
    n: int = 2,
    pad_id: int = PAD_ID,
    max_vocab: int = 5000,
    sample_pairs: int = 3000,
    seed: int = 0,
) -> float | None:
    if vocab_size > max_vocab:
        return None
    n_seq = tokens.size(0)
    if n_seq < 2:
        return None
    rng = torch.Generator().manual_seed(seed)
    idx_a = torch.randint(0, n_seq, (sample_pairs,), generator=rng).tolist()
    idx_b = torch.randint(0, n_seq, (sample_pairs,), generator=rng).tolist()

    hash_cache: dict[int, set[int]] = {}

    def _grams(seq_idx: int) -> set[int]:
        cached = hash_cache.get(seq_idx)
        if cached is not None:
            return cached
        uniq, _ = ngram_counts_hashed(
            tokens[seq_idx : seq_idx + 1],
            vocab_size,
            n=n,
            pad_id=pad_id,
        )
        cached = set(uniq.tolist())
        hash_cache[seq_idx] = cached
        return cached

    overlaps: list[float] = []
    for a, b in zip(idx_a, idx_b, strict=True):
        if a == b:
            continue
        ga = _grams(a)
        gb = _grams(b)
        if not ga or not gb:
            continue
        inter = len(ga & gb)
        union = len(ga | gb)
        if union > 0:
            overlaps.append(inter / union)
    if not overlaps:
        return None
    return sum(overlaps) / len(overlaps)


def token_marginal_jsd(
    tokens_real: torch.Tensor,
    tokens_gen: torch.Tensor,
    vocab_size: int,
    *,
    pad_id: int = PAD_ID,
) -> float:
    real = tokens_real[tokens_real != pad_id]
    gen = tokens_gen[tokens_gen != pad_id]
    p = torch.bincount(real, minlength=vocab_size).float()[:vocab_size]
    q = torch.bincount(gen, minlength=vocab_size).float()[:vocab_size]
    return jsd(p, q)
