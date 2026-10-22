from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr

from semoflow.metrics.semantic_id_trajectory import (
    collapse_consecutive_pois,
)
from semoflow.tokenization.decoding.grounding_index import (
    SemanticIDGroundingIndex,
)
from semoflow.tokenization.token_space import (
    sid_code_columns,
)


def empty_cf_alignment_metrics(k: int) -> dict[str, Any]:
    return {
        "knn_recall_at_k": None,
        "knn_k": k,
        "cosine_hamming_spearman": None,
        "cf_sample_size": 0,
        "cf_pair_sample_size": 0,
        "knn_recall_at_k_h2": None,
        "knn_recall_at_k_h3": None,
        "prefix_knn_recall_at_k": None,
        "prefix_knn_recall_at_k_by_depth": None,
        "cf_transition_recall_at_k": None,
        "cf_transition_recall_at_k_by_prefix": None,
        "weighted_hamming_spearman": None,
        "prefix_cf_coherence": None,
        "prefix_cf_coherence_baseline": None,
    }


def cf_alignment_metrics(
    grounding: SemanticIDGroundingIndex,
    cf_embeddings: torch.Tensor,
    cf_index: pl.DataFrame,
    *,
    k: int = 10,
    sample_size: int = 5000,
    seed: int = 42,
    pair_sample_size: int = 2_000_000,
    knn_chunk_size: int = 1024,
    stays: pl.DataFrame | None = None,
    transition_sample_size: int = 10_000,
) -> dict[str, Any]:
    grounding_frame = grounding.to_frame()
    code_cols = list(sid_code_columns(grounding_frame.columns))
    if not code_cols:
        return {**empty_cf_alignment_metrics(k)}

    join_col = (
        "placekey_id"
        if "placekey_id" in grounding_frame.columns
        and "placekey_id" in cf_index.columns
        else "placekey"
    )
    joined = grounding_frame.join(cf_index, on=join_col, how="inner")
    if "embedding_row" not in joined.columns:
        return {**empty_cf_alignment_metrics(k)}

    joined = joined.unique(subset=[join_col])
    n = joined.height
    if n < k + 1:
        result = empty_cf_alignment_metrics(k)
        result.update({"cf_sample_size": n, "cf_pair_sample_size": 0})
        return result

    rng = np.random.RandomState(seed)
    sample_n = min(sample_size, n)
    indices = rng.choice(n, size=sample_n, replace=False)
    indices.sort()
    rows = joined[indices.tolist()]
    embedding_rows = torch.as_tensor(
        rows["embedding_row"].to_numpy().copy(), dtype=torch.long
    )
    embeddings = cf_embeddings[embedding_rows].float()
    codes = np.column_stack(
        [rows[column].to_numpy().copy() for column in code_cols]
    )

    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    normalized_np = normalized.numpy()
    weights = np.array(
        [2.0 ** (len(code_cols) - 1 - i) for i in range(len(code_cols))],
        dtype=np.float64,
    )

    recalls_h1: list[float] = []
    recalls_h2: list[float] = []
    recalls_h3: list[float] = []
    recalls_prefix: list[float] = []
    recalls_prefix_by_depth: dict[int, list[float]] = {
        depth: [] for depth in range(1, len(code_cols) + 1)
    }
    for start in range(0, sample_n, knn_chunk_size):
        end = min(start + knn_chunk_size, sample_n)
        similarities = normalized[start:end] @ normalized.T
        row_indices = torch.arange(end - start)
        similarities[row_indices, torch.arange(start, end)] = -2.0
        neighbors = torch.topk(similarities, k=k, dim=1).indices.cpu().numpy()
        query_codes = codes[start:end, None, :]
        neighbor_codes = codes[neighbors]
        hamming = np.sum(query_codes != neighbor_codes, axis=2)
        prefix_depth = np.cumprod(query_codes == neighbor_codes, axis=2).sum(
            axis=2
        )
        recalls_h1.extend((hamming <= 1).mean(axis=1).astype(float).tolist())
        recalls_h2.extend((hamming <= 2).mean(axis=1).astype(float).tolist())
        recalls_h3.extend((hamming <= 3).mean(axis=1).astype(float).tolist())
        recalls_prefix.extend(
            (prefix_depth >= 1).mean(axis=1).astype(float).tolist()
        )
        for depth in range(1, len(code_cols) + 1):
            recalls_prefix_by_depth[depth].extend(
                (prefix_depth >= depth).mean(axis=1).astype(float).tolist()
            )

    possible_pairs = sample_n * (sample_n - 1) // 2
    n_pairs = min(pair_sample_size, possible_pairs)
    if n_pairs > 0:
        row_idx = rng.randint(0, sample_n, size=n_pairs)
        col_idx = rng.randint(0, sample_n, size=n_pairs)
        mask = row_idx != col_idx
        row_idx, col_idx = row_idx[mask], col_idx[mask]
    else:
        row_idx = np.array([], dtype=np.int64)
        col_idx = np.array([], dtype=np.int64)

    cosine = np.sum(normalized_np[row_idx] * normalized_np[col_idx], axis=1)
    mismatch = codes[row_idx] != codes[col_idx]
    hamming = mismatch.sum(axis=1).astype(np.float64)
    weighted_hamming = np.einsum(
        "ij,j->i", mismatch.astype(np.float64), weights
    )
    correlation_hamming = _spearman_or_none(cosine, -hamming)
    correlation_weighted = _spearman_or_none(cosine, -weighted_hamming)

    coherence, baseline = _prefix_cf_coherence(normalized_np, codes, rng)
    transition_identity = grounding_frame.join(
        cf_index,
        on=join_col,
        how="inner",
    ).unique(subset=[join_col])
    transition_recall, transition_recall_by_prefix = (
        _cf_transition_recall(
            stays,
            transition_identity,
            cf_embeddings,
            k=k,
            code_cols=code_cols,
            seed=seed,
            max_transitions=transition_sample_size,
            chunk_size=knn_chunk_size,
        )
        if stays is not None
        else (None, None)
    )
    return {
        "knn_recall_at_k": float(np.mean(recalls_h1)),
        "knn_k": k,
        "cosine_hamming_spearman": correlation_hamming,
        "cf_sample_size": sample_n,
        "cf_pair_sample_size": len(cosine),
        "knn_recall_at_k_h2": float(np.mean(recalls_h2)),
        "knn_recall_at_k_h3": float(np.mean(recalls_h3)),
        "prefix_knn_recall_at_k": float(np.mean(recalls_prefix)),
        "prefix_knn_recall_at_k_by_depth": {
            depth: float(np.mean(values))
            for depth, values in recalls_prefix_by_depth.items()
            if values
        },
        "cf_transition_recall_at_k": transition_recall,
        "cf_transition_recall_at_k_by_prefix": transition_recall_by_prefix,
        "weighted_hamming_spearman": correlation_weighted,
        "prefix_cf_coherence": coherence,
        "prefix_cf_coherence_baseline": baseline,
    }


def _spearman_or_none(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) <= 1 or len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
        return None
    correlation, _ = spearmanr(x, y)
    return float(correlation)


def _cf_transition_recall(
    stays: pl.DataFrame | None,
    embedding_joined: pl.DataFrame,
    cf_embeddings: torch.Tensor,
    *,
    k: int,
    code_cols: list[str],
    seed: int,
    max_transitions: int,
    chunk_size: int,
) -> tuple[float | None, dict[int, float] | None]:
    if stays is None or stays.is_empty() or "device_key" not in stays.columns:
        return None, None
    join_col = (
        "placekey_id"
        if "placekey_id" in stays.columns
        and "placekey_id" in embedding_joined.columns
        else "placekey"
    )
    if (
        join_col not in stays.columns
        or join_col not in embedding_joined.columns
    ):
        return None, None
    collapsed, _anchor_rate, _ = collapse_consecutive_pois(
        stays,
        poi_column=join_col,
    )
    next_id = pl.col(join_col).shift(-1).over("device_key")
    transitions = collapsed.with_columns(next_id.alias("_next_poi")).filter(
        pl.col("_next_poi").is_not_null()
    )
    if transitions.is_empty():
        return None, None
    rng = np.random.RandomState(seed)
    if transitions.height > max_transitions:
        selected = np.sort(
            rng.choice(transitions.height, size=max_transitions, replace=False)
        )
        transitions = transitions[selected.tolist()]

    identity = embedding_joined.select(
        [join_col, "embedding_row", *code_cols]
    ).unique(subset=[join_col])
    identity_rows = identity["embedding_row"].cast(pl.Int64).to_numpy()
    valid_identity = (identity_rows >= 0) & (
        identity_rows < int(cf_embeddings.shape[0])
    )
    identity = identity.filter(pl.Series(valid_identity))
    if identity.is_empty() or identity.height <= k:
        return None, None
    ids = identity[join_col].to_list()
    id_to_row = {value: index for index, value in enumerate(ids)}
    rows = identity["embedding_row"].cast(pl.Int64).to_numpy().copy()
    all_embeddings = torch.nn.functional.normalize(
        cf_embeddings[torch.as_tensor(rows)].float(), dim=1
    )
    source_indices: list[int] = []
    target_indices: list[int] = []
    for row in transitions.select([join_col, "_next_poi"]).iter_rows(
        named=True
    ):
        source = id_to_row.get(row[join_col])
        target = id_to_row.get(row["_next_poi"])
        if source is None or target is None or source == target:
            continue
        source_indices.append(source)
        target_indices.append(target)
    if not source_indices:
        return None, None

    hits: list[bool] = []
    target_array = np.asarray(target_indices, dtype=np.int64)
    source_array = np.asarray(source_indices, dtype=np.int64)
    try:
        import faiss

        all_embeddings_np = np.ascontiguousarray(all_embeddings.numpy())
        index: Any = faiss.IndexFlatIP(all_embeddings_np.shape[1])
        index.add(all_embeddings_np)
        for start in range(0, len(source_indices), chunk_size):
            end = min(start + chunk_size, len(source_indices))
            _scores, neighbors = index.search(
                all_embeddings_np[source_array[start:end]],
                min(k + 1, identity.height),
            )
            target_batch = target_array[start:end]
            hits.extend(
                (neighbors == target_batch[:, None]).any(axis=1).tolist()
            )
    except ImportError:
        target_tensor = torch.as_tensor(target_indices, dtype=torch.long)
        for start in range(0, len(source_indices), chunk_size):
            end = min(start + chunk_size, len(source_indices))
            source_tensor = torch.as_tensor(source_indices[start:end])
            similarities = all_embeddings[source_tensor] @ all_embeddings.T
            similarities[torch.arange(end - start), source_tensor] = -2.0
            neighbors = (
                torch.topk(similarities, k=min(k, identity.height - 1), dim=1)
                .indices.cpu()
                .numpy()
            )
            target_batch = target_tensor[start:end].numpy()
            hits.extend(
                (neighbors == target_batch[:, None]).any(axis=1).tolist()
            )
    return float(np.mean(hits)), None


def _prefix_cf_coherence(
    normalized: np.ndarray,
    codes: np.ndarray,
    rng: np.random.RandomState,
    max_groups: int = 200,
) -> tuple[float | None, float | None]:
    prefixes = np.unique(codes[:, 0])
    if len(prefixes) < 2:
        return None, None
    if len(prefixes) > max_groups:
        prefixes = rng.choice(prefixes, size=max_groups, replace=False)

    group_sims: list[float] = []
    for prefix in prefixes:
        members = np.where(codes[:, 0] == prefix)[0]
        if len(members) < 2:
            continue
        if len(members) > 100:
            members = rng.choice(members, size=100, replace=False)
        pair_sims = [
            float(np.dot(normalized[members[i]], normalized[members[j]]))
            for i in range(len(members))
            for j in range(i + 1, len(members))
        ]
        if pair_sims:
            group_sims.append(float(np.mean(pair_sims)))

    if not group_sims:
        return None, None
    coherence = float(np.mean(group_sims))
    n = len(codes)
    n_random_pairs = min(50000, n * (n - 1) // 2)
    row_idx = rng.randint(0, n, size=n_random_pairs)
    col_idx = rng.randint(0, n, size=n_random_pairs)
    mask = row_idx != col_idx
    row_idx, col_idx = row_idx[mask], col_idx[mask]
    baseline = (
        float(
            np.mean(np.sum(normalized[row_idx] * normalized[col_idx], axis=1))
        )
        if len(row_idx) > 0
        else None
    )
    return coherence, baseline


__all__ = [
    "cf_alignment_metrics",
    "empty_cf_alignment_metrics",
]
