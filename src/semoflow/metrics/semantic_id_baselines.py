from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl


def _factorize(values: list[Any]) -> np.ndarray:
    labels = sorted({str(value) for value in values})
    lookup = {label: index for index, label in enumerate(labels)}
    return np.asarray([lookup[str(value)] for value in values], dtype=np.int64)


def _encode_in_base(
    values: np.ndarray,
    *,
    n_codebooks: int,
    codebook_size: int,
) -> np.ndarray:
    result = np.zeros((len(values), n_codebooks), dtype=np.int32)
    remaining = np.asarray(values, dtype=np.int64).copy()
    for depth in range(n_codebooks - 1, -1, -1):
        result[:, depth] = (remaining % codebook_size).astype(np.int32)
        remaining //= codebook_size
    return result


def _suffix_balanced_codes(
    labels: np.ndarray,
    *,
    n_codebooks: int,
    codebook_size: int,
    seed: int,
) -> np.ndarray:
    result = np.zeros((len(labels), n_codebooks), dtype=np.int32)
    result[:, 0] = labels.astype(np.int32) % codebook_size
    if n_codebooks <= 1:
        return result
    rng = np.random.RandomState(seed)
    suffix = np.zeros(len(labels), dtype=np.int64)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        permutation = indices[rng.permutation(len(indices))]
        suffix[permutation] = np.arange(len(indices), dtype=np.int64)
    result[:, 1:] = _encode_in_base(
        suffix,
        n_codebooks=n_codebooks - 1,
        codebook_size=codebook_size,
    )
    return result


def _with_codes(
    frame: pl.DataFrame,
    codes: np.ndarray,
    code_cols: list[str],
) -> pl.DataFrame:
    existing = [column for column in code_cols if column in frame.columns]
    return frame.drop(existing).hstack(
        [
            pl.Series(column, codes[:, index])
            for index, column in enumerate(code_cols)
        ]
    )


def _align_embeddings(
    frame: pl.DataFrame,
    embeddings: np.ndarray,
    embedding_index: pl.DataFrame,
    *,
    id_column: str,
) -> np.ndarray | None:
    if id_column not in embedding_index.columns:
        return None
    if "embedding_row" not in embedding_index.columns:
        if embedding_index.height != len(embeddings):
            return None
        embedding_index = embedding_index.with_row_index("embedding_row")
    joined = frame.select(id_column).join(
        embedding_index.select([id_column, "embedding_row"]),
        on=id_column,
        how="left",
        maintain_order="left",
    )
    if joined["embedding_row"].null_count() > 0:
        return None
    rows = joined["embedding_row"].cast(pl.Int64).to_numpy().copy()
    if np.any((rows < 0) | (rows >= len(embeddings))):
        return None
    return np.asarray(embeddings)[rows]


def build_simple_baseline_lookups(
    catalog: pl.DataFrame,
    *,
    n_codebooks: int,
    codebook_size: int,
    seed: int = 20260729,
    text_embeddings: np.ndarray | None = None,
    graph_embeddings: np.ndarray | None = None,
    text_embedding_index: pl.DataFrame | None = None,
    graph_embedding_index: pl.DataFrame | None = None,
) -> dict[str, pl.DataFrame]:
    code_cols = [f"sem_code_{index}" for index in range(n_codebooks)]
    id_column = (
        "placekey_id" if "placekey_id" in catalog.columns else "placekey"
    )
    frame = catalog.unique(subset=[id_column], maintain_order=True)
    result: dict[str, pl.DataFrame] = {}

    ids = _factorize(frame[id_column].to_list())
    result["raw_poi_ids"] = _with_codes(
        frame,
        _encode_in_base(
            ids,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
        ),
        code_cols,
    )

    rng = np.random.RandomState(seed)
    result["random_balanced_codes"] = _with_codes(
        frame,
        _encode_in_base(
            rng.permutation(frame.height),
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
        ),
        code_cols,
    )

    for name, column in (
        ("category_ids", "top_category"),
        ("geohash_5_cells", "geohash_5"),
    ):
        if column in frame.columns:
            labels = _factorize(
                frame[column].fill_null("__missing__").to_list()
            )
            result[name] = _with_codes(
                frame,
                _suffix_balanced_codes(
                    labels,
                    n_codebooks=n_codebooks,
                    codebook_size=codebook_size,
                    seed=seed,
                ),
                code_cols,
            )

    for name, embeddings, embedding_index in (
        ("text_kmeans", text_embeddings, text_embedding_index),
        ("graph_kmeans", graph_embeddings, graph_embedding_index),
    ):
        if embeddings is None or embedding_index is None:
            continue
        aligned = _align_embeddings(
            frame, embeddings, embedding_index, id_column=id_column
        )
        if aligned is None or len(aligned) < 2:
            continue
        from sklearn.cluster import KMeans

        labels = KMeans(
            n_clusters=min(codebook_size, len(aligned)),
            n_init=10,
            random_state=seed,
        ).fit_predict(aligned)
        result[name] = _with_codes(
            frame,
            _suffix_balanced_codes(
                labels,
                n_codebooks=n_codebooks,
                codebook_size=codebook_size,
                seed=seed,
            ),
            code_cols,
        )
    return result


def select_baseline_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "prefix_category_purity",
        "prefix_category_macro_purity",
        "prefix_category_entropy_weighted",
        "prefix_category_ami",
        "prefix_category_nmi",
        "prefix_category_neighbor_recall",
        "prefix_category_purity_ci",
        "prefix_candidate_set_size_mean",
        "prefix_geo_radius_mean_km",
        "prefix_collision_rate",
        "transition_entropy_by_prefix",
        "prefix_temporal_coherence_by_depth",
        "prefix_temporal_coherence_baseline_by_depth",
        "prefix_temporal_coherence_effect_ci_by_depth",
        "prefix_knn_recall_at_k_by_depth",
        "cf_transition_recall_at_k",
        "next_prefix_prediction_accuracy",
        "next_poi_mutual_information",
        "anchor_repetition_rate",
    )
    return {field: metrics.get(field) for field in fields}


__all__ = [
    "build_simple_baseline_lookups",
    "select_baseline_metrics",
]
