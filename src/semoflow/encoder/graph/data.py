from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch_geometric.data import HeteroData

from semoflow.encoder.graph.covisit import (
    covisit_adjacency,
    top_k_per_row,
)

logger = logging.getLogger(__name__)


def build_key_index(series: pl.Series, name: str = "key") -> pl.DataFrame:
    uniq = series.unique().sort()
    return pl.DataFrame(
        {name: uniq, "idx": pl.arange(0, uniq.len(), eager=True)}
    ).cast({"idx": pl.Int32})


def _join_idx(df: pl.DataFrame, col: str, lookup: pl.DataFrame) -> pl.Series:
    """Row index of each ``df[col]`` value in a ``(key, idx)`` lookup."""
    return df.select(pl.col(col).alias("key")).join(
        lookup, on="key", how="left", maintain_order="left"
    )["idx"]


def _poi_key_column(
    stays_df: pl.DataFrame, places_index_df: pl.DataFrame
) -> str:
    if (
        "placekey_id" in stays_df.columns
        and "placekey_id" in places_index_df.columns
    ):
        return "placekey_id"
    if (
        "placekey" in stays_df.columns
        and "placekey" in places_index_df.columns
    ):
        return "placekey"
    raise ValueError(
        "stays_df and places_index_df must share placekey_id or placekey"
    )


_CBG_NUMERIC_COLS = [
    "total_population",
    "median_household_income",
    "per_capita_income",
    "median_age",
    "pct_white",
    "pct_black",
    "pct_hispanic",
    "pct_college",
    "unemployment_rate",
    "median_gross_rent",
]


def _build_cbg_demographic_features(
    data: HeteroData,
    cbg_df: pl.DataFrame,
    cbg_lookup: pl.DataFrame,
    places_index_df: pl.DataFrame,
    cbg_col: str,
) -> bool:
    available = [c for c in _CBG_NUMERIC_COLS if c in cbg_df.columns]
    if not available:
        logger.warning("No numeric columns in cbg_attributes; skipping.")
        return False

    has_coords = (
        "latitude" in places_index_df.columns
        and "longitude" in places_index_df.columns
    )
    if has_coords:
        centroids = (
            places_index_df.select([cbg_col, "latitude", "longitude"])
            .drop_nulls()
            .group_by(cbg_col)
            .agg(
                pl.col("latitude").mean().alias("centroid_lat"),
                pl.col("longitude").mean().alias("centroid_lon"),
            )
        )
    else:
        centroids = None

    aligned = cbg_lookup.rename({"key": cbg_col}).join(
        cbg_df.rename({"cbg_geoid": cbg_col}),
        on=cbg_col,
        how="left",
    )
    if centroids is not None:
        aligned = aligned.join(centroids, on=cbg_col, how="left")
        feature_cols = [*available, "centroid_lat", "centroid_lon"]
    else:
        feature_cols = available

    aligned = aligned.sort("idx")
    arr = (
        aligned.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    )

    col_medians = np.nanmedian(arr, axis=0, keepdims=True)
    col_medians = np.nan_to_num(col_medians, nan=0.0)
    nan_mask = np.isnan(arr)
    arr = np.where(nan_mask, col_medians, arr)

    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    arr = (arr - mean) / std

    data["cbg"].x = torch.from_numpy(arr)
    logger.info(
        "  cbg features: %d dims (demographics + centroids)",
        arr.shape[1],
    )
    return True


_UNKNOWN_LOOKUP = pl.DataFrame({"key": ["Unknown"], "idx": [0]}).cast(
    {"idx": pl.Int32}
)
# Rows per index_add_ call when averaging POI features into user nodes.
_AGGREGATION_CHUNK = 500_000
_TIME_OF_DAY_RELATIONS = (
    "next_weekend",
    "next_morning",
    "next_afternoon",
    "next_evening",
)


def _attribute_lookup(
    places_index_df: pl.DataFrame, column: str | None
) -> pl.DataFrame:
    if column is None:
        return _UNKNOWN_LOOKUP
    return build_key_index(places_index_df[column].drop_nulls(), "key")


def _add_edges(
    data: HeteroData,
    relation: tuple[str, str, str],
    reverse: str,
    edge_index: torch.Tensor,
) -> None:
    source, name, target = relation
    data[source, name, target].edge_index = edge_index
    data[target, reverse, source].edge_index = edge_index.flip(0)


def _mean_features(
    num_nodes: int,
    node_index: np.ndarray,
    feature_rows: np.ndarray,
    features: torch.Tensor,
) -> torch.Tensor:
    """Average ``features[feature_rows]`` into ``node_index`` buckets."""
    total = torch.zeros(num_nodes, features.shape[1])
    count = torch.zeros(num_nodes, 1)
    for start in range(0, len(node_index), _AGGREGATION_CHUNK):
        stop = min(start + _AGGREGATION_CHUNK, len(node_index))
        nodes = torch.as_tensor(node_index[start:stop])
        total.index_add_(0, nodes, features[feature_rows[start:stop]])
        count.index_add_(0, nodes, torch.ones(stop - start, 1))
    return total / count.clamp(min=1)


def _membership_edges(
    places_index_df: pl.DataFrame,
    poi_col: str,
    poi_lookup: pl.DataFrame,
    column: str | None,
    lookup: pl.DataFrame,
) -> torch.Tensor:
    """POI -> attribute-node edges for every POI with a known attribute."""
    if column is None:
        return torch.empty((2, 0), dtype=torch.long)
    pairs = places_index_df.select([poi_col, column]).drop_nulls()
    poi = _join_idx(pairs, poi_col, poi_lookup).fill_null(-1).to_numpy()
    node = _join_idx(pairs, column, lookup).fill_null(-1).to_numpy()
    known = (poi != -1) & (node != -1)
    return torch.tensor(np.stack([poi[known], node[known]]), dtype=torch.long)


def _transition_edges(
    stays_valid: pl.DataFrame,
    *,
    poi_col: str,
    ts_col: str,
    poi_lookup: pl.DataFrame,
    train_fraction: float | None,
    seed: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, int]:
    """Consecutive-visit POI edges keyed by the arrival's time of day.

    With ``train_fraction``, a seeded share of the transitions is held out
    and returned as validation edges instead.
    """
    seq = (
        stays_valid.select(["device_key", poi_col, ts_col])
        .drop_nulls()
        .sort(["device_key", ts_col])
        .with_columns(
            pl.col(poi_col).shift(-1).over("device_key").alias("dst_pk"),
            pl.col(ts_col).shift(-1).over("device_key").alias("dst_ts"),
        )
        .drop_nulls("dst_pk")
        .filter(pl.col(poi_col) != pl.col("dst_pk"))
    )
    if seq.is_empty():
        return {}, None, 0
    arrival = pl.col("dst_ts").cast(pl.Datetime)
    seq = seq.with_columns(
        arrival.dt.hour().alias("dst_hour"),
        arrival.dt.weekday().alias("dst_weekday"),
    )
    src = _join_idx(seq, poi_col, poi_lookup).to_numpy()
    dst = _join_idx(seq, "dst_pk", poi_lookup).to_numpy()
    hour = seq["dst_hour"].to_numpy()
    weekday = seq["dst_weekday"].to_numpy()

    val_edge_index = None
    if train_fraction is not None:
        n = len(src)
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=generator).numpy()
        train_rows = np.zeros(n, dtype=bool)
        train_rows[perm[: int(n * train_fraction)]] = True
        val_rows = ~train_rows
        val_edge_index = torch.tensor(
            np.stack([src[val_rows], dst[val_rows]]), dtype=torch.long
        )
        logger.info(
            "  temporal edge split  train=%d  val=%d  fraction=%.2f  seed=%d",
            train_rows.sum(),
            val_rows.sum(),
            train_fraction,
            seed,
        )
        src, dst = src[train_rows], dst[train_rows]
        hour, weekday = hour[train_rows], weekday[train_rows]

    weekend = weekday >= 6
    masks = (
        weekend,
        ~weekend & (hour >= 5) & (hour < 12),
        ~weekend & (hour >= 12) & (hour < 17),
        ~weekend & ((hour >= 17) | (hour < 5)),
    )
    edges = {
        relation: torch.tensor(
            np.stack([src[mask], dst[mask]]), dtype=torch.long
        )
        for relation, mask in zip(_TIME_OF_DAY_RELATIONS, masks, strict=True)
        if mask.any()
    }
    return edges, val_edge_index, seq.height


def _add_covisit_edges(
    data: HeteroData,
    stays_df: pl.DataFrame,
    poi_lookup: pl.DataFrame,
    *,
    tc_hours: float,
    ts_intermediate: int,
    min_shift: int,
    min_covisits: int,
    top_k: int,
) -> None:
    start = time.perf_counter()
    adjacency = covisit_adjacency(
        stays_df,
        poi_lookup["key"].to_list(),
        tc_hours=tc_hours,
        ts_intermediate=ts_intermediate,
        min_covisits=min_covisits,
        min_shift=min_shift,
    )
    src, dst = top_k_per_row(adjacency, top_k)
    if src.size:
        _add_edges(
            data,
            ("poi", "covisits", "poi"),
            "rev_covisits",
            torch.tensor(np.stack([src, dst]), dtype=torch.long),
        )
    logger.info(
        "  co-visitation edges: %d  (tc=%.1fh min_shift=%d top_k=%d)  %.1fs",
        int(src.size),
        tc_hours,
        min_shift,
        top_k,
        time.perf_counter() - start,
    )


def build_hetero_data(
    stays_df: pl.DataFrame,
    places_index_df: pl.DataFrame,
    text_embeddings_path: str | Path | None = None,
    cbg_attributes_df: pl.DataFrame | None = None,
    hidden_dim: int = 64,
    train_fraction: float | None = None,
    seed: int = 42,
    covisit_edges: bool = False,
    covisit_tc_hours: float = 6.0,
    covisit_ts_intermediate: int = 1,
    covisit_min_shift: int = 2,
    covisit_min_covisits: int = 1,
    covisit_top_k: int = 50,
    include_user_nodes: bool = True,
) -> tuple[HeteroData, torch.Tensor | None]:
    """POI graph with user, category and CBG nodes plus transition edges.

    Returns the graph and, when ``train_fraction`` is set, the held-out
    transition edges for link-prediction validation.
    """
    t0 = time.perf_counter()
    logger.info(
        "Building HeteroData graph  stays=%d  pois=%d",
        stays_df.height,
        places_index_df.height,
    )
    ts_col = next(
        (c for c in ("local_ts", "timestamp") if c in stays_df.columns), None
    )
    if ts_col is None:
        raise ValueError("stays_df must have a local_ts or timestamp column")

    poi_col = _poi_key_column(stays_df, places_index_df)
    poi_lookup = build_key_index(places_index_df[poi_col], "key")
    user_lookup = (
        build_key_index(stays_df["device_key"], "key")
        if include_user_nodes
        else None
    )
    columns = places_index_df.columns
    cat_col = "top_category" if "top_category" in columns else None
    cbg_col = "census_block_group" if "census_block_group" in columns else None
    cat_lookup = _attribute_lookup(places_index_df, cat_col)
    cbg_lookup = _attribute_lookup(places_index_df, cbg_col)
    logger.info(
        "  nodes  pois=%d  users=%d  cats=%d  cbgs=%d",
        poi_lookup.height,
        0 if user_lookup is None else user_lookup.height,
        cat_lookup.height,
        cbg_lookup.height,
    )

    data = HeteroData()
    if "city_id" in columns:
        poi_city = (
            poi_lookup.rename({"key": poi_col})
            .join(
                places_index_df.select([poi_col, "city_id"]),
                on=poi_col,
                how="left",
            )
            .sort("idx")["city_id"]
        )
        if poi_city.null_count() > 0:
            raise ValueError("places_index city_id is missing for a POI")
        data["poi"].city_id = torch.tensor(
            poi_city.to_list(), dtype=torch.long
        )

    if text_embeddings_path and Path(text_embeddings_path).exists():
        from semoflow.encoder.artifacts import load_poi_embedding_artifact

        poi_emb = (
            load_poi_embedding_artifact(text_embeddings_path, kind="text")
            .align_to(places_index_df)
            .numpy()
        )
        poi_x = torch.from_numpy(poi_emb.astype(np.float32, copy=False))
    else:
        logger.warning("Text embeddings not found, using random features.")
        poi_x = torch.randn(poi_lookup.height, hidden_dim)
    data["poi"].x = poi_x

    poi_keys = poi_lookup.select(pl.col("key").alias(poi_col))
    stays_valid = stays_df.join(
        poi_keys, on=poi_col, how="semi", maintain_order="left"
    )
    logger.info(
        "  stays after POI filter: %d / %d",
        stays_valid.height,
        stays_df.height,
    )
    if user_lookup is not None:
        users = _join_idx(stays_valid, "device_key", user_lookup).to_numpy()
        pois = _join_idx(stays_valid, poi_col, poi_lookup).to_numpy()
        visits = torch.tensor(np.stack([users, pois]), dtype=torch.long)
        _add_edges(data, ("user", "visits", "poi"), "rev_visits", visits)
        logger.info("  visit edges: %d", visits.shape[1])
        data["user"].x = _mean_features(user_lookup.height, users, pois, poi_x)

    poi_cat_edges = _membership_edges(
        places_index_df, poi_col, poi_lookup, cat_col, cat_lookup
    )
    _add_edges(
        data,
        ("poi", "has_category", "category"),
        "rev_category",
        poi_cat_edges,
    )
    data["category"].x = _mean_features(
        cat_lookup.height,
        poi_cat_edges[1].numpy(),
        poi_cat_edges[0].numpy(),
        poi_x,
    )
    poi_cbg_edges = _membership_edges(
        places_index_df, poi_col, poi_lookup, cbg_col, cbg_lookup
    )
    _add_edges(data, ("poi", "located_in", "cbg"), "contains", poi_cbg_edges)
    if not (
        cbg_attributes_df is not None
        and cbg_col is not None
        and _build_cbg_demographic_features(
            data, cbg_attributes_df, cbg_lookup, places_index_df, cbg_col
        )
    ):
        data["cbg"].x = _mean_features(
            cbg_lookup.height,
            poi_cbg_edges[1].numpy(),
            poi_cbg_edges[0].numpy(),
            poi_x,
        )

    logger.info("  building temporal transition edges...")
    t_trans = time.perf_counter()
    transitions, val_edge_index, transition_count = _transition_edges(
        stays_valid,
        poi_col=poi_col,
        ts_col=ts_col,
        poi_lookup=poi_lookup,
        train_fraction=train_fraction,
        seed=seed,
    )
    for relation, edge_index in transitions.items():
        _add_edges(
            data, ("poi", relation, "poi"), f"rev_{relation}", edge_index
        )
    logger.info(
        "  temporal edges: %d transitions  %.1fs",
        transition_count,
        time.perf_counter() - t_trans,
    )
    if covisit_edges:
        _add_covisit_edges(
            data,
            stays_df,
            poi_lookup,
            tc_hours=covisit_tc_hours,
            ts_intermediate=covisit_ts_intermediate,
            min_shift=covisit_min_shift,
            min_covisits=covisit_min_covisits,
            top_k=covisit_top_k,
        )
    logger.info(
        "HeteroData complete  %.1fs  metadata=%s",
        time.perf_counter() - t0,
        data.metadata(),
    )
    return data, val_edge_index
