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


def _build_index(series: pl.Series, name: str = "key") -> pl.DataFrame:
    uniq = series.unique().sort()
    return pl.DataFrame(
        {name: uniq, "idx": pl.arange(0, uniq.len(), eager=True)}
    ).cast({"idx": pl.Int32})


def _join_idx(
    df: pl.DataFrame,
    col: str,
    lookup: pl.DataFrame,
    key: str = "key",
) -> pl.Series:
    return df.select(pl.col(col).alias(key)).join(lookup, on=key, how="left")[
        "idx"
    ]


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
    t0 = time.perf_counter()
    logger.info(
        "Building HeteroData graph  stays=%d  pois=%d",
        stays_df.height,
        places_index_df.height,
    )

    ts_col = (
        "local_ts"
        if "local_ts" in stays_df.columns
        else ("timestamp" if "timestamp" in stays_df.columns else None)
    )
    if not ts_col:
        raise ValueError("stays_df must have a local_ts or timestamp column")

    poi_col = _poi_key_column(stays_df, places_index_df)
    poi_lookup = _build_index(places_index_df[poi_col], "key")
    num_pois = poi_lookup.height

    if include_user_nodes:
        user_lookup = _build_index(stays_df["device_key"], "key")
        num_users = user_lookup.height
    else:
        user_lookup = None
        num_users = 0

    cat_col = (
        "top_category" if "top_category" in places_index_df.columns else None
    )
    if cat_col:
        cat_lookup = _build_index(places_index_df[cat_col].drop_nulls(), "key")
    else:
        cat_lookup = pl.DataFrame({"key": ["Unknown"], "idx": [0]}).cast(
            {"idx": pl.Int32}
        )
    num_cats = cat_lookup.height

    cbg_col = (
        "census_block_group"
        if "census_block_group" in places_index_df.columns
        else None
    )
    if cbg_col:
        cbg_lookup = _build_index(places_index_df[cbg_col].drop_nulls(), "key")
    else:
        cbg_lookup = pl.DataFrame({"key": ["Unknown"], "idx": [0]}).cast(
            {"idx": pl.Int32}
        )
    num_cbgs = cbg_lookup.height

    logger.info(
        "  nodes  pois=%d  users=%d  cats=%d  cbgs=%d",
        num_pois,
        num_users,
        num_cats,
        num_cbgs,
    )

    data = HeteroData()

    if "city_id" in places_index_df.columns:
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
        from semoflow.encoder.artifacts import (
            load_poi_embedding_artifact,
        )

        poi_emb = (
            load_poi_embedding_artifact(
                text_embeddings_path,
                kind="text",
            )
            .align_to(places_index_df)
            .numpy()
        )
        data["poi"].x = torch.from_numpy(
            poi_emb.astype(np.float32, copy=False)
        )
        poi_dim = poi_emb.shape[1]
    else:
        logger.warning("Text embeddings not found, using random features.")
        poi_dim = hidden_dim
        data["poi"].x = torch.randn(num_pois, poi_dim)

    poi_x = data["poi"].x

    poi_keys = poi_lookup.select(pl.col("key").alias(poi_col))
    stays_valid = stays_df.join(poi_keys, on=poi_col, how="semi")
    logger.info(
        "  stays after POI filter: %d / %d",
        stays_valid.height,
        stays_df.height,
    )

    if include_user_nodes:
        assert user_lookup is not None
        u_idx = _join_idx(stays_valid, "device_key", user_lookup)
        p_idx = _join_idx(stays_valid, poi_col, poi_lookup)

        visits = torch.tensor(
            np.stack(
                [u_idx.to_numpy(), p_idx.to_numpy()],
            ),
            dtype=torch.long,
        )
        data["user", "visits", "poi"].edge_index = visits
        data["poi", "rev_visits", "user"].edge_index = visits.flip(0)
        logger.info("  visit edges: %d", visits.shape[1])

        user_agg = torch.zeros(num_users, poi_dim)
        user_cnt = torch.zeros(num_users, 1)
        u_np = np.array(u_idx.to_numpy())
        p_np = np.array(p_idx.to_numpy())
        _CHUNK = 500_000
        for _s in range(0, len(u_np), _CHUNK):
            _e = min(_s + _CHUNK, len(u_np))
            u_chunk = torch.from_numpy(u_np[_s:_e])
            user_agg.index_add_(0, u_chunk, poi_x[p_np[_s:_e]])
            user_cnt.index_add_(0, u_chunk, torch.ones(_e - _s, 1))
        data["user"].x = user_agg / user_cnt.clamp(min=1)

    if cat_col:
        poi_cat = places_index_df.select([poi_col, cat_col]).drop_nulls()
        pc_p_np = (
            _join_idx(poi_cat, poi_col, poi_lookup).fill_null(-1).to_numpy()
        )
        pc_c_np = (
            _join_idx(poi_cat, cat_col, cat_lookup).fill_null(-1).to_numpy()
        )
        m = (pc_p_np != -1) & (pc_c_np != -1)
        poi_cat_edges = torch.tensor(
            np.stack([pc_p_np[m], pc_c_np[m]]), dtype=torch.long
        )
    else:
        poi_cat_edges = torch.empty((2, 0), dtype=torch.long)

    data["poi", "has_category", "category"].edge_index = poi_cat_edges
    data["category", "rev_category", "poi"].edge_index = poi_cat_edges.flip(0)

    cat_agg = torch.zeros(num_cats, poi_dim)
    cat_cnt = torch.zeros(num_cats, 1)
    if poi_cat_edges.numel() > 0:
        cat_agg.index_add_(0, poi_cat_edges[1], poi_x[poi_cat_edges[0]])
        cat_cnt.index_add_(
            0,
            poi_cat_edges[1],
            torch.ones(poi_cat_edges.shape[1], 1),
        )
    data["category"].x = cat_agg / cat_cnt.clamp(min=1)

    if cbg_col:
        poi_cbg = places_index_df.select([poi_col, cbg_col]).drop_nulls()
        pcbg_p = _join_idx(poi_cbg, poi_col, poi_lookup)
        pcbg_c = _join_idx(poi_cbg, cbg_col, cbg_lookup)
        pcbg_p_np = pcbg_p.fill_null(-1).to_numpy()
        pcbg_c_np = pcbg_c.fill_null(-1).to_numpy()
        m = (pcbg_p_np != -1) & (pcbg_c_np != -1)
        poi_cbg_edges = torch.tensor(
            np.stack([pcbg_p_np[m], pcbg_c_np[m]]), dtype=torch.long
        )
    else:
        poi_cbg_edges = torch.empty((2, 0), dtype=torch.long)

    data["poi", "located_in", "cbg"].edge_index = poi_cbg_edges
    data["cbg", "contains", "poi"].edge_index = poi_cbg_edges.flip(0)

    cbg_features_set = False
    if cbg_attributes_df is not None and cbg_col:
        cbg_features_set = _build_cbg_demographic_features(
            data,
            cbg_attributes_df,
            cbg_lookup,
            places_index_df,
            cbg_col,
        )
    if not cbg_features_set:
        cbg_agg = torch.zeros(num_cbgs, poi_dim)
        cbg_cnt = torch.zeros(num_cbgs, 1)
        if poi_cbg_edges.numel() > 0:
            cbg_agg.index_add_(0, poi_cbg_edges[1], poi_x[poi_cbg_edges[0]])
            cbg_cnt.index_add_(
                0,
                poi_cbg_edges[1],
                torch.ones(poi_cbg_edges.shape[1], 1),
            )
        data["cbg"].x = cbg_agg / cbg_cnt.clamp(min=1)

    logger.info("  building temporal transition edges...")
    t_trans = time.perf_counter()
    val_edge_index = None

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

    if not seq.is_empty():
        seq = seq.with_columns(
            pl.col("dst_ts").cast(pl.Datetime).dt.hour().alias("dst_hour"),
            pl.col("dst_ts")
            .cast(pl.Datetime)
            .dt.weekday()
            .alias("dst_weekday"),
        )

        src_np = _join_idx(seq, poi_col, poi_lookup).to_numpy()
        dst_np = _join_idx(seq, "dst_pk", poi_lookup).to_numpy()
        hour_np = seq["dst_hour"].to_numpy()
        wd_np = seq["dst_weekday"].to_numpy()
        if train_fraction is not None:
            n = len(src_np)
            gen = torch.Generator().manual_seed(seed)
            perm = torch.randperm(n, generator=gen).numpy()
            n_train = int(n * train_fraction)
            train_rows = np.zeros(n, dtype=bool)
            train_rows[perm[:n_train]] = True
            val_rows = ~train_rows

            val_edge_index = torch.tensor(
                np.stack([src_np[val_rows], dst_np[val_rows]]),
                dtype=torch.long,
            )
            logger.info(
                "  temporal edge split  train=%d  val=%d  "
                "fraction=%.2f  seed=%d",
                train_rows.sum(),
                val_rows.sum(),
                train_fraction,
                seed,
            )

            src_np = src_np[train_rows]
            dst_np = dst_np[train_rows]
            hour_np = hour_np[train_rows]
            wd_np = wd_np[train_rows]

        weekend = wd_np >= 6
        morning = ~weekend & (hour_np >= 5) & (hour_np < 12)
        afternoon = ~weekend & (hour_np >= 12) & (hour_np < 17)
        evening = ~weekend & ((hour_np >= 17) | (hour_np < 5))
        for label, mask in [
            ("next_weekend", weekend),
            ("next_morning", morning),
            ("next_afternoon", afternoon),
            ("next_evening", evening),
        ]:
            if mask.any():
                edge_idx = torch.tensor(
                    np.stack([src_np[mask], dst_np[mask]]),
                    dtype=torch.long,
                )
                data["poi", label, "poi"].edge_index = edge_idx
                data["poi", f"rev_{label}", "poi"].edge_index = edge_idx.flip(
                    0
                )

    logger.info(
        "  temporal edges: %d transitions  %.1fs",
        seq.height,
        time.perf_counter() - t_trans,
    )

    if covisit_edges:
        t_cov = time.perf_counter()
        poi_ids = poi_lookup["key"].to_list()
        adjacency = covisit_adjacency(
            stays_df,
            poi_ids,
            tc_hours=covisit_tc_hours,
            ts_intermediate=covisit_ts_intermediate,
            min_covisits=covisit_min_covisits,
            min_shift=covisit_min_shift,
        )
        src_cov, dst_cov = top_k_per_row(adjacency, covisit_top_k)
        if src_cov.size:
            cov_idx = torch.tensor(
                np.stack([src_cov, dst_cov]), dtype=torch.long
            )
            data["poi", "covisits", "poi"].edge_index = cov_idx
            data["poi", "rev_covisits", "poi"].edge_index = cov_idx.flip(0)
        logger.info(
            "  co-visitation edges: %d  (tc=%.1fh min_shift=%d top_k=%d)  "
            "%.1fs",
            int(src_cov.size),
            covisit_tc_hours,
            covisit_min_shift,
            covisit_top_k,
            time.perf_counter() - t_cov,
        )

    logger.info(
        "HeteroData complete  %.1fs  metadata=%s",
        time.perf_counter() - t0,
        data.metadata(),
    )
    return data, val_edge_index
