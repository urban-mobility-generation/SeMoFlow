"""The heterogeneous POI graph built from train-split visits, and the link-prediction supervision and neighbor loaders that train on it."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from semoflow.encoder.embeddings import load_poi_embeddings

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


_AGGREGATION_CHUNK = 500_000


_TIME_OF_DAY_RELATIONS = (
    "next_weekend",
    "next_morning",
    "next_afternoon",
    "next_evening",
)


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
    column: str,
    lookup: pl.DataFrame,
) -> torch.Tensor:
    """POI -> attribute-node edges for every POI with a known attribute."""
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


def build_hetero_data(
    stays_df: pl.DataFrame,
    places_index_df: pl.DataFrame,
    text_embeddings_path: Path,
    train_fraction: float | None = None,
    seed: int = 42,
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
    ts_col = "timestamp"

    poi_col = "placekey_id"
    poi_lookup = build_key_index(places_index_df[poi_col], "key")
    user_lookup = build_key_index(stays_df["device_key"], "key")
    cat_col, cbg_col = "top_category", "census_block_group"
    cat_lookup = build_key_index(places_index_df[cat_col].drop_nulls(), "key")
    cbg_lookup = build_key_index(places_index_df[cbg_col].drop_nulls(), "key")
    logger.info(
        "  nodes  pois=%d  users=%d  cats=%d  cbgs=%d",
        poi_lookup.height,
        user_lookup.height,
        cat_lookup.height,
        cbg_lookup.height,
    )

    data = HeteroData()
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
    data["poi"].city_id = torch.tensor(poi_city.to_list(), dtype=torch.long)

    poi_x = load_poi_embeddings(text_embeddings_path, places_index_df)
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
    logger.info(
        "HeteroData complete  %.1fs  metadata=%s",
        time.perf_counter() - t0,
        data.metadata(),
    )
    return data, val_edge_index


@dataclass(frozen=True)
class GraphSupervision:
    positive_edges: torch.Tensor
    negative_edges: torch.Tensor

    @property
    def edge_label_index(self) -> torch.Tensor:
        return torch.cat([self.positive_edges, self.negative_edges], dim=1)

    @property
    def edge_label(self) -> torch.Tensor:
        return torch.cat(
            [
                torch.ones(self.positive_edges.size(1)),
                torch.zeros(self.negative_edges.size(1)),
            ]
        )


def _edge_keys(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    src = edge_index[0].to(torch.int64)
    dst = edge_index[1].to(torch.int64)
    keep = src != dst
    return torch.unique(src[keep] * n_nodes + dst[keep])


def _category_ids(data: HeteroData) -> torch.Tensor | None:
    edge_type = ("poi", "has_category", "category")
    if edge_type not in data.edge_types:
        return None
    edge_index = data[edge_type].edge_index
    result = torch.full((int(data["poi"].num_nodes),), -1, dtype=torch.long)
    result[edge_index[0]] = edge_index[1]
    return result


def _in_sorted(keys: torch.Tensor, sorted_pool: torch.Tensor) -> torch.Tensor:
    if sorted_pool.numel() == 0:
        return torch.zeros(keys.shape, dtype=torch.bool)
    index = torch.searchsorted(sorted_pool, keys).clamp_(
        max=sorted_pool.numel() - 1
    )
    return sorted_pool[index] == keys


def _take_random(
    keys: torch.Tensor, need: int, generator: torch.Generator
) -> torch.Tensor:
    if need <= 0:
        return keys[:0]
    total = int(keys.numel())
    if total <= need:
        return keys
    choice = torch.randperm(total, generator=generator)[:need]
    return keys[choice]


def build_graph_supervision(
    data: HeteroData,
    positive_edges: torch.Tensor,
    *,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> GraphSupervision:
    all_temporal = [
        data[edge_type].edge_index
        for edge_type in data.metadata()[1]
        if edge_type[0] == "poi"
        and edge_type[2] == "poi"
        and not edge_type[1].startswith("rev_")
    ]
    n_pois = int(data["poi"].num_nodes)
    exclusion = _edge_keys(
        torch.cat(all_temporal, dim=1) if all_temporal else positive_edges,
        n_pois,
    )
    positives = positive_edges[:, rank::world_size].contiguous()
    n_negative = int(positives.size(1))
    if n_negative == 0:
        return GraphSupervision(
            positives, torch.empty((2, 0), dtype=torch.long)
        )

    categories = _category_ids(data)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + rank)
    src_pool = positives[0].to(torch.int64)

    accepted = torch.empty(0, dtype=torch.int64)
    accepted_sorted = torch.empty(0, dtype=torch.int64)

    def accept(keys: torch.Tensor) -> None:
        nonlocal accepted, accepted_sorted
        keys = keys[~_in_sorted(keys, exclusion)]
        keys = keys[~_in_sorted(keys, accepted_sorted)]
        taken = _take_random(
            keys, n_negative - int(accepted.numel()), generator
        )
        if taken.numel():
            accepted = torch.cat([accepted, taken])
            accepted_sorted = torch.sort(
                torch.cat([accepted_sorted, taken])
            ).values

    draw = min(max(4 * n_negative, 1 << 20), 1 << 24)
    attempts = 0
    max_attempts = max(50 * n_negative, 100)
    while int(accepted.numel()) < n_negative and attempts < max_attempts:
        take = min(draw, max_attempts - attempts)
        attempts += take
        idx = torch.randint(
            int(src_pool.numel()), (take,), generator=generator
        )
        src = src_pool[idx]
        dst = torch.randint(
            n_pois, (take,), generator=generator, dtype=torch.int64
        )
        mask = src != dst
        if categories is not None:
            src_cat = categories[src]
            mask &= (src_cat < 0) | (src_cat == categories[dst])
        accept(torch.unique(src[mask] * n_pois + dst[mask]))

    if int(accepted.numel()) < n_negative:
        all_dst = torch.arange(n_pois, dtype=torch.int64)
        for src_value in positives[0].tolist():
            if int(accepted.numel()) >= n_negative:
                break
            accept((src_value * n_pois + all_dst)[all_dst != src_value])
    if int(accepted.numel()) < n_negative:
        raise ValueError(
            "unable to construct enough exact graph negatives: "
            f"requested={n_negative}, available={int(accepted.numel())}"
        )
    negatives = torch.stack([accepted // n_pois, accepted % n_pois]).to(
        torch.long
    )
    return GraphSupervision(positives, negatives)


def resolve_num_neighbors(
    data: HeteroData,
    num_neighbors: list[int] | dict[str, list[int]],
) -> list[int] | dict[tuple[str, str, str], list[int]]:
    if not isinstance(num_neighbors, dict):
        return num_neighbors
    default = num_neighbors["default"]
    resolved: dict[tuple[str, str, str], list[int]] = {}
    for edge_type in data.metadata()[1]:
        resolved[edge_type] = num_neighbors.get("__".join(edge_type), default)
    unknown = {
        key
        for key in num_neighbors
        if key != "default"
        and tuple(key.split("__")) not in set(data.metadata()[1])
    }
    if unknown:
        raise ValueError(
            "num_neighbors names relations absent from the graph: "
            f"{sorted(unknown)}"
        )
    return resolved


def supervision_edge_types(
    data: HeteroData,
) -> list[tuple[str, str, str]]:
    return [
        edge_type
        for edge_type in data.metadata()[1]
        if edge_type[0] == "poi"
        and edge_type[2] == "poi"
        and not edge_type[1].startswith("rev_")
        and edge_type[1].startswith("next")
    ]


def get_train_loader(
    data: HeteroData,
    batch_size: int = 1024,
    num_neighbors: list[int] | dict[str, list[int]] | None = None,
    num_workers: int = 0,
    train_edge_index: torch.Tensor | None = None,
    pin_memory: bool = False,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> LinkNeighborLoader:
    fanout = resolve_num_neighbors(
        data, [10, 10] if num_neighbors is None else num_neighbors
    )

    target_edges = supervision_edge_types(data)

    if not target_edges:
        raise ValueError("No temporal POI-POI edges found for supervision.")

    supervision_edge_type = target_edges[0]
    if len(target_edges) > 1:
        logger.info(
            "merging %d temporal edge types under %s for supervision: %s",
            len(target_edges),
            supervision_edge_type,
            target_edges,
        )

    if train_edge_index is not None:
        pos_edge_index = train_edge_index
    else:
        pos_edge_index = torch.cat(
            [data[et].edge_index for et in target_edges], dim=1
        )
    started = time.perf_counter()
    supervision = build_graph_supervision(
        data,
        pos_edge_index,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    logger.info(
        "graph supervision built  positives=%d  %.1fs",
        int(supervision.positive_edges.size(1)),
        time.perf_counter() - started,
    )
    edge_label_index = supervision.edge_label_index
    edge_label = supervision.edge_label

    started = time.perf_counter()
    loader = LinkNeighborLoader(
        data,
        num_neighbors=fanout,
        edge_label_index=(supervision_edge_type, edge_label_index),
        edge_label=edge_label,
        neg_sampling=None,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    logger.info(
        "LinkNeighborLoader built  %.1fs",
        time.perf_counter() - started,
    )

    return loader


__all__ = []
