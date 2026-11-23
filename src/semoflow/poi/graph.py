from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv

from semoflow.poi.sids import find_sorted
from semoflow.poi.text import load_poi_embeddings

logger = logging.getLogger(__name__)
_AGGREGATION_CHUNK = 500_000
_TIME_OF_DAY_RELATIONS = (
    "next_weekend",
    "next_morning",
    "next_afternoon",
    "next_evening",
)


def _key_index(series: pl.Series) -> pl.DataFrame:
    keys = series.unique().sort()
    return pl.DataFrame(
        {"key": keys, "idx": pl.arange(0, keys.len(), eager=True)}
    ).cast({"idx": pl.Int32})


def _join_idx(df: pl.DataFrame, col: str, lookup: pl.DataFrame) -> pl.Series:
    return df.select(pl.col(col).alias("key")).join(
        lookup, on="key", how="left", maintain_order="left"
    )["idx"]


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
    total = torch.zeros(num_nodes, features.shape[1])
    count = torch.zeros(num_nodes, 1)
    for start in range(0, len(node_index), _AGGREGATION_CHUNK):
        stop = min(start + _AGGREGATION_CHUNK, len(node_index))
        nodes = torch.as_tensor(node_index[start:stop])
        total.index_add_(0, nodes, features[feature_rows[start:stop]])
        count.index_add_(0, nodes, torch.ones(stop - start, 1))
    return total / count.clamp(min=1)


def _membership_edges(
    places: pl.DataFrame,
    poi_lookup: pl.DataFrame,
    column: str,
    lookup: pl.DataFrame,
) -> torch.Tensor:
    pairs = places.select(["placekey_id", column]).drop_nulls()
    poi = _join_idx(pairs, "placekey_id", poi_lookup).fill_null(-1).to_numpy()
    node = _join_idx(pairs, column, lookup).fill_null(-1).to_numpy()
    known = (poi != -1) & (node != -1)
    return torch.tensor(np.stack([poi[known], node[known]]), dtype=torch.long)


def _transition_edges(
    stays: pl.DataFrame,
    *,
    poi_lookup: pl.DataFrame,
    train_fraction: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    seq = (
        stays.select(["device_key", "placekey_id", "timestamp"])
        .drop_nulls()
        .sort(["device_key", "timestamp"])
        .with_columns(
            pl.col("placekey_id").shift(-1).over("device_key").alias("dst"),
            pl.col("timestamp").shift(-1).over("device_key").alias("dst_ts"),
        )
        .drop_nulls("dst")
        .filter(pl.col("placekey_id") != pl.col("dst"))
    )
    arrival = seq["dst_ts"].cast(pl.Datetime)
    hour = arrival.dt.hour().to_numpy()
    weekday = arrival.dt.weekday().to_numpy()
    src = _join_idx(seq, "placekey_id", poi_lookup).to_numpy()
    dst = _join_idx(seq, "dst", poi_lookup).to_numpy()

    n = len(src)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    train = np.zeros(n, dtype=bool)
    train[perm[: int(n * train_fraction)].numpy()] = True
    val_edge_index = torch.tensor(
        np.stack([src[~train], dst[~train]]), dtype=torch.long
    )
    logger.info(
        "temporal edges  transitions=%d  train=%d  val=%d",
        n,
        train.sum(),
        (~train).sum(),
    )
    src, dst, hour, weekday = (
        src[train],
        dst[train],
        hour[train],
        weekday[train],
    )
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
    return edges, val_edge_index


def build_hetero_data(
    stays: pl.DataFrame,
    places: pl.DataFrame,
    text_embeddings_path: Path,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[HeteroData, torch.Tensor]:
    started = time.perf_counter()
    poi_lookup = _key_index(places["placekey_id"])
    user_lookup = _key_index(stays["device_key"])
    category_lookup = _key_index(places["top_category"].drop_nulls())
    cbg_lookup = _key_index(places["census_block_group"].drop_nulls())
    logger.info(
        "building graph  stays=%d  pois=%d  users=%d  categories=%d  cbgs=%d",
        stays.height,
        poi_lookup.height,
        user_lookup.height,
        category_lookup.height,
        cbg_lookup.height,
    )

    data = HeteroData()
    data["poi"].city_id = torch.tensor(
        poi_lookup.rename({"key": "placekey_id"})
        .join(places.select("placekey_id", "city_id"), on="placekey_id")
        .sort("idx")["city_id"]
        .to_list(),
        dtype=torch.long,
    )
    poi_x = load_poi_embeddings(text_embeddings_path, places)
    data["poi"].x = poi_x

    stays = stays.join(
        poi_lookup.select(pl.col("key").alias("placekey_id")),
        on="placekey_id",
        how="semi",
        maintain_order="left",
    )
    users = _join_idx(stays, "device_key", user_lookup).to_numpy()
    pois = _join_idx(stays, "placekey_id", poi_lookup).to_numpy()
    visits = torch.tensor(np.stack([users, pois]), dtype=torch.long)
    _add_edges(data, ("user", "visits", "poi"), "rev_visits", visits)
    data["user"].x = _mean_features(user_lookup.height, users, pois, poi_x)

    for node, relation, reverse, column, lookup in (
        (
            "category",
            "has_category",
            "rev_category",
            "top_category",
            category_lookup,
        ),
        ("cbg", "located_in", "contains", "census_block_group", cbg_lookup),
    ):
        edges = _membership_edges(places, poi_lookup, column, lookup)
        _add_edges(data, ("poi", relation, node), reverse, edges)
        data[node].x = _mean_features(
            lookup.height, edges[1].numpy(), edges[0].numpy(), poi_x
        )

    transitions, val_edge_index = _transition_edges(
        stays, poi_lookup=poi_lookup, train_fraction=train_fraction, seed=seed
    )
    for relation, edge_index in transitions.items():
        _add_edges(
            data, ("poi", relation, "poi"), f"rev_{relation}", edge_index
        )
    logger.info(
        "graph built  %.1fs  visits=%d  metadata=%s",
        time.perf_counter() - started,
        visits.shape[1],
        data.metadata(),
    )
    return data, val_edge_index


def _temporal_edge_types(data: HeteroData) -> list[tuple[str, str, str]]:
    return [
        edge_type
        for edge_type in data.edge_types
        if edge_type[0] == "poi"
        and edge_type[2] == "poi"
        and edge_type[1].startswith("next")
    ]


def _negative_edges(
    data: HeteroData, positives: torch.Tensor, *, seed: int
) -> torch.Tensor:
    n_pois = int(data["poi"].num_nodes)
    temporal = torch.cat(
        [
            data[edge_type].edge_index
            for edge_type in _temporal_edge_types(data)
        ],
        dim=1,
    ).to(torch.int64)
    keep = temporal[0] != temporal[1]
    exclusion = torch.unique(temporal[0, keep] * n_pois + temporal[1, keep])
    category_edges = data["poi", "has_category", "category"].edge_index
    categories = torch.full((n_pois,), -1, dtype=torch.long)
    categories[category_edges[0]] = category_edges[1]

    n_negative = int(positives.size(1))
    generator = torch.Generator().manual_seed(seed)
    sources = positives[0].to(torch.int64)
    accepted = torch.empty(0, dtype=torch.int64)

    def accept(keys: torch.Tensor) -> None:
        nonlocal accepted
        keys = keys[~torch.isin(keys, exclusion) & ~torch.isin(keys, accepted)]
        need = n_negative - accepted.numel()
        if keys.numel() > need:
            keys = keys[
                torch.randperm(keys.numel(), generator=generator)[:need]
            ]
        accepted = torch.cat([accepted, keys])

    draw = min(max(4 * n_negative, 1 << 20), 1 << 24)
    attempts, max_attempts = 0, max(50 * n_negative, 100)
    while accepted.numel() < n_negative and attempts < max_attempts:
        take = min(draw, max_attempts - attempts)
        attempts += take
        src = sources[
            torch.randint(sources.numel(), (take,), generator=generator)
        ]
        dst = torch.randint(
            n_pois, (take,), generator=generator, dtype=torch.int64
        )
        src_category = categories[src]
        same = (src_category < 0) | (src_category == categories[dst])
        accept(torch.unique((src * n_pois + dst)[(src != dst) & same]))
    all_dst = torch.arange(n_pois, dtype=torch.int64)
    for source in sources.tolist():
        if accepted.numel() >= n_negative:
            break
        accept((source * n_pois + all_dst)[all_dst != source])
    return torch.stack([accepted // n_pois, accepted % n_pois])


def link_loader(
    data: HeteroData,
    *,
    train_edge_index: torch.Tensor | None,
    batch_size: int,
    num_neighbors: list[int],
    num_workers: int,
    pin_memory: bool,
    seed: int,
    rank: int,
    world_size: int,
) -> LinkNeighborLoader:
    edge_types = _temporal_edge_types(data)
    if train_edge_index is None:
        train_edge_index = torch.cat(
            [data[edge_type].edge_index for edge_type in edge_types], dim=1
        )
    positives = train_edge_index[:, rank::world_size].contiguous()
    negatives = _negative_edges(data, positives, seed=seed + rank)
    logger.info(
        "link loader  positives=%d  negatives=%d",
        positives.size(1),
        negatives.size(1),
    )
    return LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        edge_label_index=(
            edge_types[0],
            torch.cat([positives, negatives], dim=1),
        ),
        edge_label=torch.cat(
            [torch.ones(positives.size(1)), torch.zeros(negatives.size(1))]
        ),
        neg_sampling=None,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


class HeteroSAGE(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_dims: dict[str, int],
        *,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        aggr: str,
        layer_norm: bool,
        residual: bool,
        id_embedding_dim: int,
        id_embedding_counts: dict[str, int],
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.residual = residual
        self.id_embeddings = nn.ModuleDict()
        for node_type, count in id_embedding_counts.items():
            table = nn.Embedding(count, id_embedding_dim)
            nn.init.normal_(table.weight, std=0.01)
            self.id_embeddings[node_type] = table
        self.lin_dict = nn.ModuleDict(
            {
                node_type: nn.Linear(
                    in_dim
                    + (
                        id_embedding_dim
                        if node_type in self.id_embeddings
                        else 0
                    ),
                    hidden_dim,
                )
                for node_type, in_dim in in_dims.items()
            }
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            last = i == num_layers - 1
            channels = out_dim if last else hidden_dim
            self.convs.append(
                HeteroConv(
                    {
                        edge_type: SAGEConv((hidden_dim, hidden_dim), channels)
                        for edge_type in metadata[1]
                    },
                    aggr=aggr,
                )
            )
            self.norms.append(
                nn.LayerNorm(channels)
                if layer_norm and not last
                else nn.Identity()
            )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        n_id_dict: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        x_out = {
            node_type: self.lin_dict[node_type](
                torch.cat(
                    [x, self.id_embeddings[node_type](n_id_dict[node_type])],
                    dim=-1,
                )
                if n_id_dict is not None and node_type in self.id_embeddings
                else x
            )
            for node_type, x in x_dict.items()
        }
        for i, conv in enumerate(self.convs):
            previous, x_out = x_out, conv(x_out, edge_index_dict)
            if i == len(self.convs) - 1:
                break
            for node_type, x in x_out.items():
                x = self.norms[i](x)
                if self.residual:
                    x = x + previous[node_type]
                x_out[node_type] = F.dropout(
                    F.relu(x), p=self.dropout, training=self.training
                )
        return x_out


def variance_covariance_penalty(
    poi_emb: torch.Tensor, *, variance_target: float
) -> tuple[torch.Tensor, torch.Tensor]:
    n_rows, n_dims = poi_emb.shape
    centered = poi_emb - poi_emb.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0) + 1e-4)
    variance_loss = F.relu(variance_target - std).mean()
    covariance = (centered.T @ centered) / (n_rows - 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    return variance_loss, off_diagonal.pow(2).sum() / n_dims


def contrastive_link_loss(
    poi_emb: torch.Tensor,
    pos_edges: tuple[torch.Tensor, torch.Tensor],
    neg_edges: tuple[torch.Tensor, torch.Tensor],
    *,
    temperature: float,
    max_negatives: int,
    max_queries: int,
) -> torch.Tensor:
    source, target = pos_edges
    if source.numel() == 0:
        return poi_emb.sum() * 0.0
    if source.numel() > max_queries:
        keep = torch.randperm(source.numel(), device=poi_emb.device)
        source, target = source[keep[:max_queries]], target[keep[:max_queries]]

    query = F.normalize(poi_emb[source].float(), dim=-1)
    key = F.normalize(poi_emb[target].float(), dim=-1)
    n_queries = query.shape[0]
    positive_logits = (query * key).sum(dim=-1, keepdim=True) / temperature

    negative_index = torch.randint(
        n_queries, (min(max_negatives, n_queries),), device=query.device
    )
    negative_logits = (query @ key[negative_index].T) / temperature
    negative_targets = target[negative_index].expand(n_queries, -1).clone()
    negative_source, negative_target = neg_edges
    if negative_source.numel():
        order = torch.argsort(negative_source)
        positions, matched = find_sorted(negative_source[order], source)
        explicit = negative_target[order[positions[matched]]]
        explicit_logits = (
            query[matched] * F.normalize(poi_emb[explicit].float(), dim=-1)
        ).sum(dim=-1) / temperature
        negative_logits[matched, 0] = explicit_logits.to(negative_logits.dtype)
        negative_targets[matched, 0] = explicit
    negative_logits = negative_logits.masked_fill(
        negative_targets == target[:, None], float("-inf")
    )
    logits = torch.cat([positive_logits, negative_logits], dim=1)
    labels = torch.zeros(n_queries, dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, labels)


__all__ = [
    "HeteroSAGE",
    "build_hetero_data",
    "contrastive_link_loss",
    "link_loader",
    "variance_covariance_penalty",
]
