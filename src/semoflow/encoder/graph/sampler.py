from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

logger = logging.getLogger(__name__)


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


def balance_positive_edges_by_city(
    positive_edges: torch.Tensor,
    city_ids: torch.Tensor,
    *,
    seed: int = 0,
) -> torch.Tensor:
    if positive_edges.ndim != 2 or positive_edges.shape[0] != 2:
        raise ValueError("positive_edges must have shape [2, E]")
    if city_ids.ndim != 1:
        raise ValueError("city_ids must have shape [num_pois]")
    if positive_edges.numel() == 0:
        return positive_edges
    if int(positive_edges.max()) >= city_ids.numel():
        raise ValueError("positive_edges reference an unknown POI city")

    edge_cities = city_ids[positive_edges[0]]
    cities = torch.unique(edge_cities, sorted=True)
    if cities.numel() < 2:
        return positive_edges
    counts = torch.stack([(edge_cities == city).sum() for city in cities])
    per_city = int(counts.min().item())
    if per_city <= 0:
        return positive_edges[:, :0]

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    selected: list[torch.Tensor] = []
    for city in cities.tolist():
        rows = torch.nonzero(edge_cities == int(city), as_tuple=False).view(-1)
        if rows.numel() > per_city:
            rows = rows[
                torch.randperm(rows.numel(), generator=generator)[:per_city]
            ]
        selected.append(rows)
    return positive_edges[:, torch.cat(selected)]


def build_graph_supervision(
    data: HeteroData,
    positive_edges: torch.Tensor,
    *,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> GraphSupervision:
    if positive_edges.ndim != 2 or positive_edges.shape[0] != 2:
        raise ValueError("positive_edges must have shape [2, E]")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
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
    city_balanced: bool = False,
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
    if city_balanced:
        city_ids = getattr(data["poi"], "city_id", None)
        if city_ids is None:
            raise ValueError(
                "city_balanced supervision requires data['poi'].city_id"
            )
        pos_edge_index = balance_positive_edges_by_city(
            pos_edge_index,
            city_ids,
            seed=seed,
        )
        logger.info(
            "city-balanced temporal supervision positives=%d",
            int(pos_edge_index.size(1)),
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
