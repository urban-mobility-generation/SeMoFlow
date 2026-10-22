from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import torch

from semoflow.encoder.graph.data import build_hetero_data
from semoflow.encoder.graph.heterosage import HeteroSAGE
from semoflow.preprocess.places_index import (
    load_places_index,
)
from semoflow.training.graph_encoder import (
    _apply_graph_warm_start,
)

_EMPTY_STAYS = {
    "device_key": pl.Utf8,
    "placekey_id": pl.Int32,
    "timestamp": pl.Datetime("ns"),
}


def encode_catalog_only_graph(
    *,
    places_index: Path,
    text_embeddings: Path,
    cbg_attributes: Path,
    warm_start_checkpoint: Path,
    output_dir: Path,
    hidden_dim: int = 192,
    out_dim: int = 128,
    num_layers: int = 2,
    id_embedding_dim: int = 64,
    dropout: float = 0.0,
    aggr: str = "mean",
    device: str = "cpu",
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    emb_path = output_dir / "poi_graph_emb.pt"
    if emb_path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing artifact: {emb_path}"
        )

    torch.manual_seed(seed)
    places = load_places_index(places_index)
    data, _ = build_hetero_data(
        stays_df=pl.DataFrame(schema=_EMPTY_STAYS),
        places_index_df=places,
        text_embeddings_path=text_embeddings,
        cbg_attributes_df=pl.read_parquet(cbg_attributes),
        hidden_dim=hidden_dim,
        include_user_nodes=False,
    )
    temporal = [et for et in data.edge_types if et[0] == et[2] == "poi"]
    if temporal:
        raise ValueError(
            f"catalog-only graph must carry no POI-POI relation, got {temporal}"
        )

    id_counts = {
        nt: int(data[nt].num_nodes)
        for nt in ("poi", "category", "cbg")
        if nt in data.node_types
    }
    model = HeteroSAGE(
        metadata=data.metadata(),
        in_dims={nt: data[nt].x.size(1) for nt in data.node_types},
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=num_layers,
        dropout=dropout,
        aggr=aggr,
        id_embedding_dim=id_embedding_dim,
        id_embedding_counts=id_counts,
    )
    record = _apply_graph_warm_start(
        model, warm_start_checkpoint, device="cpu"
    )
    unexpected = [
        name
        for name in record["reinitialized_tensors"]
        if not name.startswith("id_embeddings.")
    ]
    if unexpected:
        raise ValueError(
            "these tensors did not transfer from the source and are not "
            f"ID tables: {unexpected}; the graph inputs do not match the "
            "source encoder"
        )
    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    zeroed = []
    with torch.no_grad():
        for name in record["reinitialized_tensors"]:
            tensor = tensors.get(name)
            if tensor is not None:
                tensor.zero_()
                zeroed.append(name)
    record["zeroed_untrained_tensors"] = zeroed

    target = torch.device(device)
    model = model.to(target).eval()
    with torch.no_grad():
        z = model(
            {k: v.to(target) for k, v in data.x_dict.items()},
            {k: v.to(target) for k, v in data.edge_index_dict.items()},
            {
                nt: torch.arange(data[nt].num_nodes, device=target)
                for nt in id_counts
            },
        )["poi"].cpu()

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(z, emb_path)
    places.write_parquet(output_dir / "places_index.parquet")
    report = {
        "artifact_kind": "catalog_only_graph_encode",
        "rung": "0%",
        "trained_epochs": 0,
        "reason_no_training": (
            "the graph objective is link prediction over temporal transition "
            "edges, and a catalog-only graph has none"
        ),
        "knn_relation_added": False,
        "untrained_id_tables_zeroed_not_random": True,
        "places_index": str(places_index),
        "text_embeddings": str(text_embeddings),
        "warm_start_checkpoint": str(warm_start_checkpoint),
        "node_counts": {nt: int(data[nt].num_nodes) for nt in data.node_types},
        "edge_types": [list(et) for et in data.edge_types],
        "embedding_shape": list(z.shape),
        "output": str(emb_path),
        **record,
    }
    (output_dir / "warm_start.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = ["encode_catalog_only_graph"]
