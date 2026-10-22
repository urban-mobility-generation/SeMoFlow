from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GraphEncoderConfig:
    enabled: bool = False
    device: str = "cpu"
    hidden_dim: int = 256
    out_dim: int = 256
    num_layers: int = 2
    epochs: int = 5
    batch_size: int = 1024
    lr: float = 0.001
    num_workers: int = 0
    mixed_precision: str = "bf16"
    num_neighbors: list[int] | dict[str, list[int]] = field(
        default_factory=lambda: [10, 10]
    )
    hash_train_fraction: float = 0.70
    seed: int = 42
    out_dir: str = "outputs/encoder/graph"
    resume_checkpoint: str = "outputs/encoder/graph/checkpoints/best.pt"
    warm_start_checkpoint: str | None = None
    cache_version: int = 1
    cache_payload: str = "graph_data.pt"
    cache_meta: str = "meta.json"
    temperature: float = 0.07
    max_negatives: int = 2048
    max_queries: int = 65536
    city_balanced_supervision: bool = False
    aggr: str = "sum"
    layer_norm: bool = False
    residual: bool = False
    dropout: float = 0.0
    id_embedding_dim: int = 0
    id_embedding_node_types: list[str] | None = None
    include_user_nodes: bool = True
    variance_weight: float = 0.0
    covariance_weight: float = 0.0
    variance_target: float = 1.0
    covisit_edges: bool = False
    covisit_tc_hours: float = 6.0
    covisit_ts_intermediate: int = 1
    dependency_min_covisits: int = 1
    covisit_min_shift: int = 2
    covisit_top_k: int = 50

    def validate(self) -> None:
        self._validate_num_neighbors()
        if self.temperature <= 0.0:
            raise ValueError(
                "encoder_pipeline.graph.temperature must be positive"
            )
        if self.max_negatives < 1 or self.max_queries < 1:
            raise ValueError(
                "encoder_pipeline.graph.max_negatives and max_queries must "
                "be positive"
            )
        if self.aggr not in {"sum", "mean", "max", "min"}:
            raise ValueError(
                "encoder_pipeline.graph.aggr must be one of "
                f"'sum', 'mean', 'max', 'min'; got {self.aggr!r}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "encoder_pipeline.graph.dropout must be in [0, 1)"
            )
        if self.id_embedding_dim < 0:
            raise ValueError(
                "encoder_pipeline.graph.id_embedding_dim must be non-negative"
            )
        if self.covisit_edges:
            if self.covisit_top_k < 1:
                raise ValueError(
                    "encoder_pipeline.graph.covisit_top_k must be positive"
                )
            if self.covisit_min_shift < 1:
                raise ValueError(
                    "encoder_pipeline.graph.covisit_min_shift must be >= 1"
                )
            if self.covisit_min_shift > self.covisit_ts_intermediate + 1:
                raise ValueError(
                    "encoder_pipeline.graph.covisit_min_shift exceeds the "
                    "largest offset covisit_ts_intermediate allows; raise "
                    "covisit_ts_intermediate or lower covisit_min_shift"
                )
        if self.id_embedding_node_types is not None:
            if not self.id_embedding_node_types:
                raise ValueError(
                    "encoder_pipeline.graph.id_embedding_node_types must be "
                    "null (every node type) or a non-empty list; pass "
                    "id_embedding_dim=0 to disable ID embeddings entirely"
                )
            if len(self.id_embedding_node_types) != len(
                set(self.id_embedding_node_types)
            ):
                raise ValueError(
                    "encoder_pipeline.graph.id_embedding_node_types must not "
                    "repeat a node type"
                )
        if self.variance_weight < 0.0 or self.covariance_weight < 0.0:
            raise ValueError(
                "encoder_pipeline.graph regularization weights must be "
                "non-negative"
            )

    def _validate_num_neighbors(self) -> None:
        value = self.num_neighbors
        if isinstance(value, dict):
            if "default" not in value:
                raise ValueError(
                    "encoder_pipeline.graph.num_neighbors dict must define a "
                    "'default' entry for unnamed relations"
                )
            hops = {len(v) for v in value.values()}
            if len(hops) != 1:
                raise ValueError(
                    "every encoder_pipeline.graph.num_neighbors entry must "
                    f"declare the same number of hops; got {sorted(hops)}"
                )
            entries: list[list[int]] = list(value.values())
        else:
            entries = [value]
        for entry in entries:
            if not entry:
                raise ValueError(
                    "encoder_pipeline.graph.num_neighbors entries must be "
                    "non-empty"
                )
            if any(int(n) < 1 for n in entry):
                raise ValueError(
                    "encoder_pipeline.graph.num_neighbors values must be "
                    "positive"
                )


@dataclass
class TextEncoderConfig:
    enabled: bool = False
    model: str = "sentence-t5-xxl"
    device: str = "cpu"
    batch_size: int = 512
    precision: str = "float16"
    backend: str = "torch"
    out_path: str = "outputs/encoder/text/text_embeddings.npy"
    include_address_in_template: bool = True
    include_name_in_template: bool = True

    def validate(self) -> None:
        if self.precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                "text.precision must be float32, float16, or bfloat16"
            )
        if self.backend not in {"torch", "onnx"}:
            raise ValueError("text.backend must be 'torch' or 'onnx'")
        if self.backend == "onnx" and self.precision != "float32":
            raise ValueError("ONNX text encoding requires float32 precision")
        if not isinstance(self.include_address_in_template, bool):
            raise ValueError(
                "text.include_address_in_template must be a bool; "
                f"got {self.include_address_in_template!r}"
            )
        if not isinstance(self.include_name_in_template, bool):
            raise ValueError(
                "text.include_name_in_template must be a bool; "
                f"got {self.include_name_in_template!r}"
            )


@dataclass
class EncoderPipelineConfig:
    graph: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
