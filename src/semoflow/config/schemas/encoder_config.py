from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GraphEncoderConfig:
    enabled: bool = True
    device: str = "cuda:0"
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
    temperature: float = 0.07
    max_negatives: int = 2048
    max_queries: int = 65536
    aggr: str = "sum"
    layer_norm: bool = False
    residual: bool = False
    dropout: float = 0.0
    id_embedding_dim: int = 0
    id_embedding_node_types: list[str] | None = None
    variance_weight: float = 0.0
    covariance_weight: float = 0.0
    variance_target: float = 1.0

    def validate(self) -> None:
        self._validate_num_neighbors()
        if self.temperature <= 0.0:
            raise ValueError("encoder.graph.temperature must be positive")
        if self.max_negatives < 1 or self.max_queries < 1:
            raise ValueError(
                "encoder.graph.max_negatives and max_queries must be positive"
            )
        if self.aggr not in {"sum", "mean", "max", "min"}:
            raise ValueError(
                "encoder.graph.aggr must be one of "
                f"'sum', 'mean', 'max', 'min'; got {self.aggr!r}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("encoder.graph.dropout must be in [0, 1)")
        if self.id_embedding_dim < 0:
            raise ValueError(
                "encoder.graph.id_embedding_dim must be non-negative"
            )
        if self.id_embedding_node_types is not None:
            if not self.id_embedding_node_types:
                raise ValueError(
                    "encoder.graph.id_embedding_node_types must be "
                    "null (every node type) or a non-empty list; pass "
                    "id_embedding_dim=0 to disable ID embeddings entirely"
                )
            if len(self.id_embedding_node_types) != len(
                set(self.id_embedding_node_types)
            ):
                raise ValueError(
                    "encoder.graph.id_embedding_node_types must not "
                    "repeat a node type"
                )
        if self.variance_weight < 0.0 or self.covariance_weight < 0.0:
            raise ValueError(
                "encoder.graph regularization weights must be non-negative"
            )

    def _validate_num_neighbors(self) -> None:
        value = self.num_neighbors
        if isinstance(value, dict):
            if "default" not in value:
                raise ValueError(
                    "encoder.graph.num_neighbors dict must define a "
                    "'default' entry for unnamed relations"
                )
            hops = {len(v) for v in value.values()}
            if len(hops) != 1:
                raise ValueError(
                    "every encoder.graph.num_neighbors entry must "
                    f"declare the same number of hops; got {sorted(hops)}"
                )
            entries: list[list[int]] = list(value.values())
        else:
            entries = [value]
        for entry in entries:
            if not entry:
                raise ValueError(
                    "encoder.graph.num_neighbors entries must be non-empty"
                )
            if any(int(n) < 1 for n in entry):
                raise ValueError(
                    "encoder.graph.num_neighbors values must be positive"
                )


@dataclass
class TextEncoderConfig:
    enabled: bool = True
    model: str = "sentence-t5-xxl"
    device: str = "cuda:0"
    batch_size: int = 512
    precision: str = "float16"
    include_address_in_template: bool = True
    include_name_in_template: bool = True

    def validate(self) -> None:
        if self.precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                "encoder.text.precision must be float32, float16, or bfloat16"
            )
        if not isinstance(self.include_address_in_template, bool):
            raise ValueError(
                "encoder.text.include_address_in_template must be a bool; "
                f"got {self.include_address_in_template!r}"
            )
        if not isinstance(self.include_name_in_template, bool):
            raise ValueError(
                "encoder.text.include_name_in_template must be a bool; "
                f"got {self.include_name_in_template!r}"
            )


@dataclass
class EncoderConfig:
    graph: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
