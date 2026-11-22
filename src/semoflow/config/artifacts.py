from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from semoflow.utils.serialization import write_json


@dataclass(frozen=True)
class PreprocessingArtifacts:
    run_dir: Path

    @property
    def processed_visits(self) -> Path:
        return self.run_dir / "processed_visits.parquet"

    @property
    def places_index(self) -> Path:
        return self.run_dir / "places_index.parquet"


@dataclass(frozen=True)
class TokenizerArtifacts:
    run_dir: Path

    @property
    def augmented_processed_visits(self) -> Path:
        return self.run_dir / "augmented_processed_visits.parquet"

    @property
    def placekey_lookup(self) -> Path:
        return self.run_dir / "placekey_lookup.parquet"

    @property
    def tokenizer_model(self) -> Path:
        return self.run_dir / "tokenizer_model.pt"

    @property
    def tokenizer_codebooks(self) -> Path:
        return self.run_dir / "tokenizer_codebooks.pt"

    @property
    def grounding_index(self) -> Path:
        return self.run_dir / "grounding_index.pkl"

    @property
    def _completion_marker(self) -> Path:
        return self.run_dir / "completion.json"

    def _outputs(self) -> tuple[Path, ...]:
        return (
            self.augmented_processed_visits,
            self.placekey_lookup,
            self.tokenizer_model,
            self.tokenizer_codebooks,
            self.grounding_index,
        )

    def mark_complete(self) -> None:
        missing = [str(path) for path in self._outputs() if not path.exists()]
        if missing:
            raise RuntimeError(f"tokenizer outputs missing: {missing}")
        write_json(self._completion_marker, {"status": "complete"})

    def is_complete(self) -> bool:
        return self._completion_marker.exists() and all(
            path.exists() for path in self._outputs()
        )


@dataclass(frozen=True)
class RunLayout:
    """Where each stage of one experiment reads and writes, under ``root``."""

    root: Path

    @property
    def preprocessing(self) -> PreprocessingArtifacts:
        return PreprocessingArtifacts(self.root / "preprocessing")

    @property
    def encoder_dir(self) -> Path:
        return self.root / "encoder"

    @property
    def text_embeddings(self) -> Path:
        return self.encoder_dir / "text" / "text_embeddings.npy"

    @property
    def graph_dir(self) -> Path:
        return self.encoder_dir / "graph"

    @property
    def tokenizer(self) -> TokenizerArtifacts:
        return TokenizerArtifacts(self.root / "tokenization")

    @property
    def segments(self) -> Path:
        return self.root / "segments.parquet"

    @property
    def planner_dataset(self) -> Path:
        return self.root / "planner_dataset.pt"

    @property
    def planner_dir(self) -> Path:
        return self.root / "planner"

    @property
    def bank(self) -> Path:
        return self.root / "bank.pt"

    @property
    def realizer_dir(self) -> Path:
        return self.root / "realizer"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"


__all__ = ["PreprocessingArtifacts", "RunLayout", "TokenizerArtifacts"]
