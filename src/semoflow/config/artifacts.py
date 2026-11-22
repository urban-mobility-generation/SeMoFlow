from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from semoflow.utils.serialization import write_json


@dataclass(frozen=True)
class TokenizerArtifacts:
    run_dir: Path
    augmented_processed_visits: Path
    placekey_lookup: Path
    tokenizer_model: Path
    tokenizer_codebooks: Path
    grounding_index: Path

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> TokenizerArtifacts:
        return cls(
            run_dir=run_dir,
            augmented_processed_visits=run_dir
            / "augmented_processed_visits.parquet",
            placekey_lookup=run_dir / "placekey_lookup.parquet",
            tokenizer_model=run_dir / "tokenizer_model.pt",
            tokenizer_codebooks=run_dir / "tokenizer_codebooks.pt",
            grounding_index=run_dir / "grounding_index.pkl",
        )

    @property
    def completion_marker(self) -> Path:
        return self.run_dir / "completion.json"

    def outputs(self) -> tuple[Path, ...]:
        return (
            self.augmented_processed_visits,
            self.placekey_lookup,
            self.tokenizer_model,
            self.tokenizer_codebooks,
            self.grounding_index,
        )

    def mark_complete(self) -> None:
        missing = [str(path) for path in self.outputs() if not path.exists()]
        if missing:
            raise RuntimeError(f"tokenizer outputs missing: {missing}")
        write_json(self.completion_marker, {"status": "complete"})

    def is_complete(self) -> bool:
        return self.completion_marker.exists() and all(
            path.exists() for path in self.outputs()
        )


@dataclass(frozen=True)
class PreprocessingArtifacts:
    run_dir: Path
    processed_visits: Path
    places_index: Path

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> PreprocessingArtifacts:
        return cls(
            run_dir=run_dir,
            processed_visits=run_dir / "processed_visits.parquet",
            places_index=run_dir / "places_index.parquet",
        )


@dataclass(frozen=True)
class RunLayout:
    """Where each stage of one experiment reads and writes, under ``root``."""

    root: Path

    @property
    def preprocessing(self) -> PreprocessingArtifacts:
        return PreprocessingArtifacts.from_run_dir(self.root / "preprocessing")

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
        return TokenizerArtifacts.from_run_dir(self.root / "tokenization")

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
