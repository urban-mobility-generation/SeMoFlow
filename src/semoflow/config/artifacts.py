from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from semoflow.utils.serialization import write_json_atomic


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
    def config_yaml(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def completion_marker(self) -> Path:
        return self.run_dir / "completion.json"

    def is_complete(self) -> bool:
        if not self.completion_marker.exists():
            return False
        try:
            payload = json.loads(
                self.completion_marker.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return False
        if payload.get("status") != "complete":
            return False
        required = payload.get("required", {})
        if not isinstance(required, dict) or not required:
            return False
        known = self.export_paths()
        return all(
            name in known and known[name] == Path(path) and Path(path).exists()
            for name, path in required.items()
        )

    def write_completion(
        self,
        *,
        status: str,
        required: tuple[str, ...],
    ) -> None:
        payload = {
            "schema_version": 1,
            "status": status,
            "required": {
                name: str(self.export_paths()[name]) for name in required
            },
        }
        write_json_atomic(self.completion_marker, payload)

    def export_paths(self) -> dict[str, Path]:
        return {
            "augmented_processed_visits": self.augmented_processed_visits,
            "placekey_lookup": self.placekey_lookup,
            "tokenizer_model": self.tokenizer_model,
            "tokenizer_codebooks": self.tokenizer_codebooks,
            "grounding_index": self.grounding_index,
        }


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
