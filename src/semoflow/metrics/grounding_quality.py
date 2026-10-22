from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GroundingQualityTally:
    n_special: int = 0
    n_groundable: int = 0
    n_valid: int = 0
    n_ambiguous: int = 0
    _valid_candidate_total: int = field(default=0, init=False, repr=False)

    def add_special(self, occurrences: int = 1) -> None:
        self._validate_occurrences(occurrences)
        self.n_special += occurrences

    def add_candidates(
        self, candidate_count: int, occurrences: int = 1
    ) -> None:
        self._validate_occurrences(occurrences)
        self.n_groundable += occurrences
        if candidate_count > 0:
            self.n_valid += occurrences
            self._valid_candidate_total += candidate_count * occurrences
        if candidate_count > 1:
            self.n_ambiguous += occurrences

    @staticmethod
    def _validate_occurrences(occurrences: int) -> None:
        if occurrences < 0:
            raise ValueError("occurrences must be non-negative")

    @property
    def n_positions(self) -> int:
        return self.n_special + self.n_groundable

    @property
    def valid_rate(self) -> float:
        return self.n_valid / self.n_positions if self.n_positions else 0.0

    @property
    def ambiguous_rate(self) -> float:
        return self.n_ambiguous / self.n_valid if self.n_valid else 0.0

    @property
    def mean_candidates_per_valid(self) -> float:
        return (
            self._valid_candidate_total / self.n_valid if self.n_valid else 0.0
        )


__all__ = ["GroundingQualityTally"]
