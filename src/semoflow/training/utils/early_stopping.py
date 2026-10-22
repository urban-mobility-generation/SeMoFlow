from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EarlyStopDecision:
    improved: bool
    should_stop: bool


class EarlyStopper:
    def __init__(
        self,
        *,
        patience: int | None,
        min_delta: float = 0.0,
        best_value: float = float("inf"),
        best_epoch: int = 0,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = best_value
        self.best_epoch = best_epoch
        self.epochs_without_improvement = 0

    def step(self, value: float, *, epoch: int) -> EarlyStopDecision:
        improved = value < (self.best_value - self.min_delta)
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        elif self.patience is not None:
            self.epochs_without_improvement += 1
        should_stop = (
            self.patience is not None
            and self.epochs_without_improvement >= self.patience
        )
        return EarlyStopDecision(improved=improved, should_stop=should_stop)


__all__ = ["EarlyStopDecision", "EarlyStopper"]
