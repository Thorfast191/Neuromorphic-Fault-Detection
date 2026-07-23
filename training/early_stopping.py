"""
Early stopping on a monitored validation metric.
"""

from __future__ import annotations


class EarlyStopping:
    """
    Parameters
    ----------
    patience : int
        Number of non-improving epochs to tolerate before signaling
        a stop.

    mode : str
        "max" if higher metric values are better (e.g. accuracy),
        "min" if lower is better (e.g. loss).

    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 0.0,
    ):
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'.")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best: float | None = None
        self.best_epoch: int | None = None
        self.counter = 0

    def _is_improvement(self, value: float) -> bool:

        if self.best is None:
            return True

        if self.mode == "max":
            return value > self.best + self.min_delta

        return value < self.best - self.min_delta

    def step(self, value: float, epoch: int | None = None) -> bool:
        """
        Register a new metric value.

        Returns
        -------
        bool
            True if training should stop.
        """

        if self._is_improvement(value):
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
            return False

        self.counter += 1

        return self.counter >= self.patience

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, mode='{self.mode}', "
            f"best={self.best}, counter={self.counter})"
        )
