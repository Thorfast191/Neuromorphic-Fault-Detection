"""
Checkpoint orchestration.

Thin layer over `utils/checkpoint.py`'s save/load primitives that
tracks the best-so-far value of a monitored metric and always keeps
both a "last" and a "best" checkpoint on disk.
"""

from __future__ import annotations

from pathlib import Path

from utils.checkpoint import load_checkpoint, save_checkpoint


class CheckpointManager:
    """
    Parameters
    ----------
    output_dir : str | Path
        Directory to store `last.pt` and `best.pt`.

    metric_name : str
        Name of the metric tracked for "best" selection.

    mode : str
        "max" or "min".
    """

    def __init__(
        self,
        output_dir: str | Path,
        metric_name: str = "val_accuracy",
        mode: str = "max",
    ):
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'.")

        self.output_dir = Path(output_dir)
        self.metric_name = metric_name
        self.mode = mode
        self.best_value: float | None = None

    def _is_better(self, value: float) -> bool:

        if self.best_value is None:
            return True

        if self.mode == "max":
            return value > self.best_value

        return value < self.best_value

    def update(
        self,
        state: dict,
        epoch: int,
        metric: float,
    ) -> bool:
        """
        Save a "last" checkpoint, and a "best" checkpoint if
        `metric` improved.

        Returns
        -------
        bool
            True if this checkpoint became the new best.
        """

        state = {
            **state,
            "epoch": epoch,
            self.metric_name: metric,
        }

        save_checkpoint(state, self.output_dir / "last.pt")

        improved = self._is_better(metric)

        if improved:
            self.best_value = metric
            save_checkpoint(state, self.output_dir / "best.pt")

        return improved

    def load_best(self, model, device: str = "cpu") -> dict:
        """
        Load the best checkpoint's weights into `model`.

        Returns
        -------
        dict
            The full checkpoint state (including epoch/metric).
        """

        state = load_checkpoint(self.output_dir / "best.pt", device=device)

        model.load_state_dict(state["model_state"])

        return state
