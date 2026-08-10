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

    @property
    def last_path(self) -> Path:
        return self.output_dir / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.output_dir / "best.pt"

    def has_checkpoint(self) -> bool:
        """
        True if a previous run left a resumable checkpoint.
        """

        return self.last_path.exists()

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

        improved = self._is_better(metric)

        if improved:
            self.best_value = metric

        # `best_value` rides along in both files so a resumed run
        # knows the bar to beat and cannot overwrite a better
        # `best.pt` with a worse first epoch.
        state = {
            **state,
            "epoch": epoch,
            self.metric_name: metric,
            "best_value": self.best_value,
        }

        save_checkpoint(state, self.last_path)

        if improved:
            save_checkpoint(state, self.best_path)

        return improved

    def load_last(
        self,
        model=None,
        optimizer=None,
        device: str = "cpu",
    ) -> dict | None:
        """
        Load the most recent checkpoint for resuming, restoring the
        tracked best value so later epochs are compared against it.

        Parameters
        ----------
        model, optimizer : optional
            Loaded in place when given.

        Returns
        -------
        dict | None
            The full checkpoint state, or None if there is nothing to
            resume from.
        """

        if not self.has_checkpoint():
            return None

        state = load_checkpoint(self.last_path, device=device)

        if model is not None:
            model.load_state_dict(state["model_state"])

        if optimizer is not None and state.get("optimizer_state") is not None:
            optimizer.load_state_dict(state["optimizer_state"])

        self.best_value = state.get("best_value", state.get(self.metric_name))

        return state

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
