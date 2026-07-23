"""
Model-agnostic training/evaluation loop.

Works for both the spiking LIFClassifier (forward() -> dict with a
"spikes" key) and dense baselines like CNN1DBaseline (forward() ->
plain logits tensor), via `get_logits`.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
from tqdm import tqdm

from utils.logger import get_logger
from utils.metrics import classification_metrics
from utils.timer import Timer


def get_logits(output: dict | torch.Tensor) -> torch.Tensor:
    """
    Extract (batch, num_classes) logits/spike-counts from a model's
    forward() output, regardless of whether it's a spiking-network
    dict or a plain dense tensor.
    """

    if isinstance(output, dict):
        return output["spikes"].sum(dim=0)

    return output


class Trainer:
    """
    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer
    loss_fn : Callable(output, target) -> Tensor
        See `training.loss.build_loss`.
    device : torch.device
    checkpoint_manager : training.checkpoint.CheckpointManager, optional
    early_stopping : training.early_stopping.EarlyStopping, optional
    output_dir : str | Path
    use_tensorboard : bool
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device,
        checkpoint_manager=None,
        early_stopping=None,
        output_dir: str | Path = "results",
        use_tensorboard: bool = True,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.early_stopping = early_stopping

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(
            "trainer",
            log_dir=str(self.output_dir / "logs"),
        )

        self.writer = None

        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(
                log_dir=str(self.output_dir / "tensorboard")
            )

        self._accepts_return_all = (
            "return_all" in inspect.signature(self.model.forward).parameters
        )

        self.history: list[dict] = []

    # ------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------

    def _forward(self, x: torch.Tensor, return_all: bool = False):

        if return_all and self._accepts_return_all:
            return self.model(x, return_all=True)

        return self.model(x)

    def _run_epoch(self, loader, train: bool) -> dict:

        self.model.train(train)

        total_loss = 0.0
        n_samples = 0
        y_true, y_pred = [], []

        with torch.set_grad_enabled(train):

            for x, y in tqdm(loader, leave=False, disable=not train):

                x = x.to(self.device)
                y = y.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                output = self._forward(x)
                loss = self.loss_fn(output, y)

                if train:
                    loss.backward()
                    self.optimizer.step()

                batch_size = y.size(0)
                total_loss += loss.item() * batch_size
                n_samples += batch_size

                logits = get_logits(output)
                y_true.extend(y.detach().cpu().tolist())
                y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

        metrics = classification_metrics(y_true, y_pred)
        metrics["loss"] = total_loss / max(n_samples, 1)

        return metrics

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def fit(self, train_loader, val_loader, epochs: int) -> list[dict]:
        """
        Run the training loop, checkpointing and early-stopping on
        validation accuracy.
        """

        for epoch in range(1, epochs + 1):

            with Timer() as timer:
                train_metrics = self._run_epoch(train_loader, train=True)
                val_metrics = self._run_epoch(val_loader, train=False)

            self.logger.info(
                "epoch %d/%d | train_loss=%.4f train_acc=%.4f | "
                "val_loss=%.4f val_acc=%.4f | %.1fs",
                epoch, epochs,
                train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"], val_metrics["accuracy"],
                timer.seconds,
            )

            if self.writer is not None:

                for key, value in train_metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, epoch)

                for key, value in val_metrics.items():
                    self.writer.add_scalar(f"val/{key}", value, epoch)

            self.history.append(
                {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            )

            if self.checkpoint_manager is not None:
                self.checkpoint_manager.update(
                    {
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                    },
                    epoch,
                    val_metrics["accuracy"],
                )

            if self.early_stopping is not None:

                if self.early_stopping.step(val_metrics["accuracy"], epoch=epoch):

                    self.logger.info(
                        "Early stopping at epoch %d (best=%.4f @ epoch %d)",
                        epoch,
                        self.early_stopping.best,
                        self.early_stopping.best_epoch,
                    )

                    break

        if self.writer is not None:
            self.writer.close()

        return self.history

    def evaluate(self, loader, return_all: bool = False) -> dict:
        """
        Run inference over `loader` without updating weights.

        Returns
        -------
        dict with keys:
            "y_true", "y_pred" : list[int]
            "logits"           : Tensor (N, num_classes) - spike
                                  counts (spiking model) or raw
                                  logits (dense model).
            "spike_records"    : list of per-batch forward() output
                                  dicts, only when return_all=True
                                  and the model supports it.
        """

        self.model.eval()

        y_true, y_pred = [], []
        logits_batches = []
        spike_records = [] if return_all else None

        with torch.no_grad():

            for x, y in loader:

                x = x.to(self.device)
                y = y.to(self.device)

                output = self._forward(x, return_all=return_all)

                logits = get_logits(output)

                y_true.extend(y.detach().cpu().tolist())
                y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
                logits_batches.append(logits.detach().cpu())

                if return_all and isinstance(output, dict):
                    spike_records.append(output)

        logits = (
            torch.cat(logits_batches, dim=0)
            if logits_batches
            else torch.empty(0)
        )

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "logits": logits,
            "spike_records": spike_records,
        }
