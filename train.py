"""
Training entry point.

Builds the CWRU dataset with either an event-driven (delta-encoded)
spike transform or a dense normalization transform depending on
`model.architecture`, trains the selected model, and checkpoints it
under `results/<architecture>/checkpoints`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from datasets.cwru import CWRUDataset
from encoding.factory import create_encoder
from models.factory import build_model
from preprocessing.normalization import ZScoreNormalize
from training.checkpoint import CheckpointManager
from training.early_stopping import EarlyStopping
from training.loss import build_loss
from training.optimizer import build_optimizer
from training.trainer import Trainer
from utils.config import Config
from utils.device import get_device
from utils.io import load_json, save_json
from utils.seed import set_seed


def build_spike_transform(cfg: dict):
    """
    Delta-encode a raw vibration window into a (time_steps, 2)
    ON/OFF spike-count tensor, binned over time so BPTT length is
    configurable independently of `dataset.window_size`.
    """

    encoding_cfg = cfg["encoding"]
    bin_size = int(encoding_cfg.get("bin_size", 1))

    encoder = create_encoder(
        encoding_cfg["type"],
        threshold=encoding_cfg.get("threshold", 0.1),
    )

    def transform(window: np.ndarray) -> torch.Tensor:

        spikes = encoder.encode(window)  # (2, window_size)

        n_channels, length = spikes.shape
        n_bins = length // bin_size

        trimmed = spikes[:, : n_bins * bin_size]
        binned = trimmed.reshape(n_channels, n_bins, bin_size).sum(axis=2)

        return torch.from_numpy(binned.T.astype(np.float32))  # (T, 2)

    return transform


def build_dense_transform():
    """
    Z-score normalize a raw vibration window for the dense CNN1D
    baseline, shaped as (1, window_size).
    """

    normalize = ZScoreNormalize()

    def transform(window: np.ndarray) -> torch.Tensor:

        normalized = normalize(window)

        return torch.from_numpy(normalized).unsqueeze(0)  # (1, window_size)

    return transform


def build_transform(cfg: dict):
    """
    Select the input transform matching `cfg["model"]["architecture"]`.
    """

    if cfg["model"]["architecture"] == "cnn1d":
        return build_dense_transform()

    return build_spike_transform(cfg)


def compute_or_load_split(
    dataset: CWRUDataset,
    cfg: dict,
    output_dir: Path,
) -> tuple[list[int], list[int], list[int]]:
    """
    Stratified train/val/test split over `dataset`, cached to
    `<output_dir>/splits.json` so every later run (including
    evaluate.py and a baseline model trained on a differently-
    transformed copy of the same dataset) sees an identical held-out
    test set.
    """

    split_path = output_dir / "splits.json"

    if split_path.exists():
        split = load_json(split_path)
        return split["train"], split["val"], split["test"]

    indices = np.arange(len(dataset))
    labels = dataset.labels

    split_cfg = cfg["dataset"]["split"]
    test_size = split_cfg["test"]
    val_size = split_cfg["val"]

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=cfg["seed"],
    )

    val_ratio = val_size / (1.0 - test_size)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        stratify=labels[train_val_idx],
        random_state=cfg["seed"],
    )

    save_json(
        {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
        },
        split_path,
    )

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Train the fault-detection model.")

    parser.add_argument("--config", default="configs/default.yaml")

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    cfg = Config(args.config).data

    set_seed(cfg["seed"])

    device = (
        get_device()
        if cfg.get("device", "auto") == "auto"
        else torch.device(cfg["device"])
    )

    results_dir = Path(cfg["results"]["output_dir"])
    output_dir = results_dir / cfg["model"]["architecture"]

    transform = build_transform(cfg)

    dataset = CWRUDataset(
        root=cfg["dataset"]["root"],
        window_size=cfg["dataset"]["window_size"],
        overlap=cfg["dataset"]["overlap"],
        channel=cfg["dataset"]["channel"],
        transform=transform,
    )

    print(dataset.summary())

    # Split indices are cached at the shared results root (not the
    # per-architecture output_dir) so a baseline model trained later
    # on a differently-transformed copy of this dataset sees the
    # identical held-out test set.
    train_idx, val_idx, test_idx = compute_or_load_split(dataset, cfg, results_dir)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    model = build_model(cfg)

    loss_fn = build_loss(cfg["training"]["loss"])

    optimizer = build_optimizer(
        cfg["training"]["optimizer"].lower(),
        model.parameters(),
        lr=cfg["training"]["lr"],
    )

    early_stopping = EarlyStopping(**cfg["training"]["early_stopping"])

    checkpoint_manager = CheckpointManager(
        output_dir / "checkpoints",
        metric_name="val_accuracy",
        mode="max",
    )

    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        device,
        checkpoint_manager=checkpoint_manager,
        early_stopping=early_stopping,
        output_dir=output_dir,
    )

    trainer.fit(train_loader, val_loader, epochs=cfg["training"]["epochs"])


if __name__ == "__main__":
    main()
