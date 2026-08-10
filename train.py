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

import math

import numpy as np
import torch
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
from utils.paths import resolve_dataset_root, resolve_results_dir
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
    Group-safe train/val/test split over `dataset`, cached to
    `<output_dir>/splits.json` so every later run (including
    evaluate.py and a baseline model trained on a differently-
    transformed copy of the same dataset) sees an identical held-out
    test set.

    Windows overlap within a recording (`dataset.overlap`), so a
    plain random/stratified split over window indices would put
    near-duplicate windows on both sides of the split - a model can
    then partly "test" on windows it has effectively already seen in
    training. Instead this splits chronologically *within* each
    source recording (`dataset.groups`, one id per source file) and
    drops the handful of windows straddling each cut, so no window in
    one split shares a raw sample with a window in another. Every
    CWRU file maps to exactly one class (see
    `datasets.labels.filename_to_label`), so splitting every file the
    same way keeps the per-class proportions close to
    `dataset.split` without needing separate stratification.
    """

    split_path = output_dir / "splits.json"

    if split_path.exists():
        split = load_json(split_path)
        return split["train"], split["val"], split["test"]

    split_cfg = cfg["dataset"]["split"]
    train_ratio = split_cfg["train"]
    val_ratio = split_cfg["val"]

    window_size = cfg["dataset"]["window_size"]
    step = int(window_size * (1 - cfg["dataset"]["overlap"]))

    # Windows within this many positions of each other share raw
    # samples; dropped at each cut so no split boundary leaks.
    buffer = max(0, math.ceil(window_size / step) - 1) if step > 0 else 0

    train_idx, val_idx, test_idx = [], [], []

    for group_id in np.unique(dataset.groups):

        group_indices = np.flatnonzero(dataset.groups == group_id)
        n = len(group_indices)

        cut1 = round(n * train_ratio)
        cut2 = cut1 + round(n * val_ratio)

        train_idx.extend(group_indices[: max(0, cut1 - buffer)].tolist())
        val_idx.extend(
            group_indices[cut1 + buffer : max(cut1 + buffer, cut2 - buffer)].tolist()
        )
        test_idx.extend(group_indices[cut2 + buffer :].tolist())

    save_json(
        {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
        split_path,
    )

    return train_idx, val_idx, test_idx


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Train the fault-detection model.")

    parser.add_argument("--config", default="configs/default.yaml")

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue from the last checkpoint if one exists. --epochs "
            "stays the total target, not extra epochs."
        ),
    )

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

    # Resolved rather than read straight from the config so a Colab
    # run can redirect results to mounted Drive and find the MAT
    # files wherever the clone put them. See utils/paths.py.
    results_dir = resolve_results_dir(cfg)
    output_dir = results_dir / cfg["model"]["architecture"]

    transform = build_transform(cfg)

    dataset = CWRUDataset(
        root=resolve_dataset_root(cfg),
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

    trainer.fit(
        train_loader,
        val_loader,
        epochs=cfg["training"]["epochs"],
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
