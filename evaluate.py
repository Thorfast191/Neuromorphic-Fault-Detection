"""
Evaluation entry point.

Produces the severity-tier / incipient-fault-detection breakdown
for the trained event-driven LIFClassifier and, optionally, an
energy/sparsity comparison against a dense CNN1D baseline trained
on the same held-out split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from datasets.cwru import CWRUDataset
from datasets.labels import get_class_names
from evaluation.energy import count_esops, dense_macs, energy_estimate
from evaluation.metrics import (
    classification_metrics,
    confidence_margin,
    fault_severity_matrix,
    per_class_breakdown,
    severity_tier_breakdown,
    softmax_confidence,
)
from evaluation.report import generate_report
from models.factory import build_model
from train import build_transform, compute_or_load_split
from training.checkpoint import CheckpointManager
from training.early_stopping import EarlyStopping
from training.loss import build_loss
from training.optimizer import build_optimizer
from training.trainer import Trainer
from utils.config import Config
from utils.device import get_device
from utils.seed import set_seed


def _build_dataset(cfg: dict, transform):

    return CWRUDataset(
        root=cfg["dataset"]["root"],
        window_size=cfg["dataset"]["window_size"],
        overlap=cfg["dataset"]["overlap"],
        channel=cfg["dataset"]["channel"],
        transform=transform,
    )


def _evaluate_snn(cfg: dict, results_dir: Path, device) -> dict:

    output_dir = results_dir / cfg["model"]["architecture"]

    transform = build_transform(cfg)
    dataset = _build_dataset(cfg, transform)

    _, _, test_idx = compute_or_load_split(dataset, cfg, results_dir)

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    model = build_model(cfg).to(device)

    checkpoint_manager = CheckpointManager(
        output_dir / "checkpoints", metric_name="val_accuracy", mode="max"
    )
    checkpoint_manager.load_best(model, device=device)

    loss_fn = build_loss(cfg["training"]["loss"])
    optimizer = build_optimizer(
        cfg["training"]["optimizer"].lower(), model.parameters(), lr=cfg["training"]["lr"]
    )

    trainer = Trainer(
        model, optimizer, loss_fn, device, output_dir=output_dir, use_tensorboard=False
    )

    result = trainer.evaluate(test_loader, return_all=True)

    time_steps = cfg["dataset"]["window_size"] // cfg["encoding"]["bin_size"]
    margin = confidence_margin(result["logits"], time_steps)
    conf = softmax_confidence(result["logits"])

    metrics = classification_metrics(result["y_true"], result["y_pred"])

    # One extra forward pass with return_all=True to get spike
    # telemetry for the sparsity/ESOP comparison.
    sample_x, _ = next(iter(test_loader))
    sample_x = sample_x.to(device)

    model.eval()
    with torch.no_grad():
        sample_output = model(sample_x, return_all=True)

    return {
        "model": model,
        "y_true": result["y_true"],
        "y_pred": result["y_pred"],
        "margin": margin,
        "softmax_confidence": conf,
        "metrics": metrics,
        "sample_input": sample_x,
        "sample_output": sample_output,
    }


def _train_and_evaluate_baseline(cfg: dict, results_dir: Path, device) -> dict:

    baseline_cfg = {
        **cfg,
        "model": {**cfg["model"], "architecture": cfg["evaluation"]["baseline_architecture"]},
    }

    output_dir = results_dir / baseline_cfg["model"]["architecture"]

    transform = build_transform(baseline_cfg)
    dataset = _build_dataset(cfg, transform)

    train_idx, val_idx, test_idx = compute_or_load_split(dataset, cfg, results_dir)

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=cfg["training"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=cfg["training"]["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=cfg["training"]["batch_size"], shuffle=False
    )

    model = build_model(baseline_cfg).to(device)

    loss_fn = build_loss("ce_dense")
    optimizer = build_optimizer(
        cfg["training"]["optimizer"].lower(), model.parameters(), lr=cfg["training"]["lr"]
    )
    early_stopping = EarlyStopping(**cfg["training"]["early_stopping"])
    checkpoint_manager = CheckpointManager(
        output_dir / "checkpoints", metric_name="val_accuracy", mode="max"
    )

    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        device,
        checkpoint_manager=checkpoint_manager,
        early_stopping=early_stopping,
        output_dir=output_dir,
        use_tensorboard=False,
    )

    trainer.fit(train_loader, val_loader, epochs=cfg["training"]["epochs"])
    checkpoint_manager.load_best(model, device=device)

    result = trainer.evaluate(test_loader)
    metrics = classification_metrics(result["y_true"], result["y_pred"])

    sample_x, _ = next(iter(test_loader))
    sample_x = sample_x.to(device)

    return {
        "model": model,
        "metrics": metrics,
        "sample_input": sample_x,
    }


def _network_sparsity(input_spikes, hidden_spikes, output_spikes) -> float:
    """
    Fraction of inactive neuron-timesteps across the whole network
    (input events + hidden layers + output layer), as opposed to a
    single layer's sparsity.
    """

    tensors = [input_spikes, *hidden_spikes, output_spikes]

    total_elems = sum(t.numel() for t in tensors)
    total_active = sum(float(t.float().sum()) for t in tensors)

    return 1.0 - total_active / total_elems


def _energy_comparison_table(cfg: dict, snn_result: dict, baseline_result: dict) -> pd.DataFrame:

    macs = dense_macs(baseline_result["model"], baseline_result["sample_input"])

    esops_result = count_esops(
        snn_result["model"],
        snn_result["sample_input"],
        snn_result["sample_output"]["hidden_spikes"],
    )

    energy = energy_estimate(esops_result["total_esops"], macs)

    network_sparsity = _network_sparsity(
        snn_result["sample_input"],
        snn_result["sample_output"]["hidden_spikes"],
        snn_result["sample_output"]["spikes"],
    )

    return pd.DataFrame(
        [
            {
                "model": cfg["evaluation"]["baseline_architecture"],
                "dense_MACs_per_inference": macs,
                "SNN_ESOPs_per_inference": None,
                "sparsity_pct": 0.0,
                "energy_proxy_nJ": energy["mac_energy_nj"],
                "test_accuracy": baseline_result["metrics"]["accuracy"],
            },
            {
                "model": cfg["model"]["architecture"],
                "dense_MACs_per_inference": None,
                "SNN_ESOPs_per_inference": esops_result["total_esops"],
                "sparsity_pct": network_sparsity * 100,
                "energy_proxy_nJ": energy["esop_energy_nj"],
                "test_accuracy": snn_result["metrics"]["accuracy"],
            },
        ]
    )


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Evaluate the fault-detection model.")

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

    snn_result = _evaluate_snn(cfg, results_dir, device)

    class_names = get_class_names()

    report_payload = {
        "metrics": snn_result["metrics"],
        "per_class": per_class_breakdown(
            snn_result["y_true"],
            snn_result["y_pred"],
            snn_result["margin"],
            snn_result["softmax_confidence"],
        ),
        "severity_tier": severity_tier_breakdown(
            snn_result["y_true"],
            snn_result["y_pred"],
            snn_result["margin"],
            confidence_threshold=cfg["evaluation"]["confidence_threshold"],
        ),
        "fault_severity_matrix": fault_severity_matrix(
            snn_result["y_true"], snn_result["y_pred"], margin=snn_result["margin"]
        ),
        "y_true": snn_result["y_true"],
        "y_pred": snn_result["y_pred"],
        "class_names": class_names,
    }

    if cfg["evaluation"].get("compare_baseline", False):

        baseline_result = _train_and_evaluate_baseline(cfg, results_dir, device)

        report_payload["energy_comparison"] = _energy_comparison_table(
            cfg, snn_result, baseline_result
        )

    generate_report(report_payload, results_dir / "report")


if __name__ == "__main__":
    main()
