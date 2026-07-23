"""
Evaluation report generation: writes metrics/tables to disk and
renders the confusion-matrix and severity-accuracy figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this module only saves figures, never displays them

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.io import save_json


def generate_report(results: dict, output_dir: str | Path) -> None:
    """
    Parameters
    ----------
    results : dict
        "metrics"               : dict of scalar metrics
        "per_class"              : DataFrame (evaluation.metrics.per_class_breakdown)
        "severity_tier"          : DataFrame (evaluation.metrics.severity_tier_breakdown)
        "fault_severity_matrix"  : DataFrame (evaluation.metrics.fault_severity_matrix)
        "energy_comparison"      : DataFrame, optional
        "y_true", "y_pred"       : list[int]
        "class_names"            : list[str], ordered by label id
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(results["metrics"], output_dir / "metrics.json")

    results["per_class"].to_csv(
        output_dir / "per_class_breakdown.csv", index=False
    )
    results["severity_tier"].to_csv(
        output_dir / "severity_breakdown.csv", index=False
    )
    results["fault_severity_matrix"].to_csv(
        output_dir / "fault_severity_matrix.csv"
    )

    if "energy_comparison" in results:
        results["energy_comparison"].to_csv(
            output_dir / "energy_comparison.csv", index=False
        )

    _plot_confusion_matrix(
        results["y_true"],
        results["y_pred"],
        results["class_names"],
        output_dir / "confusion_matrix.png",
    )

    _plot_severity_accuracy(
        results["severity_tier"],
        output_dir / "severity_accuracy.png",
    )

    _print_summary(results)


def _plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    path: Path,
) -> None:

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_severity_accuracy(severity_df: pd.DataFrame, path: Path) -> None:

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(severity_df["severity_tier"], severity_df["accuracy"])

    ax.set_xlabel("Severity Tier")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Fault Severity (Incipient Detection)")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _print_summary(results: dict) -> None:

    print("\n=== Evaluation summary ===")

    for key, value in results["metrics"].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nSeverity-tier breakdown:")
    print(results["severity_tier"].to_string(index=False))

    if "energy_comparison" in results:
        print("\nEnergy/sparsity comparison:")
        print(results["energy_comparison"].to_string(index=False))
