"""
Classification metrics, including the severity-tier / incipient-
fault-detection breakdown that is the core evaluation protocol for
this thesis's novelty claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from datasets.labels import BearingLabel, ID_TO_NAME
from utils.metrics import classification_metrics

__all__ = [
    "classification_metrics",
    "SEVERITY_TIERS",
    "FAULT_TYPE_OF",
    "SEVERITY_OF",
    "confidence_margin",
    "softmax_confidence",
    "per_class_breakdown",
    "severity_tier_breakdown",
    "fault_severity_matrix",
]


# ----------------------------------------------------------------
# CWRU label groupings
#
# 007/014/021 = fault diameter in thousandths of an inch, i.e. an
# ordinal mild -> severe progression per fault type. Treated here
# as a proxy for incipient (007) vs. developed (021) faults.
# ----------------------------------------------------------------

SEVERITY_TIERS: dict[str, list[int]] = {
    "healthy": [int(BearingLabel.NORMAL)],
    "007": [
        int(BearingLabel.BALL_007),
        int(BearingLabel.INNER_007),
        int(BearingLabel.OUTER_007),
    ],
    "014": [
        int(BearingLabel.BALL_014),
        int(BearingLabel.INNER_014),
        int(BearingLabel.OUTER_014),
    ],
    "021": [
        int(BearingLabel.BALL_021),
        int(BearingLabel.INNER_021),
        int(BearingLabel.OUTER_021),
    ],
}

FAULT_TYPE_OF: dict[int, str] = {
    int(BearingLabel.BALL_007): "ball",
    int(BearingLabel.BALL_014): "ball",
    int(BearingLabel.BALL_021): "ball",
    int(BearingLabel.INNER_007): "inner",
    int(BearingLabel.INNER_014): "inner",
    int(BearingLabel.INNER_021): "inner",
    int(BearingLabel.OUTER_007): "outer",
    int(BearingLabel.OUTER_014): "outer",
    int(BearingLabel.OUTER_021): "outer",
}

SEVERITY_OF: dict[int, str] = {
    int(BearingLabel.BALL_007): "007",
    int(BearingLabel.INNER_007): "007",
    int(BearingLabel.OUTER_007): "007",
    int(BearingLabel.BALL_014): "014",
    int(BearingLabel.INNER_014): "014",
    int(BearingLabel.OUTER_014): "014",
    int(BearingLabel.BALL_021): "021",
    int(BearingLabel.INNER_021): "021",
    int(BearingLabel.OUTER_021): "021",
}


# ----------------------------------------------------------------
# Confidence signals, derived from output spike counts.
# ----------------------------------------------------------------

def confidence_margin(
    logits: torch.Tensor,
    time_steps: int,
) -> torch.Tensor:
    """
    Normalized margin between the top-1 and top-2 output spike
    counts, in [0, 1]. 0 = tied prediction (low confidence),
    1 = only one class fired at all (high confidence).
    """

    k = min(2, logits.shape[-1])

    top = torch.topk(logits, k=k, dim=-1).values

    if k < 2:
        margin = top[..., 0]
    else:
        margin = top[..., 0] - top[..., 1]

    return margin / max(time_steps, 1)


def softmax_confidence(logits: torch.Tensor) -> torch.Tensor:
    """
    Softmax-based confidence (max class probability), as a
    complementary confidence measure to `confidence_margin`.
    """

    return torch.softmax(logits, dim=-1).max(dim=-1).values


# ----------------------------------------------------------------
# Breakdown tables
# ----------------------------------------------------------------

def per_class_breakdown(
    y_true: list[int],
    y_pred: list[int],
    margin: torch.Tensor,
    softmax_conf: torch.Tensor,
) -> pd.DataFrame:
    """
    Per-class accuracy and mean confidence, one row per BearingLabel.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    margin = np.asarray(margin)
    softmax_conf = np.asarray(softmax_conf)

    rows = []

    for label in sorted(ID_TO_NAME):

        mask = y_true == label
        n = int(mask.sum())

        rows.append(
            {
                "label_id": label,
                "class_name": ID_TO_NAME[label],
                "fault_type": FAULT_TYPE_OF.get(label, "healthy"),
                "severity": SEVERITY_OF.get(label, "healthy"),
                "n_samples": n,
                "accuracy": float((y_pred[mask] == y_true[mask]).mean())
                if n else float("nan"),
                "mean_margin_confidence": float(margin[mask].mean())
                if n else float("nan"),
                "mean_softmax_confidence": float(softmax_conf[mask].mean())
                if n else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def severity_tier_breakdown(
    y_true: list[int],
    y_pred: list[int],
    margin: torch.Tensor,
    confidence_threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Headline "incipient detection" table: accuracy/confidence
    rolled up by severity tier (healthy / 007 / 014 / 021).
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    margin = np.asarray(margin)

    rows = []

    for tier, labels in SEVERITY_TIERS.items():

        mask = np.isin(y_true, labels)
        n = int(mask.sum())

        tier_margin = margin[mask]

        rows.append(
            {
                "severity_tier": tier,
                "n_samples": n,
                "accuracy": float((y_pred[mask] == y_true[mask]).mean())
                if n else float("nan"),
                "mean_margin_confidence": float(tier_margin.mean())
                if n else float("nan"),
                "min_margin_confidence": float(tier_margin.min())
                if n else float("nan"),
                "pct_below_confidence_threshold": float(
                    (tier_margin < confidence_threshold).mean() * 100
                )
                if n else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def fault_severity_matrix(
    y_true: list[int],
    y_pred: list[int],
    metric: str = "accuracy",
    margin: torch.Tensor | None = None,
) -> pd.DataFrame:
    """
    3x3 fault-type x severity pivot table.

    Parameters
    ----------
    metric : str
        "accuracy" or "confidence" (requires `margin`).
    """

    per_class = per_class_breakdown(
        y_true,
        y_pred,
        margin if margin is not None else torch.zeros(len(y_true)),
        torch.zeros(len(y_true)),
    )

    faults = per_class[per_class["fault_type"] != "healthy"]

    value_col = "accuracy" if metric == "accuracy" else "mean_margin_confidence"

    pivot = faults.pivot(
        index="fault_type",
        columns="severity",
        values=value_col,
    )

    return pivot.reindex(
        index=["ball", "inner", "outer"],
        columns=["007", "014", "021"],
    )
