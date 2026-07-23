"""
Loss functions.

Provides a single `build_loss` factory whose returned callables all
share the signature `loss_fn(output, target) -> Tensor`, regardless
of whether `output` is a spiking-network dict (LIFClassifier) or a
plain logits tensor (dense baselines) — this is what lets
`training/trainer.py` stay model-agnostic.
"""

from __future__ import annotations

from typing import Callable

import snntorch.functional as snn_functional
import torch.nn as nn


def build_loss(name: str = "ce_count", **kwargs) -> Callable:
    """
    Parameters
    ----------
    name : str
        "ce_count" - cross-entropy on output spike counts
                     (snntorch.functional.ce_count_loss), for
                     spiking models whose forward() returns a
                     dict with a "spikes" key.

        "ce_dense" - standard cross-entropy for dense models
                     whose forward() returns raw logits.
    """

    name = name.lower().strip()

    if name == "ce_count":

        criterion = snn_functional.ce_count_loss(**kwargs)

        def loss_fn(output, target):
            return criterion(output["spikes"], target)

        return loss_fn

    if name == "ce_dense":

        criterion = nn.CrossEntropyLoss(**kwargs)

        def loss_fn(output, target):
            return criterion(output, target)

        return loss_fn

    raise KeyError(
        f"Unknown loss '{name}'. Available: ce_count, ce_dense"
    )
