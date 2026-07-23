"""
Optimizer factory.
"""

from __future__ import annotations

from typing import Iterable

import torch.optim as optim

_OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
    "adamw": optim.AdamW,
}


def build_optimizer(
    name: str,
    params: Iterable,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs,
) -> optim.Optimizer:
    """
    Parameters
    ----------
    name : str
        One of "adam", "sgd", "rmsprop", "adamw".

    params : iterable
        Model parameters to optimize.
    """

    name = name.lower().strip()

    if name not in _OPTIMIZERS:
        available = ", ".join(sorted(_OPTIMIZERS))
        raise KeyError(f"Unknown optimizer '{name}'. Available: {available}")

    return _OPTIMIZERS[name](
        params,
        lr=lr,
        weight_decay=weight_decay,
        **kwargs,
    )
