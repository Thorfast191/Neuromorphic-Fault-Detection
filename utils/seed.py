"""
Utilities for reproducibility.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed across libraries.

    Args:
        seed: Random seed.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict:
    """
    Snapshot every RNG `set_seed` touches, so an interrupted run can
    resume with the same shuffle order it would have had.

    Returns:
        Picklable dict accepted by `restore_rng_state`.
    """

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state: dict | None) -> None:
    """
    Restore a snapshot taken by `capture_rng_state`.

    A CUDA snapshot taken on a machine with a different GPU count is
    skipped rather than raising, since a Colab session may come back
    on different hardware.
    """

    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))

    cuda_state = state.get("torch_cuda")

    if (
        cuda_state
        and torch.cuda.is_available()
        and len(cuda_state) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all(cuda_state)