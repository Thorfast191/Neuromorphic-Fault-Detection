"""
Device utilities.
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """
    Returns the best available device.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")