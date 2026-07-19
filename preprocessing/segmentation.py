"""
Signal segmentation helpers.
"""

from __future__ import annotations

import numpy as np


def segment(
    signal: np.ndarray,
    start: int,
    end: int,
):

    return signal[start:end]


def split_signal(
    signal,
    parts,
):

    return np.array_split(signal, parts)