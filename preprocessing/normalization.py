"""
Normalization methods for vibration signals.
"""

from __future__ import annotations

import numpy as np


class ZScoreNormalize:
    """
    Z-score normalization.
    """

    def __call__(self, signal: np.ndarray) -> np.ndarray:

        signal = signal.astype(np.float32)

        mean = np.mean(signal)
        std = np.std(signal)

        return (signal - mean) / (std + 1e-8)


class MinMaxNormalize:
    """
    Min-Max normalization.
    """

    def __call__(self, signal: np.ndarray) -> np.ndarray:

        signal = signal.astype(np.float32)

        minimum = signal.min()
        maximum = signal.max()

        return (signal - minimum) / (maximum - minimum + 1e-8)


class RMSNormalize:
    """
    Normalize by RMS.
    """

    def __call__(self, signal: np.ndarray) -> np.ndarray:

        rms = np.sqrt(np.mean(signal ** 2))

        return signal / (rms + 1e-8)