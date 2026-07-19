"""
Signal resampling utilities.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample


class Resampler:

    def __init__(
        self,
        original_fs: float,
        target_fs: float,
    ):

        self.original_fs = original_fs
        self.target_fs = target_fs

    def __call__(self, signal: np.ndarray):

        duration = len(signal) / self.original_fs

        target_samples = int(
            duration * self.target_fs
        )

        return resample(
            signal,
            target_samples,
        )
    