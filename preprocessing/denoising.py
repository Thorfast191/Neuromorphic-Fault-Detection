"""
Signal denoising algorithms.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt
from scipy.signal import savgol_filter


class MedianFilter:

    def __init__(self, kernel_size: int = 5):
        self.kernel_size = kernel_size

    def __call__(self, signal: np.ndarray):

        return medfilt(
            signal,
            kernel_size=self.kernel_size,
        )


class SavitzkyGolayFilter:

    def __init__(
        self,
        window_length: int = 11,
        polyorder: int = 3,
    ):

        self.window_length = window_length
        self.polyorder = polyorder

    def __call__(self, signal: np.ndarray):

        return savgol_filter(
            signal,
            window_length=self.window_length,
            polyorder=self.polyorder,
        )


class MovingAverageFilter:

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def __call__(self, signal: np.ndarray):

        kernel = np.ones(self.window_size)

        kernel /= self.window_size

        return np.convolve(
            signal,
            kernel,
            mode="same",
        )