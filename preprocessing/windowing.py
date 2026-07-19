"""
Window generation.
"""

from __future__ import annotations

import numpy as np


class SlidingWindow:

    def __init__(
        self,
        window_size=2048,
        overlap=0.5,
    ):

        self.window_size = window_size
        self.overlap = overlap

    def __call__(self, signal):

        step = int(
            self.window_size *
            (1 - self.overlap)
        )

        windows = []

        for start in range(
            0,
            len(signal) - self.window_size + 1,
            step,
        ):

            windows.append(
                signal[
                    start:start+self.window_size
                ]
            )

        return np.asarray(windows)