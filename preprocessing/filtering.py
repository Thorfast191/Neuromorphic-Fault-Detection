"""
Signal filtering utilities.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter
from scipy.signal import filtfilt


class ButterworthFilter:

    def __init__(
        self,
        fs: float,
        cutoff,
        order: int = 4,
        btype: str = "low",
    ):

        nyquist = fs / 2

        if isinstance(cutoff, (tuple, list)):
            cutoff = [c / nyquist for c in cutoff]
        else:
            cutoff = cutoff / nyquist

        self.b, self.a = butter(
            order,
            cutoff,
            btype=btype,
        )

    def __call__(self, signal: np.ndarray):

        return filtfilt(
            self.b,
            self.a,
            signal,
        )


class LowPassFilter(ButterworthFilter):

    def __init__(
        self,
        fs,
        cutoff,
        order=4,
    ):

        super().__init__(
            fs,
            cutoff,
            order,
            "low",
        )


class HighPassFilter(ButterworthFilter):

    def __init__(
        self,
        fs,
        cutoff,
        order=4,
    ):

        super().__init__(
            fs,
            cutoff,
            order,
            "high",
        )


class BandPassFilter(ButterworthFilter):

    def __init__(
        self,
        fs,
        low,
        high,
        order=4,
    ):

        super().__init__(
            fs,
            (low, high),
            order,
            "band",
        )