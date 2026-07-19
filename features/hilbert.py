"""
Hilbert transform utilities.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


class HilbertTransform:
    """
    Hilbert transform utilities for vibration signals.

    Parameters
    ----------
    sampling_rate : float
        Signal sampling frequency (Hz).
    """

    def __init__(self, sampling_rate: float):
        if sampling_rate <= 0:
            raise ValueError(
                "sampling_rate must be positive."
            )

        self.fs = float(sampling_rate)

    def analytic_signal(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the analytic signal.

        Parameters
        ----------
        signal : np.ndarray

        Returns
        -------
        np.ndarray
            Complex analytic signal.
        """

        signal = np.asarray(signal, dtype=np.float32)

        if signal.ndim != 1:
            raise ValueError(
                "Signal must be one-dimensional."
            )

        return hilbert(signal)

    def envelope(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the signal envelope.
        """

        analytic = self.analytic_signal(signal)

        return np.abs(analytic)

    def instantaneous_phase(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Compute instantaneous phase.
        """

        analytic = self.analytic_signal(signal)

        return np.unwrap(np.angle(analytic))

    def instantaneous_frequency(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Compute instantaneous frequency.

        Returns
        -------
        np.ndarray
            Instantaneous frequency (Hz).
        """

        phase = self.instantaneous_phase(signal)

        frequency = (
            np.diff(phase)
            * self.fs
            / (2 * np.pi)
        )

        return frequency.astype(np.float32)

    def extract(
        self,
        signal: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Compute all Hilbert-derived quantities.
        """

        return {
            "analytic_signal": self.analytic_signal(signal),
            "envelope": self.envelope(signal),
            "instantaneous_phase": self.instantaneous_phase(signal),
            "instantaneous_frequency": self.instantaneous_frequency(signal),
        }