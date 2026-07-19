"""
Envelope analysis utilities.

Envelope analysis is widely used for rolling-element
bearing fault diagnosis.
"""

from __future__ import annotations

import numpy as np

from .fft import FFTFeatures
from .hilbert import HilbertTransform


class EnvelopeAnalysis:
    """
    Envelope spectrum analysis.

    Parameters
    ----------
    sampling_rate : float
        Sampling frequency (Hz).
    """

    def __init__(self, sampling_rate: float):

        self.fs = float(sampling_rate)

        self.hilbert = HilbertTransform(self.fs)
        self.fft = FFTFeatures(self.fs)

    def envelope(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Compute vibration envelope.
        """

        return self.hilbert.envelope(signal)

    def envelope_spectrum(
        self,
        signal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of the envelope signal.

        Returns
        -------
        frequencies
        magnitude
        """

        env = self.envelope(signal)

        return self.fft.spectrum(env)

    def dominant_frequency(
        self,
        signal: np.ndarray,
    ) -> float:
        """
        Dominant frequency in the envelope spectrum.
        """

        freq, mag = self.envelope_spectrum(signal)

        idx = np.argmax(mag)

        return float(freq[idx])

    def band_energy(
        self,
        signal: np.ndarray,
        low: float,
        high: float,
    ) -> float:
        """
        Envelope spectrum energy in a frequency band.
        """

        freq, mag = self.envelope_spectrum(signal)

        mask = (freq >= low) & (freq <= high)

        return float(np.sum(mag[mask] ** 2))

    def extract(
        self,
        signal: np.ndarray,
    ) -> dict[str, object]:
        """
        Extract envelope-based features.
        """

        freq, mag = self.envelope_spectrum(signal)

        return {
            "envelope": self.envelope(signal),
            "frequency": freq,
            "magnitude": mag,
            "dominant_frequency": self.dominant_frequency(signal),
        }