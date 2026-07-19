"""
Short-Time Fourier Transform (STFT) utilities.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft


class STFTFeatures:
    """
    Short-Time Fourier Transform feature extractor.

    Parameters
    ----------
    sampling_rate : float
        Sampling frequency in Hz.

    window : str
        Window type.

    nperseg : int
        Window length.

    noverlap : int
        Number of overlapping samples.
    """

    def __init__(
        self,
        sampling_rate: float,
        window: str = "hann",
        nperseg: int = 256,
        noverlap: int = 128,
    ):

        if sampling_rate <= 0:
            raise ValueError(
                "sampling_rate must be positive."
            )

        if nperseg <= 0:
            raise ValueError(
                "nperseg must be positive."
            )

        if noverlap >= nperseg:
            raise ValueError(
                "noverlap must be smaller than nperseg."
            )

        self.fs = float(sampling_rate)
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap

    def transform(
        self,
        signal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute STFT.

        Returns
        -------
        frequencies
        times
        magnitude
        """

        signal = np.asarray(signal, dtype=np.float32)

        if signal.ndim != 1:
            raise ValueError(
                "Signal must be one-dimensional."
            )

        f, t, z = stft(
            signal,
            fs=self.fs,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            boundary=None,
        )

        magnitude = np.abs(z)

        return f, t, magnitude

    def spectral_energy(
        self,
        signal: np.ndarray,
    ) -> float:
        """
        Total STFT energy.
        """

        _, _, magnitude = self.transform(signal)

        return float(np.sum(magnitude ** 2))

    def spectral_entropy(
        self,
        signal: np.ndarray,
    ) -> float:
        """
        STFT spectral entropy.
        """

        _, _, magnitude = self.transform(signal)

        power = magnitude ** 2
        power /= power.sum() + 1e-12

        entropy = -np.sum(
            power * np.log2(power + 1e-12)
        )

        entropy /= np.log2(power.size)

        return float(entropy)

    def extract(
        self,
        signal: np.ndarray,
    ) -> dict[str, object]:
        """
        Extract STFT features.
        """

        f, t, magnitude = self.transform(signal)

        return {
            "frequency": f,
            "time": t,
            "spectrogram": magnitude,
            "spectral_energy": self.spectral_energy(signal),
            "spectral_entropy": self.spectral_entropy(signal),
        }