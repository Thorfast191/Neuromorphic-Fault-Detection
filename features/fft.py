"""
FFT-based feature extraction utilities.
"""

from __future__ import annotations

import numpy as np


class FFTFeatures:
    """
    Fast Fourier Transform (FFT) feature extractor.

    Parameters
    ----------
    sampling_rate : float
        Signal sampling frequency in Hz.
    """

    def __init__(self, sampling_rate: float):
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive.")

        self.fs = float(sampling_rate)

    def spectrum(
        self,
        signal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the single-sided FFT spectrum.

        Parameters
        ----------
        signal : np.ndarray
            Input 1D signal.

        Returns
        -------
        frequencies : np.ndarray
        magnitude : np.ndarray
        """

        signal = np.asarray(signal, dtype=np.float32)

        if signal.ndim != 1:
            raise ValueError("Signal must be one-dimensional.")

        n = len(signal)

        fft = np.fft.rfft(signal)
        magnitude = np.abs(fft) / n
        frequency = np.fft.rfftfreq(n, d=1 / self.fs)

        return frequency, magnitude

    def power_spectrum(
        self,
        signal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density (PSD).

        Returns
        -------
        frequencies
        power
        """

        frequency, magnitude = self.spectrum(signal)

        power = magnitude ** 2

        return frequency, power

    def dominant_frequency(
        self,
        signal: np.ndarray,
    ) -> float:
        """
        Return dominant frequency.
        """

        frequency, magnitude = self.spectrum(signal)

        idx = np.argmax(magnitude)

        return float(frequency[idx])

    def spectral_centroid(
        self,
        signal: np.ndarray,
    ) -> float:
        """
        Compute spectral centroid.
        """

        frequency, magnitude = self.spectrum(signal)

        denominator = np.sum(magnitude)

        if denominator == 0:
            return 0.0

        return float(
            np.sum(frequency * magnitude)
            / denominator
        )

    def spectral_bandwidth(
        self,
        signal: np.ndarray,
    ) -> float:
        """
        Compute spectral bandwidth.
        """

        frequency, magnitude = self.spectrum(signal)

        centroid = self.spectral_centroid(signal)

        denominator = np.sum(magnitude)

        if denominator == 0:
            return 0.0

        bandwidth = np.sqrt(
            np.sum(
                ((frequency - centroid) ** 2)
                * magnitude
            )
            / denominator
        )

        return float(bandwidth)

    def spectral_entropy(
        self,
        signal: np.ndarray,
    ) -> float:
        """
        Compute normalized spectral entropy.
        """

        _, power = self.power_spectrum(signal)

        power = power / (np.sum(power) + 1e-12)

        entropy = -np.sum(
            power * np.log2(power + 1e-12)
        )

        entropy /= np.log2(len(power))

        return float(entropy)

    def extract(
        self,
        signal: np.ndarray,
    ) -> dict[str, float]:
        """
        Extract FFT-based features.

        Returns
        -------
        dict
        """

        return {
            "dominant_frequency": self.dominant_frequency(signal),
            "spectral_centroid": self.spectral_centroid(signal),
            "spectral_bandwidth": self.spectral_bandwidth(signal),
            "spectral_entropy": self.spectral_entropy(signal),
        }