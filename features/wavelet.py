"""
Wavelet-based feature extraction utilities.
"""

from __future__ import annotations

import numpy as np
import pywt


class WaveletFeatures:
    """
    Discrete Wavelet Transform (DWT) feature extractor.

    Parameters
    ----------
    wavelet : str, default="db4"
        Wavelet family.

    level : int, default=4
        Decomposition level.
    """

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 4,
    ):

        self.wavelet = wavelet
        self.level = level

    def decompose(
        self,
        signal: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Perform DWT decomposition.

        Parameters
        ----------
        signal : np.ndarray
            Input signal.

        Returns
        -------
        list[np.ndarray]
            Approximation and detail coefficients.
        """

        signal = np.asarray(signal, dtype=np.float32)

        if signal.ndim != 1:
            raise ValueError(
                "Signal must be one-dimensional."
            )

        return pywt.wavedec(
            signal,
            wavelet=self.wavelet,
            level=self.level,
        )

    def energy(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Energy of each wavelet coefficient.
        """

        coeffs = self.decompose(signal)

        return np.array(
            [
                np.sum(c ** 2)
                for c in coeffs
            ],
            dtype=np.float32,
        )

    def relative_energy(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Relative energy of each decomposition level.
        """

        energy = self.energy(signal)

        total = np.sum(energy)

        if total == 0:
            return energy

        return energy / total

    def entropy(
        self,
        signal: np.ndarray,
    ) -> float:
        """
        Shannon entropy of wavelet energies.
        """

        p = self.relative_energy(signal)

        return float(
            -np.sum(
                p * np.log2(p + 1e-12)
            )
        )

    def extract(
        self,
        signal: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        """
        Extract wavelet features.
        """

        coeffs = self.decompose(signal)

        return {
            "coefficients": coeffs,
            "energy": self.energy(signal),
            "relative_energy": self.relative_energy(signal),
            "entropy": self.entropy(signal),
        }