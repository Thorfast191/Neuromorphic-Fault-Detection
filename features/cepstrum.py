"""
Cepstrum feature extraction utilities.
"""

from __future__ import annotations

import numpy as np


class CepstrumFeatures:
    """
    Real cepstrum feature extractor.
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def real_cepstrum(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the real cepstrum.

        Parameters
        ----------
        signal : np.ndarray
            Input vibration signal.

        Returns
        -------
        np.ndarray
            Real cepstrum.
        """

        signal = np.asarray(signal, dtype=np.float32)

        if signal.ndim != 1:
            raise ValueError(
                "Signal must be one-dimensional."
            )

        spectrum = np.fft.fft(signal)

        log_magnitude = np.log(
            np.abs(spectrum) + self.eps
        )

        cepstrum = np.fft.ifft(
            log_magnitude
        ).real

        return cepstrum.astype(np.float32)

    def peak_quefrency(
        self,
        signal: np.ndarray,
    ) -> float:
        """
        Quefrency corresponding to the maximum cepstral peak.
        """

        cep = self.real_cepstrum(signal)

        # Ignore DC component
        idx = np.argmax(cep[1:]) + 1

        return float(idx)

    def cepstral_energy(
        self,
        signal: np.ndarray,
    ) -> float:
        """
        Total cepstral energy.
        """

        cep = self.real_cepstrum(signal)

        return float(
            np.sum(cep ** 2)
        )

    def rms(
        self,
        signal: np.ndarray,
    ) -> float:
        """
        Cepstral RMS.
        """

        cep = self.real_cepstrum(signal)

        return float(
            np.sqrt(
                np.mean(cep ** 2)
            )
        )

    def extract(
        self,
        signal: np.ndarray,
    ) -> dict[str, object]:
        """
        Extract cepstral features.
        """

        cep = self.real_cepstrum(signal)

        return {
            "cepstrum": cep,
            "peak_quefrency": self.peak_quefrency(signal),
            "cepstral_energy": self.cepstral_energy(signal),
            "cepstral_rms": self.rms(signal),
        }