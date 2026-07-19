"""
Feature extraction pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .cepstrum import CepstrumFeatures
from .envelope import EnvelopeAnalysis
from .fft import FFTFeatures
from .stft import STFTFeatures
from .wavelet import WaveletFeatures


class FeaturePipeline:
    """
    Unified feature extraction pipeline.

    Parameters
    ----------
    sampling_rate : float
        Signal sampling frequency.

    use_fft : bool
    use_wavelet : bool
    use_stft : bool
    use_cepstrum : bool
    use_envelope : bool
    """

    def __init__(
        self,
        sampling_rate: float,
        use_fft: bool = True,
        use_wavelet: bool = True,
        use_stft: bool = False,
        use_cepstrum: bool = False,
        use_envelope: bool = False,
    ):

        self.extractors: list[Any] = []

        if use_fft:
            self.extractors.append(
                FFTFeatures(sampling_rate)
            )

        if use_wavelet:
            self.extractors.append(
                WaveletFeatures()
            )

        if use_stft:
            self.extractors.append(
                STFTFeatures(sampling_rate)
            )

        if use_cepstrum:
            self.extractors.append(
                CepstrumFeatures()
            )

        if use_envelope:
            self.extractors.append(
                EnvelopeAnalysis(sampling_rate)
            )

    @staticmethod
    def _flatten(value: Any) -> np.ndarray:
        """
        Convert feature outputs into a 1D numeric vector.
        """

        if isinstance(value, (float, int)):
            return np.array([value], dtype=np.float32)

        if isinstance(value, np.ndarray):
            return value.astype(np.float32).ravel()

        if isinstance(value, list):
            arrays = [
                np.asarray(v).ravel()
                for v in value
            ]
            return np.concatenate(arrays).astype(np.float32)

        return np.array([], dtype=np.float32)

    def extract(
        self,
        signal: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """
        Extract all enabled features.

        Returns
        -------
        feature_vector
        raw_feature_dictionary
        """

        feature_vector = []
        feature_dict = {}

        for extractor in self.extractors:

            result = extractor.extract(signal)

            feature_dict[
                extractor.__class__.__name__
            ] = result

            for value in result.values():

                if np.iscomplexobj(value):
                    continue

                feature_vector.extend(
                    self._flatten(value)
                )

        return (
            np.asarray(
                feature_vector,
                dtype=np.float32,
            ),
            feature_dict,
        )