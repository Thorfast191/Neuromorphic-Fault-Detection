"""
Adaptive spike encoder.
"""

from __future__ import annotations

import numpy as np

from .base_encoder import BaseEncoder
from .factory import create_encoder
from .registry import ENCODERS


@ENCODERS.register("adaptive")
class AdaptiveEncoder(BaseEncoder):
    """
    Adaptive spike encoder.

    The input signal is first adaptively normalized
    according to its statistics and then passed to
    another registered encoder.

    Parameters
    ----------
    method : str
        Name of the underlying encoder.

    adaptive_factor : float
        Scaling applied after normalization.

    mode : str
        Normalization strategy.

        - "zscore"
        - "minmax"
        - "robust"
    """

    def __init__(
        self,
        method: str = "latency",
        adaptive_factor: float = 1.0,
        mode: str = "zscore",
        time_steps: int = 100,
        dt: float = 1.0,
        normalize: bool = True,
        seed: int | None = None,
        **encoder_kwargs,
    ):
        super().__init__(
            time_steps=time_steps,
            dt=dt,
            normalize=normalize,
        )

        self.method = method.lower()
        self.mode = mode.lower()
        self.adaptive_factor = adaptive_factor
        self.seed = seed
        self.encoder_kwargs = encoder_kwargs

    def _statistics(
        self,
        signal: np.ndarray,
    ) -> dict:
        """
        Compute signal statistics.
        """

        return {
            "mean": float(np.mean(signal)),
            "std": float(np.std(signal)),
            "median": float(np.median(signal)),
            "iqr": float(
                np.percentile(signal, 75)
                - np.percentile(signal, 25)
            ),
            "min": float(np.min(signal)),
            "max": float(np.max(signal)),
        }

    def _adaptive_normalize(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Apply adaptive normalization.
        """

        stats = self._statistics(signal)

        if self.mode == "zscore":

            signal = (
                signal - stats["mean"]
            ) / (stats["std"] + 1e-8)

            signal = np.clip(signal, -3, 3)

            signal = (signal + 3.0) / 6.0

        elif self.mode == "minmax":

            signal = (
                signal - stats["min"]
            ) / (
                stats["max"] - stats["min"] + 1e-8
            )

        elif self.mode == "robust":

            signal = (
                signal - stats["median"]
            ) / (stats["iqr"] + 1e-8)

            signal = np.clip(signal, -3, 3)

            signal = (signal + 3.0) / 6.0

        else:

            raise ValueError(
                f"Unknown normalization mode '{self.mode}'."
            )

        signal *= self.adaptive_factor

        return np.clip(signal, 0.0, 1.0)

    def encode(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Encode an adaptively normalized signal.
        """

        signal = self.preprocess(signal)

        signal = self._adaptive_normalize(signal)

        encoder = create_encoder(
            self.method,
            time_steps=self.time_steps,
            dt=self.dt,
            normalize=False,
            seed=self.seed,
            **self.encoder_kwargs,
        )

        return encoder.encode(signal)

    def extract(
        self,
        signal: np.ndarray,
    ) -> dict:
        """
        Encode signal and return statistics.
        """

        signal = self.preprocess(signal)

        stats = self._statistics(signal)

        adaptive_signal = self._adaptive_normalize(signal)

        encoder = create_encoder(
            self.method,
            time_steps=self.time_steps,
            dt=self.dt,
            normalize=False,
            seed=self.seed,
            **self.encoder_kwargs,
        )

        result = encoder.extract(adaptive_signal)

        result.update(
            {
                "statistics": stats,
                "adaptive_factor": self.adaptive_factor,
                "mode": self.mode,
                "method": self.method,
            }
        )

        return result

    def get_config(self) -> dict:
        """
        Return encoder configuration.
        """

        config = super().get_config()

        config.update(
            {
                "method": self.method,
                "mode": self.mode,
                "adaptive_factor": self.adaptive_factor,
                "seed": self.seed,
                "encoder_kwargs": self.encoder_kwargs,
            }
        )

        return config