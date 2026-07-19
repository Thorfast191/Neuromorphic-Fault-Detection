"""
Temporal spike encoder.
"""

from __future__ import annotations

import numpy as np

from .base_encoder import BaseEncoder
from .registry import ENCODERS


@ENCODERS.register("temporal")
class TemporalEncoder(BaseEncoder):
    """
    Temporal spike encoder.

    Larger amplitudes generate spikes more frequently
    throughout the simulation.

    Each neuron may emit multiple spikes.
    """

    def __init__(
        self,
        time_steps: int = 100,
        dt: float = 1.0,
        normalize: bool = True,
        min_interval: int = 2,
    ):
        super().__init__(
            time_steps=time_steps,
            dt=dt,
            normalize=normalize,
        )

        if min_interval < 1:
            raise ValueError(
                "min_interval must be >= 1."
            )

        self.min_interval = int(min_interval)

    def encode(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Encode a signal using temporal spike coding.

        Parameters
        ----------
        signal : np.ndarray
            One-dimensional input signal.

        Returns
        -------
        np.ndarray
            Binary spike train of shape
            (time_steps, signal_length).
        """

        signal = self.preprocess(signal)

        n_neurons = signal.size

        spikes = np.zeros(
            (self.time_steps, n_neurons),
            dtype=np.uint8,
        )

        intervals = np.interp(
            signal,
            [0.0, 1.0],
            [self.time_steps, self.min_interval],
        ).astype(np.int32)

        intervals = np.maximum(
            intervals,
            self.min_interval,
        )

        for neuron, interval in enumerate(intervals):

            spikes[
                ::interval,
                neuron,
            ] = 1

        return spikes

    def spike_intervals(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Return spike interval for each neuron.
        """

        signal = self.preprocess(signal)

        intervals = np.interp(
            signal,
            [0.0, 1.0],
            [self.time_steps, self.min_interval],
        )

        return intervals.astype(np.int32)

    def extract(
        self,
        signal: np.ndarray,
    ) -> dict:
        """
        Encode signal and return temporal features.
        """

        result = super().extract(signal)

        result.update(
            {
                "intervals": self.spike_intervals(
                    signal
                ),
                "min_interval": self.min_interval,
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
                "min_interval": self.min_interval,
            }
        )

        return config