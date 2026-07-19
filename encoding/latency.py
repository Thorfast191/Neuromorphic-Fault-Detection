"""
Latency (Time-to-First-Spike) encoder.
"""

from __future__ import annotations

import numpy as np

from .base_encoder import BaseEncoder
from .registry import ENCODERS


@ENCODERS.register("latency")
class LatencyEncoder(BaseEncoder):
    """
    Time-to-First-Spike (TTFS) encoder.

    Larger amplitudes generate earlier spikes.

    Each neuron emits exactly one spike.
    """

    def __init__(
        self,
        time_steps: int = 100,
        dt: float = 1.0,
        normalize: bool = True,
    ):
        super().__init__(
            time_steps=time_steps,
            dt=dt,
            normalize=normalize,
        )

    def encode(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Encode a signal using latency coding.

        Parameters
        ----------
        signal : np.ndarray
            One-dimensional input signal.

        Returns
        -------
        np.ndarray
            Binary spike train with shape
            (time_steps, signal_length).
        """

        signal = self.preprocess(signal)

        n_neurons = signal.size

        spikes = np.zeros(
            (self.time_steps, n_neurons),
            dtype=np.uint8,
        )

        spike_times = np.round(
            (1.0 - signal)
            * (self.time_steps - 1)
        ).astype(np.int32)

        spike_times = np.clip(
            spike_times,
            0,
            self.time_steps - 1,
        )

        spikes[
            spike_times,
            np.arange(n_neurons),
        ] = 1

        return spikes

    def spike_times(
        self,
        spikes: np.ndarray,
    ) -> np.ndarray:
        """
        Return the first spike time of each neuron.

        Parameters
        ----------
        spikes : np.ndarray
            Spike train with shape
            (time_steps, neurons).

        Returns
        -------
        np.ndarray
            Spike time for each neuron.
            Neurons that never spike return -1.
        """

        spikes = np.asarray(spikes)

        if spikes.ndim != 2:
            raise ValueError(
                "Spike train must be 2-dimensional."
            )

        n_neurons = spikes.shape[1]

        times = np.full(
            n_neurons,
            -1,
            dtype=np.int32,
        )

        active = np.argwhere(spikes)

        for t, neuron in active:
            if times[neuron] == -1:
                times[neuron] = t

        return times

    def latency(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Compute spike latencies directly from the input.

        Parameters
        ----------
        signal : np.ndarray

        Returns
        -------
        np.ndarray
            Spike time for each input sample.
        """

        signal = self.preprocess(signal)

        return np.round(
            (1.0 - signal)
            * (self.time_steps - 1)
        ).astype(np.int32)

    def extract(
        self,
        signal: np.ndarray,
    ) -> dict:
        """
        Encode the signal and compute latency features.
        """

        result = super().extract(signal)

        result.update(
            {
                "spike_times": self.spike_times(
                    result["spikes"]
                ),
                "latency": self.latency(signal),
            }
        )

        return result

    def get_config(self) -> dict:
        """
        Return encoder configuration.
        """

        return super().get_config()