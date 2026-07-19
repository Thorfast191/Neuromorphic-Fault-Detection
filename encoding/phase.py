"""
Phase-based spike encoder.
"""

from __future__ import annotations

import numpy as np

from .base_encoder import BaseEncoder
from .registry import ENCODERS


@ENCODERS.register("phase")
class PhaseEncoder(BaseEncoder):
    """
    Simplified phase-based spike encoder.

    The normalized signal amplitude is mapped to a phase
    in the interval [0, 2π], which is then converted into
    a spike time within the simulation window.

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

    def phase(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Convert signal amplitude to phase (radians).

        Parameters
        ----------
        signal : np.ndarray
            One-dimensional input signal.

        Returns
        -------
        np.ndarray
            Phase values in radians.
        """

        signal = self.preprocess(signal)

        return signal * (2.0 * np.pi)

    def phase_to_time(
        self,
        phase: np.ndarray,
    ) -> np.ndarray:
        """
        Convert phase values into spike times.

        Parameters
        ----------
        phase : np.ndarray
            Phase values in radians.

        Returns
        -------
        np.ndarray
            Spike times.
        """

        spike_times = np.round(
            phase
            / (2.0 * np.pi)
            * (self.time_steps - 1)
        ).astype(np.int32)

        return np.clip(
            spike_times,
            0,
            self.time_steps - 1,
        )

    def encode(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Encode signal using phase coding.

        Returns
        -------
        np.ndarray
            Binary spike train with shape
            (time_steps, signal_length).
        """

        phase = self.phase(signal)

        spike_times = self.phase_to_time(phase)

        n_neurons = spike_times.size

        spikes = np.zeros(
            (self.time_steps, n_neurons),
            dtype=np.uint8,
        )

        spikes[
            spike_times,
            np.arange(n_neurons),
        ] = 1

        return spikes

    def extract(
        self,
        signal: np.ndarray,
    ) -> dict:
        """
        Encode signal and return phase information.
        """

        result = super().extract(signal)

        phase = self.phase(signal)

        result.update(
            {
                "phase": phase,
                "phase_degrees": np.degrees(phase),
                "spike_times": self.phase_to_time(
                    phase
                ),
            }
        )

        return result

    def get_config(self) -> dict:
        """
        Return encoder configuration.
        """

        return super().get_config()