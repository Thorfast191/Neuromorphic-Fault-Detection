"""
Abstract base class for spike encoders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEncoder(ABC):
    """
    Base class for all spike encoders.

    Parameters
    ----------
    time_steps : int
        Number of simulation time steps.

    dt : float
        Simulation time step.

    normalize : bool
        Normalize input signal before encoding.
    """

    def __init__(
        self,
        time_steps: int = 100,
        dt: float = 1.0,
        normalize: bool = True,
    ):

        if time_steps <= 0:
            raise ValueError("time_steps must be positive.")

        if dt <= 0:
            raise ValueError("dt must be positive.")

        self.time_steps = int(time_steps)
        self.dt = float(dt)
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Encoder name."""
        return self.__class__.__name__

    @property
    def output_shape(self) -> tuple[int, int]:
        """Expected output shape."""
        return (self.time_steps, -1)

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------

    @abstractmethod
    def encode(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Encode a signal into spikes.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def preprocess(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Validate and optionally normalize input.
        """

        signal = np.asarray(signal, dtype=np.float32)

        if signal.ndim != 1:
            raise ValueError("Signal must be one-dimensional.")

        if signal.size == 0:
            raise ValueError("Signal cannot be empty.")

        if self.normalize:

            minimum = signal.min()
            maximum = signal.max()

            if maximum > minimum:
                signal = (signal - minimum) / (maximum - minimum)
            else:
                signal = np.zeros_like(signal)

        return signal

    # ------------------------------------------------------------------
    # Generic spike statistics
    # ------------------------------------------------------------------

    def spike_count(
        self,
        spikes: np.ndarray,
    ) -> np.ndarray:
        """
        Count spikes per neuron.
        """

        spikes = np.asarray(spikes)

        if spikes.ndim != 2:
            raise ValueError("Spike train must be 2D.")

        return spikes.sum(axis=0)

    def firing_rate(
        self,
        spikes: np.ndarray,
    ) -> np.ndarray:
        """
        Average firing rate.
        """

        return self.spike_count(spikes) / self.time_steps

    def spike_density(
        self,
        spikes: np.ndarray,
    ) -> np.ndarray:
        """
        Population activity over time.
        """

        spikes = np.asarray(spikes)

        if spikes.ndim != 2:
            raise ValueError("Spike train must be 2D.")

        return spikes.sum(axis=1)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        signal: np.ndarray,
    ) -> dict:
        """
        Standard extraction interface.
        """

        spikes = self.encode(signal)

        return {
            "spikes": spikes,
            "spike_count": self.spike_count(spikes),
            "firing_rate": self.firing_rate(spikes),
            "spike_density": self.spike_density(spikes),
        }

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """
        Return encoder configuration.
        """

        return {
            "encoder": self.name,
            "time_steps": self.time_steps,
            "dt": self.dt,
            "normalize": self.normalize,
        }

    # ------------------------------------------------------------------
    # Magic methods
    # ------------------------------------------------------------------

    def __call__(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        return self.encode(signal)

    def __repr__(self) -> str:

        return (
            f"{self.name}("
            f"time_steps={self.time_steps}, "
            f"dt={self.dt}, "
            f"normalize={self.normalize})"
        )