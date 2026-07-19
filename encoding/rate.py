"""
Poisson rate-based spike encoder.
"""

from __future__ import annotations

import numpy as np

from .base_encoder import BaseEncoder
from .registry import ENCODERS


@ENCODERS.register("rate")
class RateEncoder(BaseEncoder):
    """
    Poisson rate-based spike encoder.

    Each normalized input value represents the firing
    probability of one neuron.

    Higher amplitudes produce higher firing rates.

    Parameters
    ----------
    time_steps : int
        Number of simulation time steps.

    dt : float
        Simulation time step.

    normalize : bool
        Normalize input signal before encoding.

    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        time_steps: int = 100,
        dt: float = 1.0,
        normalize: bool = True,
        seed: int | None = None,
    ):
        super().__init__(
            time_steps=time_steps,
            dt=dt,
            normalize=normalize,
        )

        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def encode(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Encode a signal using Poisson rate coding.

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

        spikes = self.rng.random(
            (self.time_steps, signal.size)
        ) < signal

        return spikes.astype(np.uint8)

    def reseed(
        self,
        seed: int | None = None,
    ) -> None:
        """
        Reset the random number generator.

        Parameters
        ----------
        seed : int | None
            New random seed.
        """

        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def get_config(self) -> dict:
        """
        Return encoder configuration.
        """

        config = super().get_config()

        config.update(
            {
                "seed": self.seed,
            }
        )

        return config