"""
Population spike encoder.
"""

from __future__ import annotations

import numpy as np

from .base_encoder import BaseEncoder
from .registry import ENCODERS


@ENCODERS.register("population")
class PopulationEncoder(BaseEncoder):
    """
    Population coding encoder.

    Each input value is represented by multiple neurons
    with Gaussian receptive fields.

    Parameters
    ----------
    neurons_per_input : int
        Number of neurons representing each input sample.

    sigma : float
        Width of Gaussian receptive fields.

    stochastic : bool
        If True, generate Poisson-like spikes.
        If False, emit one spike according to the neuron response.

    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        neurons_per_input: int = 8,
        sigma: float = 0.15,
        stochastic: bool = True,
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

        if neurons_per_input < 2:
            raise ValueError(
                "neurons_per_input must be >= 2."
            )

        if sigma <= 0:
            raise ValueError(
                "sigma must be positive."
            )

        self.neurons_per_input = neurons_per_input
        self.sigma = sigma
        self.stochastic = stochastic
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        self.centers = np.linspace(
            0.0,
            1.0,
            neurons_per_input,
            dtype=np.float32,
        )

    def encode(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Encode signal using Gaussian population coding.

        Returns
        -------
        np.ndarray
            Shape:
            (time_steps,
             signal_length * neurons_per_input)
        """

        signal = self.preprocess(signal)

        # (n_inputs, neurons_per_input)
        response = np.exp(
            -(
                signal[:, None] - self.centers[None, :]
            ) ** 2
            / (2 * self.sigma**2)
        )

        response /= (
            response.max(axis=1, keepdims=True)
            + 1e-12
        )

        response = response.reshape(-1)

        n_neurons = response.size

        if self.stochastic:

            spikes = (
                self.rng.random(
                    (self.time_steps, n_neurons)
                )
                < response
            )

        else:

            spikes = np.zeros(
                (self.time_steps, n_neurons),
                dtype=np.uint8,
            )

            spike_time = self.time_steps // 2

            spikes[
                spike_time,
                response >= 0.5,
            ] = 1

        return spikes.astype(np.uint8)

    def receptive_fields(self) -> np.ndarray:
        """
        Return receptive field centers.
        """

        return self.centers.copy()

    def reseed(
        self,
        seed: int | None = None,
    ) -> None:
        """
        Reset random number generator.
        """

        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def extract(
        self,
        signal: np.ndarray,
    ) -> dict:
        """
        Encode signal and return encoder metadata.
        """

        result = super().extract(signal)

        result.update(
            {
                "population_size": result["spikes"].shape[1],
                "neurons_per_input": self.neurons_per_input,
                "sigma": self.sigma,
                "centers": self.receptive_fields(),
                "stochastic": self.stochastic,
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
                "neurons_per_input": self.neurons_per_input,
                "sigma": self.sigma,
                "stochastic": self.stochastic,
                "seed": self.seed,
            }
        )

        return config