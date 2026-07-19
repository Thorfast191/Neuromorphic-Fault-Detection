"""
Hybrid spike encoder.
"""

from __future__ import annotations

import numpy as np

from .base_encoder import BaseEncoder
from .factory import create_encoder
from .registry import ENCODERS


@ENCODERS.register("hybrid")
class HybridEncoder(BaseEncoder):
    """
    Hybrid spike encoder.

    Combines multiple spike encoders by concatenating
    their outputs along the neuron dimension.

    Parameters
    ----------
    encoders : tuple[str] | list[str]
        Names of registered encoders.

    encoder_kwargs : dict[str, dict], optional
        Encoder-specific keyword arguments.

        Example
        -------
        {
            "population": {
                "neurons_per_input": 8,
                "sigma": 0.2,
            },
            "rate": {
                "seed": 42,
            },
        }
    """

    def __init__(
        self,
        encoders=("rate", "latency"),
        encoder_kwargs=None,
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

        if len(encoders) == 0:
            raise ValueError(
                "At least one encoder must be provided."
            )

        self.encoders = tuple(encoders)
        self.encoder_kwargs = encoder_kwargs or {}
        self.seed = seed

    def _build(self, name: str):
        """
        Construct a registered encoder.
        """

        kwargs = self.encoder_kwargs.get(name, {}).copy()

        kwargs.setdefault(
            "time_steps",
            self.time_steps,
        )

        kwargs.setdefault(
            "dt",
            self.dt,
        )

        kwargs.setdefault(
            "normalize",
            False,
        )

        if self.seed is not None:
            kwargs.setdefault(
                "seed",
                self.seed,
            )

        return create_encoder(
            name,
            **kwargs,
        )

    def encode(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Encode a signal using multiple encoders.

        Returns
        -------
        np.ndarray
            Concatenated spike trains.
        """

        signal = self.preprocess(signal)

        outputs = []

        for name in self.encoders:

            encoder = self._build(name)

            spikes = encoder.encode(signal)

            if spikes.ndim != 2:
                raise ValueError(
                    f"{name} encoder returned "
                    f"{spikes.ndim}D output. "
                    "HybridEncoder requires 2D spike trains."
                )

            outputs.append(spikes)

        return np.concatenate(
            outputs,
            axis=1,
        )

    def extract(
        self,
        signal: np.ndarray,
    ) -> dict:
        """
        Encode signal using multiple encoders.
        """

        signal = self.preprocess(signal)

        outputs = {}
        spike_list = []

        for name in self.encoders:

            encoder = self._build(name)

            result = encoder.extract(signal)

            outputs[name] = result

            spike_list.append(result["spikes"])

        spikes = np.concatenate(
            spike_list,
            axis=1,
        )

        return {
            "spikes": spikes,
            "neurons": spikes.shape[1],
            "encoders": list(self.encoders),
            "individual": outputs,
        }

    def get_config(self) -> dict:
        """
        Return encoder configuration.
        """

        config = super().get_config()

        config.update(
            {
                "encoders": list(self.encoders),
                "encoder_kwargs": self.encoder_kwargs,
                "seed": self.seed,
            }
        )

        return config