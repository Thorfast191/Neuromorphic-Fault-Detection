"""
Delta spike encoder.

Encodes changes in a signal into ON/OFF spike events.
"""

from __future__ import annotations

import numpy as np

from .base_encoder import BaseEncoder
from .registry import ENCODERS


@ENCODERS.register("delta")
class DeltaEncoder(BaseEncoder):
    """
    Delta (event-based) spike encoder.

    A spike is generated whenever the change between two
    consecutive signal samples exceeds a threshold.

    Positive changes generate ON spikes.

    Negative changes generate OFF spikes.

    Notes
    -----
    This encoder operates on the temporal evolution of the
    input signal (consecutive samples), making it suitable
    for vibration signals and event-based processing.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        dt: float = 1.0,
        normalize: bool = True,
    ):
        # Simulation time is determined by the signal length
        super().__init__(
            time_steps=1,
            dt=dt,
            normalize=normalize,
        )

        if threshold <= 0:
            raise ValueError(
                "threshold must be positive."
            )

        self.threshold = float(threshold)

    def encode(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Encode a signal into ON/OFF events.

        Parameters
        ----------
        signal : np.ndarray
            Input vibration signal.

        Returns
        -------
        np.ndarray
            Shape
            -----
            (2, signal_length)

            spikes[0] -> ON events

            spikes[1] -> OFF events
        """

        signal = self.preprocess(signal)

        delta = np.diff(
            signal,
            prepend=signal[0],
        )

        on = delta >= self.threshold
        off = delta <= -self.threshold

        spikes = np.vstack(
            (
                on.astype(np.uint8),
                off.astype(np.uint8),
            )
        )

        return spikes

    def on_spikes(
        self,
        spikes: np.ndarray,
    ) -> np.ndarray:
        """
        Return ON events.
        """

        return spikes[0]

    def off_spikes(
        self,
        spikes: np.ndarray,
    ) -> np.ndarray:
        """
        Return OFF events.
        """

        return spikes[1]

    def spike_count(
        self,
        spikes: np.ndarray,
    ) -> dict[str, int]:
        """
        Count ON/OFF spikes.
        """

        return {
            "on": int(spikes[0].sum()),
            "off": int(spikes[1].sum()),
            "total": int(spikes.sum()),
        }

    def event_indices(
        self,
        spikes: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Return indices of ON and OFF events.
        """

        return {
            "on": np.where(spikes[0])[0],
            "off": np.where(spikes[1])[0],
        }

    def extract(
        self,
        signal: np.ndarray,
    ) -> dict:
        """
        Encode signal and return event statistics.
        """

        spikes = self.encode(signal)

        return {
            "spikes": spikes,
            "on_spikes": self.on_spikes(spikes),
            "off_spikes": self.off_spikes(spikes),
            "event_indices": self.event_indices(spikes),
            "spike_count": self.spike_count(spikes),
            "threshold": self.threshold,
        }

    def get_config(self) -> dict:
        """
        Return encoder configuration.
        """

        config = super().get_config()

        config.update(
            {
                "threshold": self.threshold,
            }
        )

        return config