"""
Visualization utilities for spike encoders.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class SpikeVisualizer:
    """
    Visualization utilities for spike encoders.
    """

    # ---------------------------------------------------------
    # Signal
    # ---------------------------------------------------------

    @staticmethod
    def signal(
        signal: np.ndarray,
        figsize=(10, 3),
    ):
        """
        Plot input signal.
        """

        signal = np.asarray(signal)

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(signal)

        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.set_title("Input Signal")
        ax.grid(True)

        plt.tight_layout()

        return fig, ax

    # ---------------------------------------------------------
    # Raster
    # ---------------------------------------------------------

    @staticmethod
    def raster(
        spikes: np.ndarray,
        figsize=(10, 5),
    ):
        """
        Raster plot.

        Parameters
        ----------
        spikes : ndarray
            Shape:
            (time_steps, neurons)
        """

        spikes = np.asarray(spikes)

        if spikes.ndim != 2:
            raise ValueError(
                "Raster expects a 2D spike train."
            )

        t, neurons = np.where(spikes)

        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(
            t,
            neurons,
            s=5,
            marker="|",
        )

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Neuron")
        ax.set_title("Spike Raster")

        plt.tight_layout()

        return fig, ax

    # ---------------------------------------------------------
    # Signal + Raster
    # ---------------------------------------------------------

    @staticmethod
    def signal_and_spikes(
        signal: np.ndarray,
        spikes: np.ndarray,
        figsize=(10, 6),
    ):
        """
        Plot signal and spike raster together.
        """

        spikes = np.asarray(spikes)

        if spikes.ndim != 2:
            raise ValueError(
                "Spike train must be 2D."
            )

        fig, ax = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=False,
        )

        ax[0].plot(signal)
        ax[0].set_title("Input Signal")
        ax[0].grid(True)

        t, neurons = np.where(spikes)

        ax[1].scatter(
            t,
            neurons,
            s=5,
            marker="|",
        )

        ax[1].set_title("Spike Raster")
        ax[1].set_xlabel("Time Step")
        ax[1].set_ylabel("Neuron")

        plt.tight_layout()

        return fig, ax

    # ---------------------------------------------------------
    # Population activity
    # ---------------------------------------------------------

    @staticmethod
    def population_activity(
        spikes: np.ndarray,
        figsize=(10, 3),
    ):
        """
        Plot total spikes at each time step.
        """

        spikes = np.asarray(spikes)

        activity = spikes.sum(axis=1)

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(activity)

        ax.set_title("Population Activity")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Spike Count")

        plt.tight_layout()

        return fig, ax

    # ---------------------------------------------------------
    # Firing rate
    # ---------------------------------------------------------

    @staticmethod
    def firing_rate(
        spikes: np.ndarray,
        figsize=(10, 3),
    ):
        """
        Plot firing rate of each neuron.
        """

        spikes = np.asarray(spikes)

        rate = spikes.mean(axis=0)

        fig, ax = plt.subplots(figsize=figsize)

        ax.bar(
            np.arange(rate.size),
            rate,
        )

        ax.set_title("Neuron Firing Rate")
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Rate")

        plt.tight_layout()

        return fig, ax

    # ---------------------------------------------------------
    # Spike count
    # ---------------------------------------------------------

    @staticmethod
    def spike_count(
        spikes: np.ndarray,
        figsize=(10, 3),
    ):
        """
        Plot spike count.
        """

        spikes = np.asarray(spikes)

        count = spikes.sum(axis=0)

        fig, ax = plt.subplots(figsize=figsize)

        ax.bar(
            np.arange(count.size),
            count,
        )

        ax.set_title("Spike Count")
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Spikes")

        plt.tight_layout()

        return fig, ax

    # ---------------------------------------------------------
    # Compare encoders
    # ---------------------------------------------------------

    @staticmethod
    def compare(
        encoder_outputs: dict[str, np.ndarray],
        figsize=(10, 3),
    ):
        """
        Compare multiple spike encoders.

        Parameters
        ----------
        encoder_outputs

            {
                "Rate": spikes,
                "Latency": spikes,
                ...
            }
        """

        n = len(encoder_outputs)

        fig, axes = plt.subplots(
            n,
            1,
            figsize=(figsize[0], figsize[1] * n),
        )

        if n == 1:
            axes = [axes]

        for ax, (name, spikes) in zip(
            axes,
            encoder_outputs.items(),
        ):

            t, neurons = np.where(spikes)

            ax.scatter(
                t,
                neurons,
                s=4,
                marker="|",
            )

            ax.set_title(name)

        plt.tight_layout()

        return fig, axes

    # ---------------------------------------------------------
    # Delta events
    # ---------------------------------------------------------

    @staticmethod
    def delta_events(
        signal: np.ndarray,
        spikes: np.ndarray,
        figsize=(10, 4),
    ):
        """
        Visualize DeltaEncoder ON/OFF events.

        Parameters
        ----------
        spikes

            Shape:
            (2, signal_length)
        """

        if spikes.shape[0] != 2:
            raise ValueError(
                "Expected (2, signal_length)."
            )

        on = np.where(spikes[0])[0]
        off = np.where(spikes[1])[0]

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(signal)

        ax.scatter(
            on,
            signal[on],
            marker="^",
            s=40,
            label="ON",
        )

        ax.scatter(
            off,
            signal[off],
            marker="v",
            s=40,
            label="OFF",
        )

        ax.legend()

        ax.set_title("Delta Events")

        plt.tight_layout()

        return fig, ax

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    @staticmethod
    def save(
        fig,
        filename: str,
        dpi: int = 300,
    ):
        """
        Save figure.
        """

        fig.savefig(
            filename,
            dpi=dpi,
            bbox_inches="tight",
        )