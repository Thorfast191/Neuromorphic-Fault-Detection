"""
Spiking linear layer: a dense synaptic connection followed by a
LIF neuron.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.neurons.lif import LIFNeuron


class SpikingLinear(nn.Module):
    """
    Fully-connected synapse + LIF neuron layer.

    Parameters
    ----------
    in_features, out_features : int
        Synapse dimensions.

    beta : float
        LIF membrane decay rate.

    surrogate_name : str
        Surrogate gradient used for backprop through spikes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        beta: float = 0.9,
        surrogate_name: str = "fast_sigmoid",
        learn_beta: bool = False,
        threshold: float = 1.0,
        **surrogate_kwargs,
    ):
        super().__init__()

        self.out_features = out_features

        self.synapse = nn.Linear(in_features, out_features)

        self.neuron = LIFNeuron(
            beta=beta,
            surrogate_name=surrogate_name,
            learn_beta=learn_beta,
            threshold=threshold,
            **surrogate_kwargs,
        )

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Return a zeroed membrane-potential tensor for this layer.
        """

        return self.neuron.init_state(
            batch_size,
            self.out_features,
            device,
        )

    def forward(
        self,
        x: torch.Tensor,
        membrane: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance one time step.

        Parameters
        ----------
        x : Tensor
            Input at the current time step, shape (batch, in_features).

        membrane : Tensor
            Previous membrane potential, shape (batch, out_features).

        Returns
        -------
        (spikes, membrane)
        """

        current = self.synapse(x)

        return self.neuron(current, membrane)
