"""
Leaky Integrate-and-Fire (LIF) neuron.

Thin wrapper around snnTorch's Leaky neuron, kept separate so the
rest of the codebase depends on this module rather than snnTorch
directly.
"""

from __future__ import annotations

import snntorch as snn
import torch
import torch.nn as nn

from models.synapses.surrogate import get_surrogate


def build_lif(
    beta: float = 0.9,
    surrogate_name: str = "fast_sigmoid",
    learn_beta: bool = False,
    threshold: float = 1.0,
    **surrogate_kwargs,
) -> snn.Leaky:
    """
    Construct a snnTorch Leaky neuron.

    Parameters
    ----------
    beta : float
        Membrane decay rate.

    surrogate_name : str
        Name of the surrogate gradient (see
        `models.synapses.surrogate.get_surrogate`).

    learn_beta : bool
        Whether beta is a learnable parameter.

    threshold : float
        Firing threshold.
    """

    spike_grad = get_surrogate(surrogate_name, **surrogate_kwargs)

    return snn.Leaky(
        beta=beta,
        threshold=threshold,
        spike_grad=spike_grad,
        learn_beta=learn_beta,
        init_hidden=False,
    )


class LIFNeuron(nn.Module):
    """
    Explicit LIF neuron with externally managed membrane state.

    Used by `models.layers.spiking_linear.SpikingLinear` so a
    network can hold and reset the state of every neuron layer
    itself instead of relying on snnTorch's hidden-state mode.
    """

    def __init__(
        self,
        beta: float = 0.9,
        surrogate_name: str = "fast_sigmoid",
        learn_beta: bool = False,
        threshold: float = 1.0,
        **surrogate_kwargs,
    ):
        super().__init__()

        self.cell = build_lif(
            beta=beta,
            surrogate_name=surrogate_name,
            learn_beta=learn_beta,
            threshold=threshold,
            **surrogate_kwargs,
        )

    def init_state(
        self,
        batch_size: int,
        features: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Return a zeroed membrane-potential tensor.
        """

        return torch.zeros(
            batch_size,
            features,
            device=device,
        )

    def forward(
        self,
        current: torch.Tensor,
        membrane: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance the neuron by one time step.

        Returns
        -------
        (spikes, membrane)
        """

        return self.cell(current, membrane)
