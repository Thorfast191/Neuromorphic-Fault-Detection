"""
LIF-based spiking classifier.

Consumes a spike train (e.g. delta-encoded ON/OFF events) and
produces per-class output spikes over time, trained with
surrogate-gradient backpropagation-through-time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.layers.spiking_linear import SpikingLinear


class LIFClassifier(nn.Module):
    """
    Stack of spiking linear layers.

    Parameters
    ----------
    input_size : int
        Number of input channels per time step
        (e.g. 2 for delta ON/OFF encoding).

    hidden_sizes : list[int]
        Sizes of hidden spiking layers.

    num_classes : int
        Number of output classes.

    beta : float
        LIF membrane decay rate, shared by all layers.

    surrogate_name : str
        Surrogate gradient name (see
        `models.synapses.surrogate.get_surrogate`).
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_sizes: list[int] | None = None,
        num_classes: int = 10,
        beta: float = 0.9,
        surrogate_name: str = "fast_sigmoid",
        **surrogate_kwargs,
    ):
        super().__init__()

        hidden_sizes = hidden_sizes or [128]

        sizes = [input_size, *hidden_sizes, num_classes]

        self.layers = nn.ModuleList(
            [
                SpikingLinear(
                    sizes[i],
                    sizes[i + 1],
                    beta=beta,
                    surrogate_name=surrogate_name,
                    **surrogate_kwargs,
                )
                for i in range(len(sizes) - 1)
            ]
        )

        self.num_classes = num_classes

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Parameters
        ----------
        x : Tensor
            Shape (batch, time_steps, input_size).

        return_all : bool
            If True, also return per-time-step spikes for every
            hidden layer (used by explainability/energy analysis).
            Disabled by default to save memory during training.

        Returns
        -------
        dict with keys:
            "spikes"       : (time_steps, batch, num_classes)
            "membrane"     : (time_steps, batch, num_classes)
            "hidden_spikes": list of (time_steps, batch, layer_size),
                             only present when return_all=True.
        """

        batch_size, time_steps, _ = x.shape
        device = x.device

        x = x.permute(1, 0, 2)  # (T, B, input_size)

        membranes = [
            layer.init_state(batch_size, device)
            for layer in self.layers
        ]

        output_spikes = []
        output_membranes = []
        hidden_spike_records = (
            [[] for _ in self.layers[:-1]] if return_all else None
        )

        for t in range(time_steps):

            layer_input = x[t]

            for i, layer in enumerate(self.layers):

                spk, membranes[i] = layer(layer_input, membranes[i])

                layer_input = spk

                if return_all and i < len(self.layers) - 1:
                    hidden_spike_records[i].append(spk)

            output_spikes.append(layer_input)
            output_membranes.append(membranes[-1])

        result = {
            "spikes": torch.stack(output_spikes, dim=0),
            "membrane": torch.stack(output_membranes, dim=0),
        }

        if return_all:
            result["hidden_spikes"] = [
                torch.stack(records, dim=0)
                for records in hidden_spike_records
            ]

        return result
