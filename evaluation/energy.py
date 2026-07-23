"""
Sparsity and energy-proxy comparison between the event-driven
LIFClassifier and a dense baseline (e.g. CNN1DBaseline).

The energy figures here are literature-derived per-operation
estimates, NOT hardware measurements. `from_hardware_measurement`
is the explicit Phase B entry point for replacing them with real
measured Arduino/TFLite-Micro power numbers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sparsity(spike_tensor: torch.Tensor) -> float:
    """
    Fraction of inactive (non-firing) entries in a spike tensor.
    """

    return 1.0 - float(spike_tensor.float().mean())


def count_esops(
    model: nn.Module,
    input_spikes: torch.Tensor,
    hidden_spikes: list[torch.Tensor],
) -> dict:
    """
    Effective Synaptic OPerations (ESOPs): for a fully-connected
    spiking layer, each presynaptic spike triggers one accumulate
    per postsynaptic neuron, i.e. ESOPs = spike_count * fan_out.

    Parameters
    ----------
    model : LIFClassifier
        Must expose `.layers` (ModuleList of SpikingLinear).

    input_spikes : Tensor
        The network's input spike train, shape (B, T, input_size).

    hidden_spikes : list[Tensor]
        Per-hidden-layer spikes from `LIFClassifier(..., return_all=True)`,
        each shape (T, B, layer_size). Does not include the output
        layer's spikes (nothing consumes them downstream).
    """

    driving_spikes = [input_spikes, *hidden_spikes]
    fanouts = [layer.synapse.out_features for layer in model.layers]

    if len(driving_spikes) != len(fanouts):
        raise ValueError(
            "Number of spike tensors does not match number of "
            "synaptic layers in the model."
        )

    batch_size = input_spikes.shape[0]

    per_layer = [
        float(spikes.sum()) * fanout / batch_size
        for spikes, fanout in zip(driving_spikes, fanouts)
    ]

    return {
        "per_layer_esops": per_layer,
        "total_esops": sum(per_layer),
    }


def dense_macs(model: nn.Module, sample_input: torch.Tensor) -> float:
    """
    Multiply-accumulate count for a dense model (e.g. CNN1DBaseline)
    on one forward pass, computed via forward hooks so it stays
    correct regardless of the exact architecture/window size.
    """

    macs = 0.0
    hooks = []

    def conv_hook(module, _inputs, output):
        nonlocal macs
        out_length = output.shape[-1]
        macs += (
            module.out_channels
            * module.in_channels
            * module.kernel_size[0]
            * out_length
        )

    def linear_hook(module, _inputs, output):
        nonlocal macs
        macs += module.in_features * module.out_features * output.shape[0]

    for module in model.modules():

        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv_hook))

        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model.eval()

    with torch.no_grad():
        model(sample_input)

    for hook in hooks:
        hook.remove()

    return macs / sample_input.shape[0]


def energy_estimate(
    esops: float,
    macs: float,
    e_ac_pj: float = 0.9,
    e_mac_pj: float = 4.6,
) -> dict:
    """
    Convert ESOP/MAC counts into nanojoule energy ESTIMATES for
    relative comparison.

    Defaults (e_ac_pj=0.9, e_mac_pj=4.6) are widely-cited 45nm
    digital-ASIC per-operation energy figures (e.g. Horowitz,
    ISSCC 2014: ~0.9 pJ per 32-bit accumulate, ~4.6 pJ per 32-bit
    MAC) commonly used in the SNN-efficiency literature as a
    relative proxy. These are NOT hardware measurements.
    """

    esop_energy_nj = esops * e_ac_pj / 1000.0
    mac_energy_nj = macs * e_mac_pj / 1000.0

    return {
        "esop_energy_nj": esop_energy_nj,
        "mac_energy_nj": mac_energy_nj,
        "energy_ratio_dense_over_sparse": (
            mac_energy_nj / esop_energy_nj
            if esop_energy_nj > 0
            else float("inf")
        ),
    }


def from_hardware_measurement(joules: float, num_inferences: int) -> dict:
    """
    Phase B entry point: replace `energy_estimate`'s literature
    proxy with real measured energy (e.g. from an INA219 current
    sensor on an Arduino running the TFLite-Micro-exported model).
    """

    raise NotImplementedError(
        "Phase B: real hardware energy measurement not yet implemented. "
        "See deployment/export_tflite.py."
    )
