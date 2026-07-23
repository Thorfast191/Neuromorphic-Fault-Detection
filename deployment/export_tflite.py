"""
TFLite Micro export (Phase B).

Establishes the interface for exporting a trained model to
TFLite Micro, the realistic deployment path for running inference
on an Arduino-class microcontroller. Not implemented yet - Phase A
only produces the trained PyTorch model and simulated sparsity/
energy metrics; Phase B replaces the estimate with a real measured
on-device deployment.
"""

from __future__ import annotations


class TFLiteExporter:
    """
    Parameters
    ----------
    model : nn.Module
        Trained model to export.

    sample_input_shape : tuple[int, ...]
        Shape of one input sample (excluding batch dimension), used
        to trace the model for export.
    """

    def __init__(self, model, sample_input_shape: tuple[int, ...]):
        self.model = model
        self.sample_input_shape = sample_input_shape

    def export(self, path: str) -> None:
        """
        Export `self.model` to a `.tflite` file at `path`.
        """

        raise NotImplementedError(
            "Phase B: TFLite Micro export for Arduino deployment."
        )

    def quantize(self, calibration_loader) -> None:
        """
        Post-training int8 quantization using `calibration_loader`
        to estimate activation ranges.
        """

        raise NotImplementedError(
            "Phase B: TFLite Micro post-training quantization."
        )
