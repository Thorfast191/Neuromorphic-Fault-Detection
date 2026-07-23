"""
Dense 1D-CNN baseline.

Operates directly on the raw (normalized) vibration window rather
than a spike-encoded input. Used as the dense-compute reference
point for the energy/sparsity comparison against the event-driven
LIFClassifier (see evaluation/energy.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNN1DBaseline(nn.Module):
    """
    Small 1D-CNN classifier for raw vibration windows.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    in_channels : int
        Number of input channels (1 for a single vibration channel).
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape (batch, in_channels, window_size).

        Returns
        -------
        Tensor
            Class logits, shape (batch, num_classes).
        """

        x = self.features(x)
        x = x.flatten(1)

        return self.classifier(x)
