"""
Composable preprocessing pipeline.
"""

from __future__ import annotations


class Pipeline:

    def __init__(self, transforms):

        self.transforms = transforms

    def __call__(self, signal):

        for transform in self.transforms:

            signal = transform(signal)

        return signal