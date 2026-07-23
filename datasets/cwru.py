"""
Case Western Reserve University (CWRU) Bearing Dataset.

Features
--------
- Automatic MAT file discovery
- Channel selection (DE, FE, BA)
- Sliding-window generation
- Optional transforms
- PyTorch Dataset compatible
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Callable

import numpy as np

from .base_dataset import BaseDataset
from .labels import filename_to_label
from .utils import find_mat_files, load_signal
from preprocessing.windowing import SlidingWindow


class CWRUDataset(BaseDataset):
    """
    Parameters
    ----------
    root : str | Path
        Root directory containing MAT files.

    window_size : int
        Window length.

    overlap : float
        Window overlap ratio.

    channel : str
        DE, FE or BA.

    transform : callable, optional
        Transform applied to each sample.
    """

    def __init__(
        self,
        root: str | Path,
        window_size: int = 2048,
        overlap: float = 0.5,
        channel: str = "DE",
        transform: Callable | None = None,
    ):

        super().__init__(root)

        self.window_size = window_size
        self.overlap = overlap
        self.channel = channel.upper()
        self.transform = transform

        self.window_generator = SlidingWindow(
            window_size=self.window_size,
            overlap=self.overlap,
        )

        self.prepare()

    def prepare(self) -> None:
        """
        Load every MAT file and create windows.
        """

        files = find_mat_files(self.root)

        if not files:
            raise FileNotFoundError(
                f"No .mat files found in {self.root}"
            )

        for file in files:

            signal = load_signal(
                file,
                channel=self.channel,
            )

            windows = self.window_generator(signal)

            label = filename_to_label(file)

            self.samples.extend(windows)
            self.labels.extend(
                [label] * len(windows)
            )

        self.samples = np.asarray(
            self.samples,
            dtype=np.float32,
        )

        self.labels = np.asarray(
            self.labels,
            dtype=np.int64,
        )

    @property
    def num_classes(self) -> int:
        return len(np.unique(self.labels))

    @property
    def class_distribution(self):

        return dict(
            Counter(self.labels.tolist())
        )

    def summary(self) -> dict:
        """
        Return dataset statistics.
        """

        return {

            "dataset": "CWRU",

            "samples": len(self),

            "classes": self.num_classes,

            "window_size": self.window_size,

            "overlap": self.overlap,

            "channel": self.channel,

            "distribution":
                self.class_distribution,
        }

    def __getitem__(self, index):

        x = self.samples[index]
        y = self.labels[index]

        if self.transform is not None:

            x = self.transform(x)

        return x, y

    def __repr__(self):

        return (
            f"CWRUDataset("
            f"samples={len(self)}, "
            f"classes={self.num_classes}, "
            f"channel='{self.channel}')"
        )