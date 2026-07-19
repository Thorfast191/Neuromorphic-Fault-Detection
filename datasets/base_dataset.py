"""
Abstract base class for all datasets.

Provides a common interface for dataset indexing,
statistics, and PyTorch integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Base class for all datasets.

    Subclasses are responsible for implementing:
        - prepare()
        - __getitem__()
    """

    def __init__(self, root: str | Path):

        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root}"
            )

        # Data containers
        self.samples = []
        self.labels = []

        # Optional metadata
        self.metadata = []

    @abstractmethod
    def prepare(self) -> None:
        """
        Load dataset metadata and prepare samples.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return one sample.

        Returns
        -------
        (sample, label)
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def num_samples(self) -> int:
        return len(self)

    @property
    def num_classes(self) -> int:

        return len(
            set(self.labels)
        )

    @property
    def class_distribution(self) -> dict[int, int]:

        distribution = {}

        for label in self.labels:

            label = int(label)

            distribution[label] = (
                distribution.get(label, 0) + 1
            )

        return distribution

    def summary(self) -> dict[str, Any]:
        """
        Dataset summary.
        """

        return {
            "dataset": self.__class__.__name__,
            "root": str(self.root),
            "samples": self.num_samples,
            "classes": self.num_classes,
            "distribution": self.class_distribution,
        }

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"samples={self.num_samples}, "
            f"classes={self.num_classes})"
        )