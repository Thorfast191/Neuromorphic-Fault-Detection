"""
Dataset registry.

Provides a centralized registry for all supported datasets.
"""

from __future__ import annotations

from typing import Dict, Type

from .base_dataset import BaseDataset
from .cwru import CWRUDataset


class DatasetRegistry:
    """
    Registry for dataset classes.
    """

    def __init__(self):
        self._datasets: Dict[str, Type[BaseDataset]] = {}

    def register(
        self,
        name: str,
        dataset_cls: Type[BaseDataset],
    ) -> None:
        """
        Register a dataset class.
        """

        name = name.lower()

        if name in self._datasets:
            raise ValueError(
                f"Dataset '{name}' is already registered."
            )

        self._datasets[name] = dataset_cls

    def get(
        self,
        name: str,
    ) -> Type[BaseDataset]:
        """
        Retrieve a dataset class.
        """

        name = name.lower()

        if name not in self._datasets:
            available = ", ".join(self.available())
            raise KeyError(
                f"Unknown dataset '{name}'. "
                f"Available datasets: {available}"
            )

        return self._datasets[name]

    def available(self) -> list[str]:
        """
        Return registered dataset names.
        """

        return sorted(self._datasets.keys())

    def exists(self, name: str) -> bool:
        """
        Check whether a dataset exists.
        """

        return name.lower() in self._datasets

    def __contains__(self, name: str) -> bool:
        return self.exists(name)

    def __len__(self) -> int:
        return len(self._datasets)

    def __repr__(self) -> str:
        return (
            f"DatasetRegistry("
            f"{self.available()})"
        )


# ---------------------------------------------------
# Global registry
# ---------------------------------------------------

registry = DatasetRegistry()

registry.register(
    "cwru",
    CWRUDataset,
)

# Future datasets
#
# registry.register("mfpt", MFPTDataset)
# registry.register("ims", IMSDataset)
# registry.register("paderborn", PaderbornDataset)


def get_dataset(name: str):
    """
    Return a dataset class.
    """

    return registry.get(name)


def available_datasets() -> list[str]:
    """
    Return all registered datasets.
    """

    return registry.available()