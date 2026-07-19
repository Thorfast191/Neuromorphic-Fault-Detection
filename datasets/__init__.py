"""
Dataset package.

Provides access to all supported datasets.
"""

from .base_dataset import BaseDataset
from .cwru import CWRUDataset
from .registry import (
    registry,
    get_dataset,
    available_datasets,
)

__all__ = [
    "BaseDataset",
    "CWRUDataset",
    "registry",
    "get_dataset",
    "available_datasets",
]