"""
Dataset utility functions.

Provides helper functions for:

- Finding MAT files
- Loading vibration signals
- Reading metadata
- Validating datasets
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from scipy.io import loadmat


SUPPORTED_CHANNELS = {
    "DE": "DE_time",
    "FE": "FE_time",
    "BA": "BA_time",
}


def find_mat_files(root: str | Path) -> list[Path]:
    """
    Recursively locate all MAT files.

    Parameters
    ----------
    root : str | Path

    Returns
    -------
    list[Path]
    """

    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(root)

    files = sorted(root.rglob("*.mat"))

    if not files:
        raise FileNotFoundError(
            f"No MAT files found in {root}"
        )

    return files


def load_mat(file: str | Path) -> dict:
    """
    Load a MAT file.

    Returns
    -------
    dict
        MATLAB dictionary.
    """

    file = Path(file)

    if not file.exists():
        raise FileNotFoundError(file)

    return loadmat(file)


def load_signal(
    file: str | Path,
    channel: Literal["DE", "FE", "BA"] = "DE",
) -> np.ndarray:
    """
    Load a vibration signal.

    Parameters
    ----------
    file
        MAT file.

    channel
        DE, FE or BA.

    Returns
    -------
    np.ndarray
    """

    channel = channel.upper()

    if channel not in SUPPORTED_CHANNELS:
        raise ValueError(
            f"Unknown channel '{channel}'. "
            f"Supported: {list(SUPPORTED_CHANNELS)}"
        )

    mat = load_mat(file)

    suffix = SUPPORTED_CHANNELS[channel]

    for key, value in mat.items():

        if key.endswith(suffix):

            return value.squeeze().astype(np.float32)

    raise RuntimeError(
        f"{channel} signal not found in {file}"
    )


def available_channels(
    file: str | Path,
) -> list[str]:
    """
    Return channels present in a MAT file.
    """

    mat = load_mat(file)

    channels = []

    for channel, suffix in SUPPORTED_CHANNELS.items():

        for key in mat.keys():

            if key.endswith(suffix):

                channels.append(channel)

    return channels


def signal_length(
    file: str | Path,
    channel: str = "DE",
) -> int:
    """
    Length of vibration signal.
    """

    return len(
        load_signal(file, channel)
    )


def sampling_frequency(
    file: str | Path,
) -> int | None:
    """
    Return sampling frequency if available.

    Some CWRU files do not store this information.
    """

    mat = load_mat(file)

    candidates = (
        "SamplingFrequency",
        "sampling_frequency",
        "fs",
        "Fs",
    )

    for key in candidates:

        if key in mat:

            return int(
                np.squeeze(mat[key])
            )

    return None


def dataset_statistics(
    root: str | Path,
) -> dict:
    """
    Compute simple dataset statistics.
    """

    files = find_mat_files(root)

    return {
        "num_files": len(files),
        "root": str(Path(root).resolve()),
    }


def validate_dataset(
    root: str | Path,
) -> bool:
    """
    Validate dataset structure.

    Raises
    ------
    Exception if invalid.
    """

    files = find_mat_files(root)

    for file in files:

        load_signal(file)

    return True