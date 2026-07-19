"""
Sliding window utilities.

Provides efficient sliding-window generation for
1D vibration signals.
"""

from __future__ import annotations

import numpy as np


def sliding_window(
    signal: np.ndarray,
    window_size: int,
    overlap: float = 0.5,
    copy: bool = True,
) -> np.ndarray:
    """
    Generate overlapping windows from a 1D signal.

    Parameters
    ----------
    signal : np.ndarray
        Input vibration signal.

    window_size : int
        Number of samples per window.

    overlap : float, default=0.5
        Fractional overlap between consecutive windows.
        Must satisfy 0 <= overlap < 1.

    copy : bool, default=True
        Return independent copies of the windows.
        If False, returns a memory-efficient view.

    Returns
    -------
    np.ndarray
        Array of shape (n_windows, window_size)

    Examples
    --------
    >>> x = np.arange(10)
    >>> sliding_window(x, 4, overlap=0.5)

    array([
        [0,1,2,3],
        [2,3,4,5],
        [4,5,6,7],
        [6,7,8,9]
    ])
    """

    signal = np.asarray(signal)

    if signal.ndim != 1:
        raise ValueError(
            "Input signal must be one-dimensional."
        )

    if window_size <= 0:
        raise ValueError(
            "window_size must be > 0."
        )

    if window_size > len(signal):
        raise ValueError(
            "window_size cannot exceed signal length."
        )

    if not (0 <= overlap < 1):
        raise ValueError(
            "overlap must satisfy 0 <= overlap < 1."
        )

    step = max(
        1,
        int(window_size * (1 - overlap))
    )

    n_windows = (
        (len(signal) - window_size) // step
    ) + 1

    shape = (
        n_windows,
        window_size,
    )

    strides = (
        signal.strides[0] * step,
        signal.strides[0],
    )

    windows = np.lib.stride_tricks.as_strided(
        signal,
        shape=shape,
        strides=strides,
    )

    return windows.copy() if copy else windows


def num_windows(
    signal_length: int,
    window_size: int,
    overlap: float = 0.5,
) -> int:
    """
    Compute number of windows without
    actually generating them.
    """

    if signal_length < window_size:
        return 0

    step = max(
        1,
        int(window_size * (1 - overlap))
    )

    return (
        (signal_length - window_size)
        // step
    ) + 1