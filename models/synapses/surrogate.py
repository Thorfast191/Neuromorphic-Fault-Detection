"""
Surrogate gradient factory.

snnTorch's spiking nonlinearity has zero gradient almost everywhere,
so training requires a surrogate gradient substitute during the
backward pass. This module centralizes the mapping from config
names to snnTorch surrogate functions.
"""

from __future__ import annotations

from typing import Callable

import snntorch.surrogate as surrogate

_SURROGATES = {
    "fast_sigmoid": surrogate.fast_sigmoid,
    "atan": surrogate.atan,
    "sigmoid": surrogate.sigmoid,
}


def get_surrogate(name: str = "fast_sigmoid", **kwargs) -> Callable:
    """
    Return a surrogate gradient function.

    Parameters
    ----------
    name : str
        One of "fast_sigmoid", "atan", "sigmoid".

    **kwargs
        Forwarded to the underlying snnTorch surrogate factory
        (e.g. `slope` for "fast_sigmoid").
    """

    name = name.lower().strip()

    if name not in _SURROGATES:
        available = ", ".join(sorted(_SURROGATES))
        raise KeyError(
            f"Unknown surrogate '{name}'. Available: {available}"
        )

    return _SURROGATES[name](**kwargs)
