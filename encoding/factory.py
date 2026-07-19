"""
Encoder factory.

Provides convenient utilities for creating spike encoders
from names or configuration dictionaries.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .base_encoder import BaseEncoder
from .registry import ENCODERS


def create_encoder(
    name: str,
    **kwargs,
) -> BaseEncoder:
    """
    Create an encoder by name.

    Parameters
    ----------
    name : str
        Registered encoder name.

    **kwargs
        Encoder parameters.

    Returns
    -------
    BaseEncoder
    """

    return ENCODERS.create(name, **kwargs)


def create_from_config(
    config: dict[str, Any],
) -> BaseEncoder:
    """
    Create an encoder from a configuration dictionary.

    Example
    -------
    config = {
        "name": "rate",
        "time_steps": 100,
        "seed": 42
    }

    encoder = create_from_config(config)
    """

    if not isinstance(config, dict):
        raise TypeError(
            "config must be a dictionary."
        )

    cfg = deepcopy(config)

    if "name" not in cfg:
        raise KeyError(
            "Configuration requires a 'name' field."
        )

    name = cfg.pop("name")

    return create_encoder(
        name,
        **cfg,
    )


def available_encoders() -> list[str]:
    """
    Return registered encoders.
    """

    return ENCODERS.available()


def encoder_exists(
    name: str,
) -> bool:
    """
    Check whether an encoder exists.
    """

    return ENCODERS.exists(name)