"""
Model registry and factory.

Mirrors the pattern used by `encoding/registry.py` +
`encoding/factory.py`, adapted for `nn.Module` architectures.
"""

from __future__ import annotations

import inspect
from typing import Dict, Type

import torch.nn as nn


class ModelRegistry:
    """
    Registry for model architectures.
    """

    def __init__(self) -> None:
        self._models: Dict[str, Type[nn.Module]] = {}

    def register(self, name: str):
        """
        Register a model class, usable as a decorator.
        """

        if not isinstance(name, str):
            raise TypeError("Model name must be a string.")

        name = name.lower().strip()

        def decorator(cls: Type[nn.Module]):

            if not issubclass(cls, nn.Module):
                raise TypeError(f"{cls.__name__} must inherit nn.Module.")

            if name in self._models:
                raise KeyError(f"Model '{name}' is already registered.")

            self._models[name] = cls

            return cls

        return decorator

    def get(self, name: str) -> Type[nn.Module]:
        """
        Retrieve a model class by name.
        """

        name = name.lower().strip()

        try:
            return self._models[name]

        except KeyError:

            available = ", ".join(self.available())

            raise KeyError(
                f"Unknown model '{name}'. Available models: {available}"
            )

    def available(self) -> list[str]:
        """
        Return registered model names.
        """

        return sorted(self._models.keys())

    def exists(self, name: str) -> bool:
        """
        Check whether a model exists.
        """

        return name.lower().strip() in self._models

    def __contains__(self, name: str) -> bool:
        return self.exists(name)

    def __repr__(self) -> str:
        return f"ModelRegistry(models=[{', '.join(self.available())}])"


# Global registry
MODELS = ModelRegistry()


# ---------------------------------------------------------------
# Register built-in architectures.
#
# Imported here (rather than having each module self-register via
# a decorator) so `baselines/cnn1d.py` doesn't need to depend on
# this module, avoiding a models <-> baselines import cycle.
# ---------------------------------------------------------------

from models.networks.lif_classifier import LIFClassifier  # noqa: E402
from baselines.cnn1d import CNN1DBaseline  # noqa: E402

MODELS.register("lif_classifier")(LIFClassifier)
MODELS.register("cnn1d")(CNN1DBaseline)


def build_model(config: dict) -> nn.Module:
    """
    Build a model from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Full experiment config (as loaded from YAML). Only
        `config["model"]` is used; unrecognized keys for the
        selected architecture are silently ignored so a single
        `model:` config block can carry parameters for multiple
        architectures.
    """

    model_cfg = dict(config["model"])

    architecture = model_cfg.pop("architecture")

    cls = MODELS.get(architecture)

    valid_keys = set(inspect.signature(cls.__init__).parameters) - {"self"}

    kwargs = {k: v for k, v in model_cfg.items() if k in valid_keys}

    return cls(**kwargs)


def available_models() -> list[str]:
    """
    Return registered model architectures.
    """

    return MODELS.available()
