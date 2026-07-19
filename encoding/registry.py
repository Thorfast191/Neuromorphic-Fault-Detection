"""
Encoder registry.

Provides a central registry for all available spike encoders.
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base_encoder import BaseEncoder


class EncoderRegistry:
    """
    Registry for spike encoders.

    Examples
    --------
    >>> @ENCODERS.register("rate")
    >>> class RateEncoder(BaseEncoder):
    >>>     ...

    >>> encoder = ENCODERS.create("rate")
    """

    def __init__(self) -> None:
        self._encoders: Dict[str, Type[BaseEncoder]] = {}

    # -------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------

    def register(self, name: str):
        """
        Register an encoder class.

        Parameters
        ----------
        name : str
            Unique encoder name.
        """

        if not isinstance(name, str):
            raise TypeError("Encoder name must be a string.")

        name = name.lower().strip()

        def decorator(cls: Type[BaseEncoder]):

            if not issubclass(cls, BaseEncoder):
                raise TypeError(
                    f"{cls.__name__} must inherit BaseEncoder."
                )

            if name in self._encoders:
                raise KeyError(
                    f"Encoder '{name}' is already registered."
                )

            self._encoders[name] = cls

            return cls

        return decorator

    # -------------------------------------------------------------
    # Lookup
    # -------------------------------------------------------------

    def get(self, name: str) -> Type[BaseEncoder]:
        """
        Retrieve encoder class by name.
        """

        name = name.lower().strip()

        try:
            return self._encoders[name]

        except KeyError:

            available = ", ".join(self.available())

            raise KeyError(
                f"Unknown encoder '{name}'. "
                f"Available encoders: {available}"
            )

    # -------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------

    def create(
        self,
        name: str,
        **kwargs,
    ) -> BaseEncoder:
        """
        Instantiate an encoder.
        """

        cls = self.get(name)

        return cls(**kwargs)

    # -------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------

    def exists(self, name: str) -> bool:
        """
        Check whether an encoder exists.
        """

        return name.lower().strip() in self._encoders

    def available(self) -> List[str]:
        """
        Return available encoder names.
        """

        return sorted(self._encoders.keys())

    def clear(self) -> None:
        """
        Remove every registered encoder.

        Mainly useful for testing.
        """

        self._encoders.clear()

    # -------------------------------------------------------------
    # Magic methods
    # -------------------------------------------------------------

    def __contains__(self, name: str) -> bool:

        return self.exists(name)

    def __len__(self) -> int:

        return len(self._encoders)

    def __iter__(self):

        return iter(self.available())

    def __repr__(self) -> str:

        encoders = ", ".join(self.available())

        return (
            f"{self.__class__.__name__}"
            f"(encoders=[{encoders}])"
        )


# Global registry
ENCODERS = EncoderRegistry()