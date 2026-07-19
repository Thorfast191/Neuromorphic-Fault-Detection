"""
Bearing characteristic frequency calculations.

References
----------
Randall, R. B. (2011)
Vibration-based Condition Monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BearingGeometry:
    """
    Bearing geometry parameters.

    Parameters
    ----------
    n_balls : int
        Number of rolling elements.

    ball_diameter : float
        Ball diameter (mm or m).

    pitch_diameter : float
        Pitch diameter (same unit as ball diameter).

    contact_angle : float
        Contact angle in degrees.
    """

    n_balls: int
    ball_diameter: float
    pitch_diameter: float
    contact_angle: float = 0.0


class BearingFrequency:
    """
    Bearing fault frequency calculator.
    """

    def __init__(self, geometry: BearingGeometry):

        self.geometry = geometry

    @property
    def _ratio(self) -> float:
        """
        d / D
        """

        return (
            self.geometry.ball_diameter
            / self.geometry.pitch_diameter
        )

    @property
    def _cos_theta(self) -> float:
        from math import cos, radians

        return cos(
            radians(self.geometry.contact_angle)
        )

    def ftf(
        self,
        shaft_frequency: float,
    ) -> float:
        """
        Fundamental Train Frequency (FTF)
        """

        return (
            shaft_frequency
            / 2
            * (
                1
                - self._ratio * self._cos_theta
            )
        )

    def bpfo(
        self,
        shaft_frequency: float,
    ) -> float:
        """
        Ball Pass Frequency Outer Race.
        """

        return (
            self.geometry.n_balls
            * shaft_frequency
            / 2
            * (
                1
                - self._ratio * self._cos_theta
            )
        )

    def bpfi(
        self,
        shaft_frequency: float,
    ) -> float:
        """
        Ball Pass Frequency Inner Race.
        """

        return (
            self.geometry.n_balls
            * shaft_frequency
            / 2
            * (
                1
                + self._ratio * self._cos_theta
            )
        )

    def bsf(
        self,
        shaft_frequency: float,
    ) -> float:
        """
        Ball Spin Frequency.
        """

        return (
            self.geometry.pitch_diameter
            / (
                2
                * self.geometry.ball_diameter
            )
            * shaft_frequency
            * (
                1
                - (
                    self._ratio
                    * self._cos_theta
                )
                ** 2
            )
        )

    def calculate(
        self,
        shaft_frequency: float,
    ) -> dict[str, float]:
        """
        Compute all characteristic frequencies.
        """

        return {
            "FTF": self.ftf(shaft_frequency),
            "BPFO": self.bpfo(shaft_frequency),
            "BPFI": self.bpfi(shaft_frequency),
            "BSF": self.bsf(shaft_frequency),
        }