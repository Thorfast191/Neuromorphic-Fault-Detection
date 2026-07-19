"""
Spike encoding package.
"""

from .factory import (
    create_encoder,
    create_from_config,
    available_encoders,
    encoder_exists,
)

from .registry import ENCODERS

from .base_encoder import BaseEncoder

from .rate import RateEncoder
from .latency import LatencyEncoder
from .temporal import TemporalEncoder
from .population import PopulationEncoder
from .phase import PhaseEncoder
from .delta import DeltaEncoder
from .adaptive import AdaptiveEncoder
from .hybrid import HybridEncoder

from .visualization import SpikeVisualizer

__all__ = [
    "BaseEncoder",
    "RateEncoder",
    "LatencyEncoder",
    "TemporalEncoder",
    "PopulationEncoder",
    "PhaseEncoder",
    "DeltaEncoder",
    "AdaptiveEncoder",
    "HybridEncoder",
    "SpikeVisualizer",
    "ENCODERS",
    "create_encoder",
    "create_from_config",
    "available_encoders",
    "encoder_exists",
]