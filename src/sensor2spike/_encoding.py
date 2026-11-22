"""Internal: encoder contract, implementations and registry.

Not part of the public API. Import from ``sensor2spike`` instead.
"""

from __future__ import annotations

from ._core._spaces import Parameter, SearchSpace
from ._core.encoders import (
    DeltaEncoder,
    LatencyEncoder,
    PopulationEncoder,
    RateEncoder,
    SpikeEncoder,
    ThresholdEncoder,
    build_encoder,
    encoder_space,
    list_encoders,
    register_encoder,
)

__all__ = [
    "SpikeEncoder",
    "RateEncoder",
    "LatencyEncoder",
    "ThresholdEncoder",
    "DeltaEncoder",
    "PopulationEncoder",
    "build_encoder",
    "encoder_space",
    "list_encoders",
    "register_encoder",
    "Parameter",
    "SearchSpace",
]
