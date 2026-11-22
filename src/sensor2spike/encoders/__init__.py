"""Spike encoding engine.

The registry is populated by importing this package.  Use
:func:`build_encoder` to instantiate a strategy by name::

    from sensor2spike.encoders import build_encoder

    encoder = build_encoder("latency", n_neurons=16, sigma=0.2)
    spikes = encoder.fit_transform(signal, fs=1000)
"""

from __future__ import annotations

from .base import (
    SpikeEncoder,
    encode_windows,
    exp_smooth,
    reconstruction_r2,
    resample_to_steps,
)
from .delta import DeltaEncoder
from .latency import LatencyEncoder
from .population import PopulationEncoder
from .rate import RateEncoder
from .registry import (
    ENCODERS,
    build_encoder,
    encoder_space,
    get_encoder_class,
    list_encoders,
    register_encoder,
    validate_encoder_config,
)
from .threshold import ThresholdEncoder

__all__ = [
    "SpikeEncoder",
    "RateEncoder",
    "LatencyEncoder",
    "ThresholdEncoder",
    "DeltaEncoder",
    "PopulationEncoder",
    "ENCODERS",
    "build_encoder",
    "encoder_space",
    "get_encoder_class",
    "list_encoders",
    "register_encoder",
    "validate_encoder_config",
    "encode_windows",
    "exp_smooth",
    "reconstruction_r2",
    "resample_to_steps",
]
