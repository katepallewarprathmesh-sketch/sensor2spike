"""Internal: the exception hierarchy.

Not part of the public API. Import from ``sensor2spike`` instead.
"""

from __future__ import annotations

from ._core._errors import (
    ConfigurationError,
    DataValidationError,
    DependencyMissingError,
    EncodingError,
    EvaluationError,
    OptimizationError,
    RegistryError,
    ReportError,
    Sensor2SpikeError,
)

__all__ = [
    "Sensor2SpikeError",
    "DataValidationError",
    "ConfigurationError",
    "EncodingError",
    "EvaluationError",
    "OptimizationError",
    "RegistryError",
    "ReportError",
    "DependencyMissingError",
]
