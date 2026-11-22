"""Implementation: exception hierarchy."""

from .exceptions import (
    ConfigurationError,
    DataValidationError,
    DependencyMissingError,
    EncodingError,
    EvaluationError,
    OptimizationError,
    RegistryError,
    ReportError,
    Sensor2SpikeError,
    format_error_chain,
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
    "format_error_chain",
]
