"""Exception hierarchy for Sensor2Spike.

Every error raised by the library derives from :class:`Sensor2SpikeError` so
callers can catch a single base class.  The sub-classes map onto the layer that
produced the error, which keeps error reporting in the CLI honest about *where*
something failed.
"""

from __future__ import annotations

from collections.abc import Iterable


class Sensor2SpikeError(Exception):
    """Base class for all Sensor2Spike errors."""


class DataValidationError(Sensor2SpikeError, ValueError):
    """Raised when a dataset, signal or spike array violates its contract."""


class ConfigurationError(Sensor2SpikeError, ValueError):
    """Raised when a configuration object or mapping is invalid."""


class EncodingError(Sensor2SpikeError, RuntimeError):
    """Raised when an encoder cannot transform a signal."""


class EvaluationError(Sensor2SpikeError, RuntimeError):
    """Raised when an SNN backend cannot evaluate a spike train."""


class OptimizationError(Sensor2SpikeError, RuntimeError):
    """Raised when the optimisation process cannot be started or completed."""


class RegistryError(Sensor2SpikeError, RuntimeError):
    """Raised when the experiment registry cannot be read or written."""


class ReportError(Sensor2SpikeError, RuntimeError):
    """Raised when a report cannot be generated."""


class DependencyMissingError(Sensor2SpikeError, ImportError):
    """Raised when an optional dependency is required but not installed."""

    def __init__(self, package: str, *, extra: str | None = None, purpose: str = "") -> None:
        self.package = package
        self.extra = extra
        self.purpose = purpose
        hint = f" Install it with `pip install sensor2spike[{extra}]`." if extra else ""
        suffix = f" It is required for {purpose}." if purpose else ""
        super().__init__(f"Optional dependency '{package}' is not installed.{suffix}{hint}")


def format_error_chain(exc: BaseException, *, limit: int = 3) -> str:
    """Return a compact single-line description of an exception chain.

    Local filesystem paths are redacted by the CLI before this text is shown to
    a user (see :mod:`sensor2spike.utils.security`).
    """
    parts: list[str] = []
    current: BaseException | None = exc
    seen = 0
    while current is not None and seen < limit:
        text = str(current).strip()
        parts.append(f"{type(current).__name__}: {text}" if text else type(current).__name__)
        current = current.__cause__ or current.__context__
        seen += 1
    return " <- ".join(parts)


__all__: Iterable[str] = [
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
