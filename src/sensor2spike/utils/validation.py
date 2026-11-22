"""Dependency-free validation helpers.

These functions are the single place where user-supplied values are coerced and
checked.  Keeping them here means every module reports errors in the same shape
and with the same wording.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ..exceptions import ConfigurationError, DataValidationError

__all__ = [
    "as_1d_float_array",
    "as_1d_int_array",
    "as_2d_float_array",
    "reject_unknown_keys",
    "require_choice",
    "require_finite",
    "require_keys",
    "require_number",
    "require_positive",
    "require_range",
]


def as_2d_float_array(
    value: Any,
    *,
    name: str = "data",
    dtype: Any = np.float32,
    allow_empty: bool = False,
) -> np.ndarray:
    """Coerce ``value`` to a contiguous, finite 2-D float array.

    A 1-D input is treated as a single-channel signal (``shape=(n, 1)``).
    """
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"{name} could not be converted to a numeric array: {exc}"
        ) from exc
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise DataValidationError(
            f"{name} must be a 2-D array of shape (n_samples, n_channels); "
            f"got {array.ndim} dimensions with shape {array.shape}."
        )
    if array.size == 0 and not allow_empty:
        raise DataValidationError(f"{name} is empty; expected at least one sample.")
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        first = np.argwhere(~np.isfinite(array))[0]
        raise DataValidationError(
            f"{name} contains {bad} non-finite value(s) (NaN/inf); "
            f"first occurrence at index (sample={first[0]}, channel={first[1]}). "
            "Use the preprocessing stage or a missing-value policy to repair the input."
        )
    return np.ascontiguousarray(array, dtype=dtype)


def as_1d_float_array(
    value: Any, *, name: str = "values", allow_empty: bool = False
) -> np.ndarray:
    """Coerce ``value`` to a finite 1-D float64 array."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"{name} could not be converted to a numeric array: {exc}"
        ) from exc
    if array.ndim != 1:
        raise DataValidationError(
            f"{name} must be 1-D; got shape {array.shape}."
        )
    if array.size == 0 and not allow_empty:
        raise DataValidationError(f"{name} is empty; expected at least one value.")
    if not np.all(np.isfinite(array)):
        raise DataValidationError(f"{name} contains non-finite values (NaN/inf).")
    return np.ascontiguousarray(array)


def as_1d_int_array(
    value: Any, *, name: str = "values", allow_empty: bool = False, minimum: int | None = None
) -> np.ndarray:
    """Coerce ``value`` to a 1-D int64 array, optionally enforcing a minimum."""
    try:
        array = np.asarray(value, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"{name} could not be converted to an integer array: {exc}"
        ) from exc
    if array.ndim != 1:
        raise DataValidationError(f"{name} must be 1-D; got shape {array.shape}.")
    if array.size == 0 and not allow_empty:
        raise DataValidationError(f"{name} is empty; expected at least one value.")
    if minimum is not None and array.size and int(array.min()) < minimum:
        raise DataValidationError(
            f"{name} must contain values >= {minimum}; found {int(array.min())}."
        )
    return np.ascontiguousarray(array, dtype=np.int64)


def require_number(value: Any, name: str, *, allow_none: bool = False) -> float:
    if value is None:
        if allow_none:
            return 0.0
        raise ConfigurationError(f"{name} must not be None.")
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ConfigurationError(f"{name} must be a number; got {type(value).__name__}.")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f"{name} must be finite; got {value!r}.")
    return number


def require_positive(value: Any, name: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    number = require_number(value, name)
    if number <= 0:
        raise ConfigurationError(f"{name} must be > 0; got {number}.")
    return number


def require_range(
    value: Any,
    name: str,
    *,
    low: float | None = None,
    high: float | None = None,
) -> float:
    number = require_number(value, name)
    if low is not None and number < low:
        raise ConfigurationError(f"{name} must be >= {low}; got {number}.")
    if high is not None and number > high:
        raise ConfigurationError(f"{name} must be <= {high}; got {number}.")
    return number


def require_choice(value: Any, name: str, choices: Sequence[Any]) -> Any:
    if value not in tuple(choices):
        pretty = ", ".join(repr(choice) for choice in choices)
        raise ConfigurationError(f"{name} must be one of [{pretty}]; got {value!r}.")
    return value


def require_keys(config: Mapping[str, Any], required: Iterable[str], *, name: str = "config") -> None:
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigurationError(f"{name} is missing required field(s): {', '.join(sorted(missing))}.")


def reject_unknown_keys(
    config: Mapping[str, Any], allowed: Iterable[str], *, name: str = "config"
) -> None:
    allowed_set = set(allowed)
    unknown = sorted(key for key in config if key not in allowed_set)
    if unknown:
        raise ConfigurationError(
            f"{name} contains unknown field(s): {', '.join(unknown)}. "
            f"Allowed fields: {', '.join(sorted(allowed_set))}."
        )


def require_finite(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise DataValidationError(f"{name} contains non-finite values.")
    return array
