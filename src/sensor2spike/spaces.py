"""Parameter spaces used by the optimisation engine.

A :class:`Parameter` describes one dimension of an encoder's search space.
:class:`SearchSpace` is an ordered collection of parameters that can be sampled
randomly, enumerated as a grid, or validated against a concrete configuration.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .exceptions import ConfigurationError

__all__ = ["Parameter", "SearchSpace", "ParameterKind"]

ParameterKind = str


def _require_bounds(low: float | None, high: float | None, name: str) -> None:
    if low is None or high is None:
        raise ConfigurationError(f"Parameter {name!r} requires both 'low' and 'high'.")
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ConfigurationError(f"Parameter {name!r} has non-finite bounds.")
    if low >= high:
        raise ConfigurationError(
            f"Parameter {name!r} must have low < high; got low={low}, high={high}."
        )


@dataclass(frozen=True)
class Parameter:
    """One searchable dimension."""

    name: str
    kind: ParameterKind
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: tuple[Any, ...] = field(default_factory=tuple)
    description: str = ""
    default: Any = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigurationError("Parameter name must not be empty.")
        if self.kind not in {"float", "int", "choice"}:
            raise ConfigurationError(
                f"Parameter {self.name!r} has unknown kind {self.kind!r}; "
                "expected 'float', 'int' or 'choice'."
            )
        if self.kind in {"float", "int"}:
            _require_bounds(self.low, self.high, self.name)
            if self.log and (self.low <= 0 or self.high <= 0):
                raise ConfigurationError(
                    f"Parameter {self.name!r} must be positive for log sampling."
                )
            if self.kind == "int":
                if int(self.low) != self.low or int(self.high) != self.high:
                    raise ConfigurationError(
                        f"Integer parameter {self.name!r} must have integer bounds."
                    )
        else:
            if not self.choices:
                raise ConfigurationError(
                    f"Choice parameter {self.name!r} requires a non-empty 'choices' tuple."
                )
            if len(set(self.choices)) != len(self.choices):
                raise ConfigurationError(
                    f"Choice parameter {self.name!r} contains duplicate values."
                )
        if self.default is not None and not self.contains(self.default):
            raise ConfigurationError(
                f"Default {self.default!r} for parameter {self.name!r} is outside its domain."
            )

    # ------------------------------------------------------------------ sampling
    def sample(self, rng: np.random.Generator) -> Any:
        if self.kind == "float":
            assert self.low is not None and self.high is not None
            if self.log:
                value = float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
            else:
                value = float(rng.uniform(self.low, self.high))
            return float(np.clip(value, self.low, self.high))
        if self.kind == "int":
            assert self.low is not None and self.high is not None
            return int(rng.integers(int(self.low), int(self.high) + 1))
        return self.choices[int(rng.integers(0, len(self.choices)))]

    def grid(self, n_points: int = 3) -> list[Any]:
        """Return a deterministic grid of values for this parameter."""
        if n_points < 1:
            raise ConfigurationError("n_points must be >= 1.")
        if self.kind == "choice":
            return list(self.choices)
        assert self.low is not None and self.high is not None
        if self.kind == "int":
            raw = np.linspace(int(self.low), int(self.high), num=min(n_points, int(self.high - self.low) + 1))
            return sorted({int(round(value)) for value in raw})
        if self.log:
            return [float(value) for value in np.geomspace(self.low, self.high, num=n_points)]
        return [float(value) for value in np.linspace(self.low, self.high, num=n_points)]

    def contains(self, value: Any) -> bool:
        try:
            if self.kind == "choice":
                return value in self.choices
            number = float(value)
        except (TypeError, ValueError):
            return False
        if self.low is not None and number < self.low - 1e-12:
            return False
        if self.high is not None and number > self.high + 1e-12:
            return False
        return True

    def coerce(self, value: Any) -> Any:
        """Cast and validate ``value`` for this parameter.

        Strings that were coerced from the CLI or JSON are accepted when they
        unambiguously name a numeric choice (``"2.0"`` -> ``2.0``).
        """
        if self.kind == "choice":
            if value in self.choices:
                return value
            for candidate in self.choices:
                if not isinstance(candidate, (int, float)) or isinstance(candidate, bool):
                    continue
                try:
                    if float(value) == float(candidate):
                        return candidate
                except (TypeError, ValueError):
                    continue
            raise ConfigurationError(
                f"Parameter {self.name!r} must be one of {list(self.choices)}; got {value!r}."
            )
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ConfigurationError(
                f"Parameter {self.name!r} must be numeric; got {type(value).__name__}."
            )
        if self.kind == "int":
            return int(value)
        number = float(value)
        if not math.isfinite(number):
            raise ConfigurationError(f"Parameter {self.name!r} must be finite.")
        if self.low is not None:
            number = max(float(self.low), number)
        if self.high is not None:
            number = min(float(self.high), number)
        return number

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "kind": self.kind}
        for key in ("low", "high", "log", "choices", "description"):
            value = getattr(self, key)
            if value is not None and value != () and value != "":
                payload[key] = list(value) if key == "choices" else value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Parameter:
        if "name" not in payload or "kind" not in payload:
            raise ConfigurationError("Parameter payload requires 'name' and 'kind'.")
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            low=payload.get("low"),
            high=payload.get("high"),
            log=bool(payload.get("log", False)),
            choices=tuple(payload.get("choices", ())),
            description=str(payload.get("description", "")),
        )


class SearchSpace:
    """An ordered mapping of parameter name to :class:`Parameter`."""

    def __init__(self, parameters: Mapping[str, Parameter] | None = None) -> None:
        self._parameters: dict[str, Parameter] = {}
        for name, parameter in (parameters or {}).items():
            if not isinstance(parameter, Parameter):
                raise ConfigurationError(
                    f"SearchSpace entries must be Parameter instances; got {type(parameter).__name__}."
                )
            if parameter.name != name:
                raise ConfigurationError(
                    f"SearchSpace key {name!r} does not match parameter name {parameter.name!r}."
                )
            self._parameters[name] = parameter

    # ------------------------------------------------------------------ mapping API
    def __getitem__(self, name: str) -> Parameter:
        return self._parameters[name]

    def __contains__(self, name: object) -> bool:
        return name in self._parameters

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameters)

    def __len__(self) -> int:
        return len(self._parameters)

    def keys(self):
        return self._parameters.keys()

    def items(self):
        return self._parameters.items()

    def __repr__(self) -> str:
        return f"SearchSpace({', '.join(self._parameters)})"

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SearchSpace:
        parameters: dict[str, Parameter] = {}
        for name, spec in payload.items():
            if isinstance(spec, Parameter):
                parameters[name] = spec
                continue
            if not isinstance(spec, Mapping):
                raise ConfigurationError(
                    f"SearchSpace entry {name!r} must be a mapping or Parameter."
                )
            data = dict(spec)
            data.setdefault("name", name)
            parameters[name] = Parameter.from_dict(data)
        return cls(parameters)

    def to_dict(self) -> dict[str, Any]:
        return {name: parameter.to_dict() for name, parameter in self._parameters.items()}

    def merge(self, overrides: Mapping[str, Any] | None) -> SearchSpace:
        """Return a new space where ``overrides`` replace or add parameters."""
        merged = dict(self._parameters)
        for name, spec in (overrides or {}).items():
            if isinstance(spec, Parameter):
                merged[name] = spec
            elif isinstance(spec, Mapping):
                data = dict(spec)
                data.setdefault("name", name)
                merged[name] = Parameter.from_dict(data)
            else:
                # A bare value means "fix this parameter".
                merged[name] = Parameter(name=name, kind="choice", choices=(spec,))
        return SearchSpace(merged)

    # ------------------------------------------------------------------ sampling
    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return {name: parameter.sample(rng) for name, parameter in self._parameters.items()}

    def grid(self, n_points: int = 3, *, max_combinations: int = 64) -> list[dict[str, Any]]:
        """Enumerate a deterministic grid, truncated to ``max_combinations``."""
        if not self._parameters:
            return [{}]
        axes = [parameter.grid(n_points) for parameter in self._parameters.values()]
        total = math.prod(len(axis) for axis in axes)
        combinations: list[dict[str, Any]] = []
        for index in range(min(total, max_combinations)):
            combination: dict[str, Any] = {}
            remainder = index
            for name, axis in zip(list(self._parameters), axes, strict=False):
                combination[name] = axis[remainder % len(axis)]
                remainder //= len(axis)
            combinations.append(combination)
        return combinations

    def validate(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a concrete configuration against this space.

        Parameters that are omitted fall back to the space's default when it has
        one; only parameters with no default are reported as missing.
        """
        unknown = [name for name in values if name not in self._parameters]
        if unknown:
            raise ConfigurationError(
                f"Unknown parameter(s): {', '.join(sorted(unknown))}. "
                f"Expected: {', '.join(sorted(self._parameters))}."
            )
        missing = [
            name
            for name, parameter in self._parameters.items()
            if name not in values and parameter.default is None
        ]
        if missing:
            raise ConfigurationError(
                f"Configuration is missing parameter(s): {', '.join(sorted(missing))}."
            )
        resolved: dict[str, Any] = {}
        for name, parameter in self._parameters.items():
            raw = values[name] if name in values else parameter.default
            resolved[name] = parameter.coerce(raw)
        return resolved
