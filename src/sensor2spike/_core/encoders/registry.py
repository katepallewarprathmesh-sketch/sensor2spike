"""Encoder registry.

Encoders register themselves through :func:`register_encoder`.  New encoding
strategies can therefore be added without touching the optimisation engine or
the CLI (architecture principle: "add new encoding strategies without changing
the core").
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeVar

from ..exceptions import ConfigurationError
from ..spaces import SearchSpace
from .base import SpikeEncoder

__all__ = [
    "ENCODERS",
    "build_encoder",
    "encoder_space",
    "get_encoder_class",
    "list_encoders",
    "register_encoder",
    "validate_encoder_config",
]

T = TypeVar("T", bound=SpikeEncoder)

ENCODERS: dict[str, type[SpikeEncoder]] = {}
_ALIASES: dict[str, str] = {}


def register_encoder(
    cls: type[T] | None = None,
    *,
    name: str | None = None,
    aliases: Iterable[str] = (),
) -> Callable[[type[T]], type[T]] | type[T]:
    """Class decorator that adds an encoder to the registry.

    Usable as ``@register_encoder`` or ``@register_encoder(aliases=("psf",))``.
    """

    def _register(encoder_cls: type[T]) -> type[T]:
        key = (name or encoder_cls.name).strip().lower()
        if not key:
            raise ConfigurationError("Encoder name must not be empty.")
        if key in ENCODERS:
            raise ConfigurationError(f"Encoder {key!r} is already registered.")
        if not getattr(encoder_cls, "name", None):
            raise ConfigurationError(
                f"{encoder_cls.__name__} must define a non-empty 'name' attribute."
            )
        ENCODERS[key] = encoder_cls
        for alias in aliases:
            _ALIASES[alias.strip().lower()] = key
        return encoder_cls

    if cls is not None:
        return _register(cls)
    return _register


def get_encoder_class(name: str) -> type[SpikeEncoder]:
    """Resolve an encoder name or alias to its class."""
    key = str(name).strip().lower()
    if key in ENCODERS:
        return ENCODERS[key]
    if key in _ALIASES:
        return ENCODERS[_ALIASES[key]]
    available = ", ".join(sorted(ENCODERS))
    raise ConfigurationError(f"Unknown encoder {name!r}. Available encoders: {available}.")


def build_encoder(
    config: str | Mapping[str, Any] | SpikeEncoder,
    **overrides: Any,
) -> SpikeEncoder:
    """Instantiate an encoder from a name, a config mapping or an instance."""
    if isinstance(config, SpikeEncoder):
        return config
    if isinstance(config, str):
        encoder_cls = get_encoder_class(config)
        params: dict[str, Any] = {}
    elif isinstance(config, Mapping):
        payload = dict(config)
        encoder_name = payload.pop("encoder", payload.pop("name", None))
        if encoder_name is None:
            raise ConfigurationError(
                "Encoder configuration must include an 'encoder' field naming the strategy."
            )
        encoder_cls = get_encoder_class(str(encoder_name))
        params = payload
    else:
        raise ConfigurationError(
            f"Cannot build an encoder from {type(config).__name__}; expected a name or mapping."
        )
    params.update(overrides)
    return encoder_cls.from_config({"encoder": encoder_cls.name, **params})


def encoder_space(name: str, overrides: Mapping[str, Any] | None = None) -> SearchSpace:
    """Return the default search space for ``name`` merged with overrides."""
    encoder = build_encoder(name)
    return encoder.to_space(overrides)


def validate_encoder_config(name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a (possibly partial) configuration against an encoder's space.

    Omitted parameters fall back to the encoder's own constructor defaults, so
    ``{"dt_ms": 2.0}`` is a valid configuration for the rate encoder.
    """
    if not isinstance(config, Mapping):
        raise ConfigurationError("Encoder configuration must be a mapping.")
    defaults = build_encoder(name).get_config()
    defaults.pop("encoder", None)
    space = encoder_space(name)
    # ``seed`` and ``dt_ms`` are managed by the base class, not the search space.
    for reserved in ("seed", "encoder"):
        defaults.pop(reserved, None)
    unknown = [key for key in config if key not in space and key not in {"seed", "encoder"}]
    if unknown:
        raise ConfigurationError(
            f"Unknown parameter(s) for encoder {name!r}: {', '.join(sorted(unknown))}. "
            f"Expected: {', '.join(sorted(space.keys()))}."
        )
    merged = {**defaults, **dict(config)}
    resolved = {key: space[key].coerce(value) for key, value in merged.items() if key in space}
    for key in ("seed", "encoder"):
        if key in config:
            resolved[key] = config[key]
    return resolved


def list_encoders() -> list[dict[str, Any]]:
    """Describe every registered encoder (used by the CLI and reports)."""
    return [
        {
            "name": key,
            "description": cls.description,
            "class": f"{cls.__module__}.{cls.__name__}",
            "parameters": list(cls.PARAM_NAMES),
        }
        for key, cls in sorted(ENCODERS.items())
    ]
