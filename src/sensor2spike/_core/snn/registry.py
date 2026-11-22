"""SNN backend registry (mirrors the encoder registry)."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeVar

from ..exceptions import ConfigurationError
from .base import SNNBackend

__all__ = ["BACKENDS", "build_backend", "get_backend_class", "list_backends", "register_backend"]

T = TypeVar("T", bound=SNNBackend)

BACKENDS: dict[str, type[SNNBackend]] = {}


def register_backend(
    cls: type[T] | None = None, *, name: str | None = None, aliases: Iterable[str] = ()
) -> Callable[[type[T]], type[T]] | type[T]:
    """Register an SNN backend.  Usable with or without arguments."""

    def _register(backend_cls: type[T]) -> type[T]:
        key = (name or backend_cls.name).strip().lower()
        if not key:
            raise ConfigurationError("Backend name must not be empty.")
        if key in BACKENDS:
            raise ConfigurationError(f"Backend {key!r} is already registered.")
        if inspect.isabstract(backend_cls):
            missing = ", ".join(sorted(getattr(backend_cls, "__abstractmethods__", ())))
            raise ConfigurationError(
                f"Backend {backend_cls.__name__} is abstract (missing: {missing}); "
                "implement every hook before registering it."
            )
        BACKENDS[key] = backend_cls
        return backend_cls

    if cls is not None:
        return _register(cls)
    return _register


def get_backend_class(name: str) -> type[SNNBackend]:
    key = str(name).strip().lower()
    if key not in BACKENDS:
        available = ", ".join(sorted(BACKENDS))
        raise ConfigurationError(f"Unknown SNN backend {name!r}. Available backends: {available}.")
    return BACKENDS[key]


def build_backend(
    config: str | Mapping[str, Any] | SNNBackend,
    **overrides: Any,
) -> SNNBackend:
    """Instantiate a backend from a name, a config mapping or an instance."""
    if isinstance(config, SNNBackend):
        return config
    if isinstance(config, str):
        backend_cls = get_backend_class(config)
        params: dict[str, Any] = {}
    elif isinstance(config, Mapping):
        payload = dict(config)
        backend_name = payload.pop("backend", payload.pop("name", None))
        if backend_name is None:
            raise ConfigurationError("Backend configuration must include a 'backend' field.")
        backend_cls = get_backend_class(str(backend_name))
        params = payload
    else:
        raise ConfigurationError(
            f"Cannot build a backend from {type(config).__name__}; expected a name or mapping."
        )
    params.update(overrides)
    try:
        return backend_cls(**params)
    except TypeError as exc:
        raise ConfigurationError(f"Invalid configuration for {backend_cls.name}: {exc}") from exc


def list_backends() -> list[dict[str, Any]]:
    """Describe every registered backend."""
    return [
        {
            "name": key,
            "description": cls.description,
            "class": f"{cls.__module__}.{cls.__name__}",
        }
        for key, cls in sorted(BACKENDS.items())
    ]
