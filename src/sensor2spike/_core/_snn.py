"""Implementation: evaluation backends and their registry."""

from .snn import (
    BACKENDS,
    LIFRidgeBackend,
    NumpySGDBackend,
    SNNBackend,
    TorchLIFBackend,
    build_backend,
    get_backend_class,
    list_backends,
    register_backend,
)

__all__ = [
    "SNNBackend",
    "LIFRidgeBackend",
    "NumpySGDBackend",
    "TorchLIFBackend",
    "BACKENDS",
    "build_backend",
    "get_backend_class",
    "list_backends",
    "register_backend",
]
