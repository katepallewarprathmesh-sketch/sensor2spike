"""SNN evaluation engine with pluggable backends."""

from __future__ import annotations

from .base import SNNBackend
from .numpy_lif import LIFRidgeBackend
from .registry import BACKENDS, build_backend, get_backend_class, list_backends, register_backend
from .torch_lif import TorchLIFBackend

__all__ = [
    "SNNBackend",
    "LIFRidgeBackend",
    "TorchLIFBackend",
    "BACKENDS",
    "build_backend",
    "get_backend_class",
    "list_backends",
    "register_backend",
]
