"""A tiny deterministic wall-clock timer."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = ["Timer"]

T = TypeVar("T")


class Timer:
    """Context manager that records elapsed wall-clock seconds.

    Examples
    --------
    >>> with Timer() as timer:
    ...     sum(range(1000))
    >>> timer.elapsed >= 0.0
    True
    """

    __slots__ = ("_start", "_elapsed", "_running")

    def __init__(self) -> None:
        self._start = 0.0
        self._elapsed = 0.0
        self._running = False

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        self._running = True
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._elapsed = time.perf_counter() - self._start
        self._running = False

    @property
    def elapsed(self) -> float:
        """Elapsed seconds; includes the in-progress interval if still running."""
        if self._running:
            return time.perf_counter() - self._start
        return self._elapsed

    @staticmethod
    def measure(func: Callable[[], T]) -> tuple[T, float]:
        """Run ``func`` and return ``(result, elapsed_seconds)``."""
        with Timer() as timer:
            result = func()
        return result, timer.elapsed
