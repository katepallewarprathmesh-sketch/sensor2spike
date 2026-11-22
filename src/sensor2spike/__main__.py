"""Command-line entry point (``python -m sensor2spike``)."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
