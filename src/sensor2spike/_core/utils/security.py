"""Redaction helpers used when surfacing errors to end users.

Error text surfaced to users should identify *which* file failed without
exposing a user's directory layout. This module provides a small,
dependency-free implementation.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["redact_paths", "safe_name"]

# Absolute POSIX paths, Windows drive paths and file:// URLs.
_PATH_PATTERNS = (
    re.compile(r"file://[^\s'\"<>]+"),
    re.compile(r"(?<![\w./-])/(?:[\w.@+-]+/)*[\w.@+-]+"),
    re.compile(r"(?<![\w\\])(?:[A-Za-z]:\\)(?:[\w.@+-]+\\)*[\w.@+-]+"),
)


def redact_paths(text: str, *, keep_basename: bool = True) -> str:
    """Replace absolute paths in ``text`` with their basename or ``<path>``.

    Keeps error messages useful (you can still tell *which* file failed) while
    avoiding leaking a user's home directory or project layout.
    """
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        candidate = raw
        if candidate.startswith("file://"):
            candidate = candidate[len("file://") :]
        name = Path(candidate.replace("\\", "/")).name
        return f"<{name}>" if keep_basename and name else "<path>"

    for pattern in _PATH_PATTERNS:
        text = pattern.sub(_replace, text)
    return text


def safe_name(name: str, *, max_length: int = 80) -> str:
    """Return a filesystem-safe identifier derived from ``name``."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return (cleaned or "artifact")[:max_length]
