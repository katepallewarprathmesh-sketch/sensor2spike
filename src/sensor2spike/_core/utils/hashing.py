"""Deterministic hashing helpers.

Experiment identifiers must be reproducible (architecture requirement 10.3:
"deterministic experiment identifiers").  All hashing therefore goes through
canonical JSON, which sorts keys and uses a stable encoder for NumPy values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["canonical_json", "stable_hash", "sha256_file", "file_digest"]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: getattr(value, f.name)
            for f in value.__dataclass_fields__.values()  # type: ignore[attr-defined]
        }
    if isinstance(value, set):
        return sorted(str(item) for item in value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def canonical_json(value: Any) -> str:
    """Serialise ``value`` to a byte-stable JSON string."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )


def stable_hash(value: Any, *, length: int = 16, prefix: str = "") -> str:
    """Return a deterministic hex digest for any JSON-serialisable value."""
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:length]}"


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Hash a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_digest(path: str | Path, *, length: int = 16) -> str:
    """Short, URL-safe digest of a file, suitable for dataset identifiers."""
    return sha256_file(path)[:length]
