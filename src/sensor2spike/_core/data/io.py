"""Dataset I/O: CSV, NumPy ``.npy`` and ``.npz`` support.

The loaders are intentionally conservative:

* only ``.csv``, ``.npy`` and ``.npz`` are accepted;
* ``np.load`` is called with ``allow_pickle=False`` so a crafted file can never
  execute code;
* file size is checked before anything is read;
* malformed rows produce a :class:`~sensor2spike.exceptions.DataValidationError`
  naming the offending row.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import MAX_DATASET_BYTES, SensorDataset, SpikeTrain
from ..exceptions import ConfigurationError, DataValidationError

__all__ = [
    "load_dataset",
    "load_csv",
    "load_npz",
    "load_npy",
    "save_dataset",
    "save_csv",
    "save_npz",
    "load_spike_train",
    "save_spike_train",
]

_NAN_POLICIES = ("raise", "interpolate", "zero", "drop")
_RESERVED_HINTS = ("time", "timestamp", "label", "group", "trial", "class")


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def load_dataset(path: str | Path, **kwargs: Any) -> SensorDataset:
    """Load a dataset, dispatching on the file suffix."""
    path = Path(path)
    suffix = path.suffix.lower()
    loaders = {".csv": load_csv, ".npz": load_npz, ".npy": load_npy}
    loader = loaders.get(suffix)
    if loader is None:
        supported = ", ".join(sorted(loaders))
        raise DataValidationError(
            f"Unsupported dataset format {suffix or '<none>'!r}. Supported formats: {supported}."
        )
    _check_readable(path)
    return loader(path, **kwargs)


def save_dataset(dataset: SensorDataset, path: str | Path) -> Path:
    """Save ``dataset`` to ``.npz`` (preferred) or ``.csv``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return save_npz(dataset, path)
    if suffix == ".csv":
        return save_csv(dataset, path)
    raise ConfigurationError(
        f"Unsupported output format {suffix!r}. Use '.npz' or '.csv'."
    )


# --------------------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------------------
def load_csv(
    path: str | Path,
    *,
    time_column: str | None = None,
    label_column: str | None = None,
    group_column: str | None = None,
    channel_columns: Sequence[str] | None = None,
    sampling_frequency: float | None = None,
    name: str | None = None,
    sensor_type: str = "unknown",
    metadata: dict[str, Any] | None = None,
    nan_policy: str = "raise",
    delimiter: str | None = None,
    encoding: str = "utf-8-sig",
    max_bytes: int = MAX_DATASET_BYTES,
    **_: Any,
) -> SensorDataset:
    """Load a CSV sensor recording.

    Parameters
    ----------
    time_column:
        Name of a column holding numeric seconds or ISO-8601 timestamps.  When
        omitted, timestamps are synthesised from ``sampling_frequency``.
    label_column, group_column:
        Optional class-label and trial/segment columns.
    channel_columns:
        Explicit channel columns.  Defaults to every column that is not used as
        time/label/group.
    sampling_frequency:
        Sample rate in Hz.  Required unless a time column is provided, in which
        case it is inferred from the median timestamp difference.
    nan_policy:
        ``"raise"`` (default), ``"interpolate"``, ``"zero"`` or ``"drop"``.
    """
    path = Path(path)
    _check_readable(path, max_bytes=max_bytes)
    if nan_policy not in _NAN_POLICIES:
        raise ConfigurationError(
            f"nan_policy must be one of {_NAN_POLICIES}; got {nan_policy!r}."
        )

    rows = _read_rows(path, delimiter=delimiter, encoding=encoding)
    if not rows:
        raise DataValidationError(f"{path.name}: file is empty.")

    header: list[str] | None
    body: list[list[str]]
    if _row_is_header(rows[0]):
        header = [cell.strip() for cell in rows[0]]
        body = rows[1:]
    else:
        width = max(len(row) for row in rows)
        header = [f"col{i}" for i in range(width)]
        body = rows
    if not body:
        raise DataValidationError(f"{path.name}: no data rows found.")

    width = len(header)
    for row_number, row in enumerate(body, start=2):
        if len(row) != width:
            raise DataValidationError(
                f"{path.name}: row {row_number} has {len(row)} fields but the header has {width}."
            )

    def _resolve(column: str | None, purpose: str) -> int | None:
        if column is None:
            return None
        if column not in header:
            raise ConfigurationError(
                f"{path.name}: {purpose} column {column!r} was not found. "
                f"Available columns: {', '.join(header)}."
            )
        return header.index(column)

    time_idx = _resolve(time_column, "time")
    label_idx = _resolve(label_column, "label")
    group_idx = _resolve(group_column, "group")

    if channel_columns is not None:
        channel_idx = [_resolve(column, "channel") for column in channel_columns]
        channel_idx = [index for index in channel_idx if index is not None]
        names = [header[index] for index in channel_idx]
    else:
        reserved = {index for index in (time_idx, label_idx, group_idx) if index is not None}
        channel_idx = [index for index in range(width) if index not in reserved]
        names = [header[index] for index in channel_idx]
    if not channel_idx:
        raise DataValidationError(
            f"{path.name}: no channel columns were found. Specify channel_columns explicitly."
        )

    data = _to_float_matrix(body, channel_idx, path)
    labels = _to_int_vector(body, label_idx, "label", path)
    groups = _to_int_vector(body, group_idx, "group", path)
    timestamps = _parse_time_column(body, time_idx, path) if time_idx is not None else None

    data, timestamps, labels, groups = _apply_nan_policy(
        data, timestamps, labels, groups, nan_policy=nan_policy, name=path.name
    )

    if timestamps is None and sampling_frequency is None:
        raise ConfigurationError(
            f"{path.name}: provide either a time column or sampling_frequency."
        )
    if timestamps is not None:
        inferred = 1.0 / float(np.median(np.diff(timestamps)))
        fs = float(sampling_frequency) if sampling_frequency is not None else inferred
    else:
        fs = float(sampling_frequency)  # type: ignore[arg-type]

    return SensorDataset(
        data=data,
        sampling_frequency=fs,
        channel_names=tuple(names),
        timestamps=timestamps,
        labels=labels,
        groups=groups,
        name=name or path.stem,
        sensor_type=sensor_type,
        metadata=dict(metadata or {}),
    )


def save_csv(dataset: SensorDataset, path: str | Path) -> Path:
    """Write a dataset to CSV, including label and group columns when present."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(dataset.channel_names)
    if dataset.groups is not None:
        header.append("group")
    if dataset.labels is not None:
        header.append("label")
    if dataset.timestamps is not None:
        header.append("timestamp")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(dataset.n_samples):
            row: list[Any] = [f"{value:.17g}" for value in dataset.data[index]]
            if dataset.groups is not None:
                row.append(int(dataset.groups[index]))
            if dataset.labels is not None:
                row.append(int(dataset.labels[index]))
            if dataset.timestamps is not None:
                row.append(f"{dataset.timestamps[index]:.17g}")
            writer.writerow(row)
    return path


# --------------------------------------------------------------------------------------
# NumPy
# --------------------------------------------------------------------------------------
def load_npz(
    path: str | Path,
    *,
    max_bytes: int = MAX_DATASET_BYTES,
    **_: Any,
) -> SensorDataset:
    """Load a dataset previously written by :func:`save_npz`."""
    path = Path(path)
    _check_readable(path, max_bytes=max_bytes)
    # ``allow_pickle=False`` is enforced on every array access, not just on
    # ``np.load``, so the whole read is wrapped: an untrusted archive can never
    # execute code.
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = [key for key in ("data", "sampling_frequency") if key not in payload]
            if missing:
                raise DataValidationError(
                    f"{path.name}: .npz file is missing required array(s): {', '.join(missing)}."
                )
            data = np.asarray(payload["data"], dtype=np.float32)
            fs = float(np.asarray(payload["sampling_frequency"]).reshape(-1)[0])
            timestamps = _optional_array(payload, "timestamps")
            labels = _optional_array(payload, "labels")
            groups = _optional_array(payload, "groups")
            names = payload["channel_names"].tolist() if "channel_names" in payload else []
            name = _scalar(payload, "name", default=path.stem)
            sensor_type = _scalar(payload, "sensor_type", default="unknown")
            version = _scalar(payload, "version", default="1")
            metadata_raw = (
                payload["metadata_json"].reshape(-1)[0] if "metadata_json" in payload else "{}"
            )
            try:
                metadata = json.loads(str(metadata_raw))
            except json.JSONDecodeError as exc:
                raise DataValidationError(
                    f"{path.name}: metadata_json is not valid JSON: {exc}"
                ) from exc
    except ValueError as exc:
        raise DataValidationError(
            f"{path.name}: archive contains pickled objects; refusing to load untrusted "
            "data (rewrite the archive with allow_pickle=False)."
        ) from exc
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return SensorDataset(
        data=data,
        sampling_frequency=fs,
        channel_names=tuple(str(item) for item in names),
        timestamps=timestamps,
        labels=labels,
        groups=groups,
        name=str(name),
        sensor_type=str(sensor_type),
        metadata=metadata if isinstance(metadata, dict) else {"value": metadata},
        version=str(version),
    )


def load_npy(
    path: str | Path,
    *,
    sampling_frequency: float,
    channel_names: Sequence[str] | None = None,
    name: str | None = None,
    **_: Any,
) -> SensorDataset:
    """Load a single-channel (or 2-D) ``.npy`` signal."""
    path = Path(path)
    _check_readable(path)
    data = np.load(path, allow_pickle=False)
    return SensorDataset(
        data=data,
        sampling_frequency=sampling_frequency,
        channel_names=tuple(channel_names or ()),
        name=name or path.stem,
    )


def save_npz(dataset: SensorDataset, path: str | Path) -> Path:
    """Write a dataset to a compressed ``.npz`` archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "data": dataset.data.astype(np.float32),
        "sampling_frequency": np.float64(dataset.sampling_frequency),
        "channel_names": np.asarray(dataset.channel_names, dtype="U64"),
        "name": np.asarray(dataset.name),
        "sensor_type": np.asarray(dataset.sensor_type),
        "version": np.asarray(dataset.version),
        "metadata_json": np.asarray(json.dumps(dataset.metadata, sort_keys=True, default=str)),
    }
    if dataset.timestamps is not None:
        payload["timestamps"] = np.asarray(dataset.timestamps, dtype=np.float64)
    if dataset.labels is not None:
        payload["labels"] = np.asarray(dataset.labels, dtype=np.int64)
    if dataset.groups is not None:
        payload["groups"] = np.asarray(dataset.groups, dtype=np.int64)
    np.savez_compressed(path, **payload)
    return path


# --------------------------------------------------------------------------------------
# Spike trains
# --------------------------------------------------------------------------------------
def save_spike_train(spikes: SpikeTrain, path: str | Path) -> Path:
    """Save a spike train as a compressed ``.npz`` archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        spikes=spikes.spikes.astype(np.uint8),
        dt=np.float64(spikes.dt),
        channel_names=np.asarray(spikes.channel_names, dtype="U64"),
        encoder=np.asarray("" if spikes.encoder is None else spikes.encoder),
        metadata_json=np.asarray(json.dumps(spikes.metadata, sort_keys=True, default=str)),
    )
    return path


def load_spike_train(path: str | Path) -> SpikeTrain:
    """Load a spike train written by :func:`save_spike_train`."""
    path = Path(path)
    _check_readable(path)
    with np.load(path, allow_pickle=False) as payload:
        if "spikes" not in payload or "dt" not in payload:
            raise DataValidationError(f"{path.name}: not a Sensor2Spike spike-train archive.")
        metadata_raw = payload["metadata_json"].reshape(-1)[0] if "metadata_json" in payload else "{}"
        try:
            metadata = json.loads(str(metadata_raw))
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"{path.name}: metadata_json is not valid JSON.") from exc
        return SpikeTrain(
            spikes=np.asarray(payload["spikes"]),
            dt=float(np.asarray(payload["dt"]).reshape(-1)[0]),
            channel_names=tuple(str(item) for item in payload["channel_names"].tolist())
            if "channel_names" in payload
            else (),
            encoder=str(payload["encoder"]) if "encoder" in payload else None,
            metadata=metadata,
        )


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------
def _check_readable(path: Path, *, max_bytes: int = MAX_DATASET_BYTES) -> None:
    if not path.exists():
        raise DataValidationError(f"File not found: {path.name}")
    if not path.is_file():
        raise DataValidationError(f"{path.name} is not a regular file.")
    size = path.stat().st_size
    if size > max_bytes:
        raise DataValidationError(
            f"{path.name} is {size / 1024**2:.1f} MiB which exceeds the "
            f"{max_bytes / 1024**2:.0f} MiB limit."
        )


def _read_rows(path: Path, *, delimiter: str | None, encoding: str) -> list[list[str]]:
    with open(path, newline="", encoding=encoding) as handle:
        if delimiter is None:
            sample = handle.read(8192)
            handle.seek(0)
            delimiter = _sniff_delimiter(sample)
        reader = csv.reader(handle, delimiter=delimiter)
        return [row for row in reader if any(cell.strip() for cell in row)]


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _row_is_header(row: Sequence[str]) -> bool:
    for cell in row:
        cell = cell.strip()
        if not cell:
            continue
        try:
            float(cell)
        except ValueError:
            return True
    return False


def _to_float_matrix(body: Iterable[Sequence[str]], columns: Sequence[int], path: Path) -> np.ndarray:
    rows = list(body)
    out = np.full((len(rows), len(columns)), np.nan, dtype=np.float64)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            cell = row[column].strip() if column < len(row) else ""
            if cell == "":
                continue
            try:
                out[row_index, column_index] = float(cell)
            except ValueError as exc:
                raise DataValidationError(
                    f"{path.name}: row {row_index + 2}, column {column + 1}: "
                    f"{cell!r} is not a number."
                ) from exc
    if np.all(np.isnan(out)):
        raise DataValidationError(f"{path.name}: all channel values are missing.")
    return out


def _to_int_vector(
    body: Iterable[Sequence[str]], index: int | None, purpose: str, path: Path
) -> np.ndarray | None:
    if index is None:
        return None
    values: list[int] = []
    for row_number, row in enumerate(body, start=2):
        cell = row[index].strip()
        try:
            values.append(int(float(cell)))
        except ValueError as exc:
            raise DataValidationError(
                f"{path.name}: row {row_number}: {purpose} value {cell!r} is not an integer."
            ) from exc
    return np.asarray(values, dtype=np.int64)


def _parse_time_column(
    body: Iterable[Sequence[str]], index: int, path: Path
) -> np.ndarray:
    cells = [row[index].strip() for row in body]
    numeric: list[float | None] = []
    for cell in cells:
        try:
            numeric.append(float(cell))
        except ValueError:
            numeric.append(None)
    if all(value is not None for value in numeric):
        return np.asarray(numeric, dtype=np.float64)

    datetimes: list[datetime | None] = []
    for cell in cells:
        try:
            datetimes.append(datetime.fromisoformat(cell.replace("Z", "+00:00")))
        except ValueError:
            datetimes.append(None)
    if any(value is None for value in datetimes):
        raise DataValidationError(
            f"{path.name}: time column contains values that are neither numeric seconds "
            "nor ISO-8601 timestamps."
        )
    parsed = [value for value in datetimes if value is not None]
    if any(value.tzinfo is None for value in parsed) and any(
        value.tzinfo is not None for value in parsed
    ):
        raise DataValidationError(
            f"{path.name}: time column mixes timezone-aware and naive timestamps."
        )
    base = parsed[0]
    return np.asarray([(value - base).total_seconds() for value in parsed], dtype=np.float64)


def _apply_nan_policy(
    data: np.ndarray,
    timestamps: np.ndarray | None,
    labels: np.ndarray | None,
    groups: np.ndarray | None,
    *,
    nan_policy: str,
    name: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not np.isnan(data).any():
        return data, timestamps, labels, groups

    if nan_policy == "raise":
        first = np.argwhere(np.isnan(data))[0]
        raise DataValidationError(
            f"{name}: dataset contains missing values (first at row {first[0]}, "
            f"channel {first[1]}). Use nan_policy='interpolate', 'zero' or 'drop'."
        )

    if nan_policy == "zero":
        return np.nan_to_num(data, nan=0.0), timestamps, labels, groups

    if nan_policy == "drop":
        keep = ~np.isnan(data).any(axis=1)
        if not keep.any():
            raise DataValidationError(f"{name}: every row contains a missing value.")
        return (
            data[keep],
            None if timestamps is None else timestamps[keep],
            None if labels is None else labels[keep],
            None if groups is None else groups[keep],
        )

    # interpolate (linear, per channel, forward/backward filled at the edges)
    filled = data.copy()
    n_samples = filled.shape[0]
    grid = np.arange(n_samples, dtype=np.float64)
    for channel in range(filled.shape[1]):
        column = filled[:, channel]
        known = ~np.isnan(column)
        if not known.any():
            raise DataValidationError(f"{name}: channel {channel} contains only missing values.")
        if known.all():
            continue
        filled[:, channel] = np.interp(grid, grid[known], column[known])
    return filled, timestamps, labels, groups


def _optional_array(payload: Any, key: str) -> np.ndarray | None:
    return None if key not in payload else np.asarray(payload[key])


def _scalar(payload: Any, key: str, *, default: Any) -> Any:
    if key not in payload:
        return default
    array = np.asarray(payload[key])
    return array.reshape(-1)[0] if array.size else default


_ = os  # keep the import for optional future path checks
