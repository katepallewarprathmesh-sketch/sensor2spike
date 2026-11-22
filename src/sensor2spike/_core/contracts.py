"""Core data contracts for Sensor2Spike.

These objects are the explicit data contracts between pipeline stages. They are
deliberately plain NumPy containers so they can be hashed, serialised and moved
between processes without pulling in a deep-learning framework.

Contracts in this module
------------------------
``SensorDataset``
    A multichannel, uniformly sampled sensor recording with optional
    per-sample labels and trial/segment identifiers.
``WindowSet``
    Fixed-length windows extracted from a dataset, with train/test splits
    performed at the *group* level to prevent leakage.
``SpikeTrain``
    A binary spike matrix of shape ``(n_steps, n_neurons)`` plus the step size.
``SpikeStats``
    Aggregate spike-activity metrics for one or more spike trains.
``EvaluationResult``
    The output of an SNN evaluation backend.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import DataValidationError
from .utils.hashing import stable_hash
from .utils.validation import (
    as_1d_float_array,
    as_1d_int_array,
    as_2d_float_array,
    require_choice,
    require_positive,
)

__all__ = [
    "SensorDataset",
    "WindowSet",
    "SpikeTrain",
    "SpikeStats",
    "EvaluationResult",
    "MAX_DATASET_BYTES",
    "MAX_SAMPLES",
]

#: Guardrails that keep a single load from exhausting memory.
MAX_DATASET_BYTES = 4 * 1024**3
MAX_SAMPLES = 50_000_000

#: Relative tolerance when cross-checking timestamps against the declared rate.
_TIMESTAMP_TOLERANCE = 0.1

_LABEL_MODES = ("center", "majority")


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
@dataclass
class SensorDataset:
    """A multichannel sensor recording.

    Parameters
    ----------
    data:
        Array of shape ``(n_samples, n_channels)``.  A 1-D array is treated as a
        single-channel signal.
    sampling_frequency:
        Sample rate in Hz.
    channel_names:
        Optional channel names.  Defaults to ``ch0 ... chN``.
    timestamps:
        Optional per-sample timestamps in seconds.  If provided they must be
        strictly increasing and consistent with ``sampling_frequency``.
    labels:
        Optional integer class label per sample.
    groups:
        Optional integer trial/segment identifier per sample.  Windows are never
        allowed to cross a group boundary and splits are made per group.
    name, sensor_type, metadata, version:
        Free-form provenance used by the experiment registry and reports.
    """

    data: np.ndarray
    sampling_frequency: float
    channel_names: tuple[str, ...] = ()
    timestamps: np.ndarray | None = None
    labels: np.ndarray | None = None
    groups: np.ndarray | None = None
    name: str = "dataset"
    sensor_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "1"

    def __post_init__(self) -> None:
        self.data = as_2d_float_array(self.data, name=f"{self.name}.data")
        n_samples, n_channels = self.data.shape

        if n_samples > MAX_SAMPLES:
            raise DataValidationError(
                f"{self.name}: dataset has {n_samples} samples which exceeds the "
                f"configured guardrail of {MAX_SAMPLES}. Split the recording first."
            )
        if self.data.nbytes > MAX_DATASET_BYTES:
            raise DataValidationError(
                f"{self.name}: dataset is {self.data.nbytes / 1024**3:.2f} GiB which exceeds "
                f"the {MAX_DATASET_BYTES / 1024**3:.0f} GiB guardrail."
            )

        self.sampling_frequency = require_positive(
            self.sampling_frequency, f"{self.name}.sampling_frequency"
        )

        if not self.channel_names:
            self.channel_names = tuple(f"ch{i}" for i in range(n_channels))
        else:
            self.channel_names = tuple(str(name) for name in self.channel_names)
            if len(self.channel_names) != n_channels:
                raise DataValidationError(
                    f"{self.name}: {len(self.channel_names)} channel names were given but the "
                    f"data has {n_channels} channels."
                )

        if self.timestamps is not None:
            self.timestamps = as_1d_float_array(
                self.timestamps, name=f"{self.name}.timestamps"
            )
            if self.timestamps.shape[0] != n_samples:
                raise DataValidationError(
                    f"{self.name}: timestamps have length {self.timestamps.shape[0]} but the "
                    f"data has {n_samples} samples."
                )
            if n_samples > 1 and not np.all(np.diff(self.timestamps) > 0):
                raise DataValidationError(
                    f"{self.name}: timestamps must be strictly increasing."
                )
            if n_samples > 1:
                implied = 1.0 / float(np.median(np.diff(self.timestamps)))
                if abs(implied - self.sampling_frequency) / self.sampling_frequency > _TIMESTAMP_TOLERANCE:
                    warnings.warn(
                        f"{self.name}: timestamps imply {implied:.3f} Hz but "
                        f"sampling_frequency is {self.sampling_frequency:.3f} Hz. "
                        "Using the declared sampling frequency.",
                        stacklevel=2,
                    )

        if self.labels is not None:
            self.labels = as_1d_int_array(self.labels, name=f"{self.name}.labels", minimum=0)
            if self.labels.shape[0] != n_samples:
                raise DataValidationError(
                    f"{self.name}: labels have length {self.labels.shape[0]} but the data "
                    f"has {n_samples} samples."
                )

        if self.groups is not None:
            self.groups = as_1d_int_array(self.groups, name=f"{self.name}.groups", minimum=0)
            if self.groups.shape[0] != n_samples:
                raise DataValidationError(
                    f"{self.name}: groups have length {self.groups.shape[0]} but the data "
                    f"has {n_samples} samples."
                )

    # ------------------------------------------------------------------ basic properties
    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_samples, self.n_channels)

    @property
    def duration_seconds(self) -> float:
        """Recording duration in seconds (``n_samples / fs``)."""
        return self.n_samples / self.sampling_frequency

    @property
    def has_labels(self) -> bool:
        return self.labels is not None

    @property
    def has_groups(self) -> bool:
        return self.groups is not None

    @property
    def n_groups(self) -> int:
        return int(np.unique(self.groups).size) if self.groups is not None else 1

    @property
    def classes(self) -> np.ndarray:
        if self.labels is None:
            return np.empty(0, dtype=np.int64)
        return np.unique(self.labels)

    @property
    def n_classes(self) -> int:
        return int(self.classes.size)

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return (
            f"SensorDataset(name={self.name!r}, shape={self.shape}, "
            f"fs={self.sampling_frequency:g}Hz, duration={self.duration_seconds:.2f}s, "
            f"labels={self.has_labels}, groups={self.n_groups})"
        )

    # ------------------------------------------------------------------ provenance
    def fingerprint(self) -> str:
        """Deterministic digest of the dataset content and provenance."""
        payload = {
            "data": self.data,
            "sampling_frequency": float(self.sampling_frequency),
            "channel_names": list(self.channel_names),
            "labels": self.labels,
            "groups": self.groups,
            "name": self.name,
            "version": self.version,
        }
        return stable_hash(payload, length=16)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary used by reports and the registry."""
        return {
            "name": self.name,
            "sensor_type": self.sensor_type,
            "n_samples": self.n_samples,
            "n_channels": self.n_channels,
            "duration_seconds": round(self.duration_seconds, 6),
            "sampling_frequency": float(self.sampling_frequency),
            "channel_names": list(self.channel_names),
            "has_labels": self.has_labels,
            "n_classes": self.n_classes,
            "n_groups": self.n_groups,
            "version": self.version,
            "fingerprint": self.fingerprint(),
        }

    def describe(self) -> str:
        """Human-readable validation summary."""
        lines = [
            f"dataset          : {self.name}",
            f"sensor type      : {self.sensor_type}",
            f"shape            : {self.n_samples} samples x {self.n_channels} channels",
            f"sampling rate    : {self.sampling_frequency:g} Hz",
            f"duration         : {self.duration_seconds:.3f} s",
            f"channels         : {', '.join(self.channel_names)}",
            f"labels           : {self.n_classes if self.has_labels else 'none'}",
            f"groups           : {self.n_groups}",
            f"fingerprint      : {self.fingerprint()}",
        ]
        return "\n".join(lines)

    def copy(self, **overrides: Any) -> SensorDataset:
        """Return a copy with selected fields replaced."""
        return replace(self, **overrides)

    def __getitem__(self, key: slice) -> SensorDataset:
        if not isinstance(key, slice):
            raise TypeError("SensorDataset only supports slicing on the sample axis.")
        return replace(
            self,
            data=self.data[key],
            timestamps=None if self.timestamps is None else self.timestamps[key],
            labels=None if self.labels is None else self.labels[key],
            groups=None if self.groups is None else self.groups[key],
        )

    def segments(self) -> list[tuple[int, int]]:
        """Return ``(start, stop)`` index pairs for each contiguous group run."""
        if self.groups is None:
            return [(0, self.n_samples)]
        change = np.flatnonzero(np.diff(self.groups)) + 1
        boundaries = np.concatenate([[0], change, [self.n_samples]])
        return [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:], strict=False) if b > a]

    # ------------------------------------------------------------------ windowing
    def to_windows(
        self,
        window_samples: int,
        stride: int | None = None,
        *,
        label_mode: str = "center",
    ) -> WindowSet:
        """Extract fixed-length windows without crossing group boundaries.

        Parameters
        ----------
        window_samples:
            Window length in samples.
        stride:
            Step between window starts in samples.  Defaults to
            ``window_samples`` (non-overlapping windows).
        label_mode:
            ``"center"`` takes the label of the central sample, ``"majority"``
            takes the most frequent label inside the window.
        """
        window = int(require_positive(window_samples, "window_samples"))
        step = int(window if stride is None else require_positive(stride, "stride"))
        require_choice(label_mode, "label_mode", _LABEL_MODES)

        if window > self.n_samples:
            raise DataValidationError(
                f"window_samples={window} exceeds the dataset length of {self.n_samples}."
            )

        chunks: list[np.ndarray] = []
        labels: list[int] = []
        groups: list[int] = []
        starts: list[int] = []

        for seg_start, seg_stop in self.segments():
            span = seg_stop - seg_start
            if span < window:
                continue
            for start in range(seg_start, seg_stop - window + 1, step):
                stop = start + window
                chunks.append(self.data[start:stop])
                starts.append(start)
                if self.labels is not None:
                    if label_mode == "center":
                        labels.append(int(self.labels[start + window // 2]))
                    else:
                        values, counts = np.unique(self.labels[start:stop], return_counts=True)
                        labels.append(int(values[np.argmax(counts)]))
                if self.groups is not None:
                    groups.append(int(self.groups[start]))

        if not chunks:
            raise DataValidationError(
                f"No windows of {window} samples fit inside any group segment. "
                f"Largest segment is {max((b - a for a, b in self.segments()), default=0)} samples."
            )

        return WindowSet(
            samples=np.stack(chunks),
            labels=np.asarray(labels, dtype=np.int64) if self.labels is not None else None,
            groups=np.asarray(groups, dtype=np.int64) if self.groups is not None else None,
            starts=np.asarray(starts, dtype=np.int64),
            sampling_frequency=self.sampling_frequency,
            channel_names=self.channel_names,
            window_samples=window,
            stride=step,
            name=self.name,
        )

    # ------------------------------------------------------------------ persistence
    @classmethod
    def from_arrays(
        cls,
        data: Any,
        *,
        sampling_frequency: float,
        channel_names: Sequence[str] | None = None,
        timestamps: Any = None,
        labels: Any = None,
        groups: Any = None,
        name: str = "dataset",
        sensor_type: str = "unknown",
        metadata: dict[str, Any] | None = None,
        version: str = "1",
    ) -> SensorDataset:
        """Build a dataset from in-memory arrays."""
        return cls(
            data=data,
            sampling_frequency=sampling_frequency,
            channel_names=tuple(channel_names or ()),
            timestamps=timestamps,
            labels=labels,
            groups=groups,
            name=name,
            sensor_type=sensor_type,
            metadata=dict(metadata or {}),
            version=version,
        )

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> SensorDataset:
        """Load a dataset from ``.csv``, ``.npz`` or ``.npy``.

        Extra keyword arguments are forwarded to the format-specific loader.
        """
        from .data.io import load_dataset

        return load_dataset(path, **kwargs)

    def save(self, path: str | Path) -> Path:
        """Save the dataset as ``.npz`` (preferred) or ``.csv``."""
        from .data.io import save_dataset

        return save_dataset(self, path)


# --------------------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------------------
@dataclass
class WindowSet:
    """A batch of fixed-length windows extracted from a :class:`SensorDataset`."""

    samples: np.ndarray
    starts: np.ndarray
    sampling_frequency: float
    window_samples: int
    stride: int
    channel_names: tuple[str, ...] = ()
    labels: np.ndarray | None = None
    groups: np.ndarray | None = None
    name: str = ""

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.float32)
        if self.samples.ndim != 3:
            raise DataValidationError(
                f"WindowSet.samples must be 3-D (n_windows, window_samples, n_channels); "
                f"got shape {self.samples.shape}."
            )
        if not np.all(np.isfinite(self.samples)):
            raise DataValidationError("WindowSet.samples contains non-finite values.")
        n_windows = int(self.samples.shape[0])
        self.starts = as_1d_int_array(self.starts, name="WindowSet.starts", minimum=0)
        if self.starts.shape[0] != n_windows:
            raise DataValidationError(
                f"WindowSet.starts has length {self.starts.shape[0]} but there are "
                f"{n_windows} windows."
            )
        self.sampling_frequency = require_positive(
            self.sampling_frequency, "WindowSet.sampling_frequency"
        )
        self.window_samples = int(require_positive(self.window_samples, "WindowSet.window_samples"))
        self.stride = int(require_positive(self.stride, "WindowSet.stride"))
        self.channel_names = tuple(str(name) for name in self.channel_names)
        if self.labels is not None:
            self.labels = as_1d_int_array(self.labels, name="WindowSet.labels", minimum=0)
            if self.labels.shape[0] != n_windows:
                raise DataValidationError("WindowSet.labels length does not match the windows.")
        if self.groups is not None:
            self.groups = as_1d_int_array(self.groups, name="WindowSet.groups", minimum=0)
            if self.groups.shape[0] != n_windows:
                raise DataValidationError("WindowSet.groups length does not match the windows.")

    @property
    def n_windows(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.samples.shape[2])

    @property
    def window_seconds(self) -> float:
        return self.window_samples / self.sampling_frequency

    @property
    def duration_seconds(self) -> float:
        """Total duration covered by all windows."""
        return self.n_windows * self.window_seconds

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, index: int) -> np.ndarray:
        return self.samples[index]

    def __repr__(self) -> str:
        return (
            f"WindowSet(n_windows={self.n_windows}, window_samples={self.window_samples}, "
            f"n_channels={self.n_channels}, stride={self.stride})"
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        for index in range(self.n_windows):
            yield self.samples[index]

    def subset(self, indices: Any) -> WindowSet:
        """Return a new :class:`WindowSet` containing the selected windows."""
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim != 1:
            raise DataValidationError("indices must be a 1-D sequence of window indices.")
        if idx.size and (idx.min() < 0 or idx.max() >= self.n_windows):
            raise DataValidationError("Window indices are out of range.")
        return WindowSet(
            samples=self.samples[idx],
            starts=self.starts[idx],
            sampling_frequency=self.sampling_frequency,
            window_samples=self.window_samples,
            stride=self.stride,
            channel_names=self.channel_names,
            labels=None if self.labels is None else self.labels[idx],
            groups=None if self.groups is None else self.groups[idx],
            name=self.name,
        )


# --------------------------------------------------------------------------------------
# Spike representation
# --------------------------------------------------------------------------------------
@dataclass
class SpikeTrain:
    """A binary spike matrix of shape ``(n_steps, n_neurons)``."""

    spikes: np.ndarray
    dt: float
    channel_names: tuple[str, ...] = ()
    encoder: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.spikes)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise DataValidationError(
                f"SpikeTrain.spikes must be 2-D (n_steps, n_neurons); got shape {array.shape}."
            )
        if array.size == 0:
            raise DataValidationError("SpikeTrain.spikes is empty.")
        if array.dtype == np.bool_:
            array = array.astype(np.uint8)
        elif not np.issubdtype(array.dtype, np.integer):
            finite = np.all(np.isfinite(array)) and np.all((array == 0) | (array == 1))
            if not finite:
                raise DataValidationError("SpikeTrain.spikes must only contain 0 and 1.")
            array = array.astype(np.uint8)
        else:
            if array.size and (array.min() < 0 or array.max() > 1):
                raise DataValidationError("SpikeTrain.spikes must be binary (0/1).")
            array = array.astype(np.uint8)
        self.spikes = np.ascontiguousarray(array)
        self.dt = float(require_positive(self.dt, "SpikeTrain.dt"))
        self.channel_names = tuple(str(name) for name in self.channel_names)
        if self.channel_names and len(self.channel_names) != self.n_neurons:
            raise DataValidationError(
                f"SpikeTrain has {self.n_neurons} neurons but "
                f"{len(self.channel_names)} channel names were provided."
            )

    # ------------------------------------------------------------------ properties
    @property
    def n_steps(self) -> int:
        return int(self.spikes.shape[0])

    @property
    def n_neurons(self) -> int:
        return int(self.spikes.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_steps, self.n_neurons)

    @property
    def duration_seconds(self) -> float:
        return self.n_steps * self.dt

    @property
    def spike_count(self) -> int:
        return int(np.count_nonzero(self.spikes))

    @property
    def spike_rate_hz(self) -> float:
        """Mean firing rate across all neurons, in spikes per second."""
        return self.spike_count / self.duration_seconds

    @property
    def active_fraction(self) -> float:
        """Fraction of ``(step, neuron)`` entries that contain a spike."""
        return self.spike_count / max(1, self.spikes.size)

    @property
    def temporal_sparsity(self) -> float:
        """Fraction of entries without a spike (1 - :attr:`active_fraction`)."""
        return 1.0 - self.active_fraction

    @property
    def firing_rates_hz(self) -> np.ndarray:
        """Per-neuron firing rate in spikes per second."""
        return self.spikes.sum(axis=0, dtype=np.float64) / self.duration_seconds

    @property
    def active_steps(self) -> np.ndarray:
        """Indices of steps that contain at least one spike."""
        return np.flatnonzero(self.spikes.any(axis=1))

    def __len__(self) -> int:
        return self.n_steps

    def __repr__(self) -> str:
        return (
            f"SpikeTrain(n_steps={self.n_steps}, n_neurons={self.n_neurons}, "
            f"dt={self.dt * 1000:g}ms, spikes={self.spike_count}, "
            f"spikes/s={self.spike_rate_hz:.1f})"
        )

    def isi(self) -> np.ndarray:
        """Inter-spike intervals in seconds, pooled across neurons."""
        intervals = []
        for neuron in range(self.n_neurons):
            indices = np.flatnonzero(self.spikes[:, neuron])
            if indices.size > 1:
                intervals.append(np.diff(indices) * self.dt)
        if not intervals:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(intervals)

    def to_float(self) -> np.ndarray:
        """Return the spike matrix as ``float32`` (useful for ML backends)."""
        return self.spikes.astype(np.float32)

    def to_dense(self) -> np.ndarray:
        """Alias of :meth:`to_float` for symmetry with sparse representations."""
        return self.to_float()

    def to_dict(self) -> dict[str, Any]:
        return {
            "spikes": self.spikes,
            "dt": self.dt,
            "channel_names": list(self.channel_names),
            "encoder": self.encoder,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SpikeTrain:
        return cls(
            spikes=np.asarray(payload["spikes"]),
            dt=float(payload["dt"]),
            channel_names=tuple(payload.get("channel_names", ())),
            encoder=payload.get("encoder"),
            metadata=dict(payload.get("metadata", {})),
        )


# --------------------------------------------------------------------------------------
# Metrics containers
# --------------------------------------------------------------------------------------
@dataclass
class SpikeStats:
    """Aggregate spike-activity metrics."""

    spike_count: int
    n_steps: int
    n_neurons: int
    duration_seconds: float
    spike_rate_hz: float
    spikes_per_second_per_neuron: float
    active_fraction: float
    temporal_sparsity: float
    mean_isi_ms: float | None
    isi_cv: float | None
    firing_rates_hz: np.ndarray
    information_loss: float | None = None
    reconstruction_r2: float | None = None
    compute_cost: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spike_count": int(self.spike_count),
            "n_steps": int(self.n_steps),
            "n_neurons": int(self.n_neurons),
            "duration_seconds": float(self.duration_seconds),
            "spike_rate_hz": float(self.spike_rate_hz),
            "spikes_per_second_per_neuron": float(self.spikes_per_second_per_neuron),
            "active_fraction": float(self.active_fraction),
            "temporal_sparsity": float(self.temporal_sparsity),
            "mean_isi_ms": None if self.mean_isi_ms is None else float(self.mean_isi_ms),
            "isi_cv": None if self.isi_cv is None else float(self.isi_cv),
            "firing_rates_hz": self.firing_rates_hz.tolist(),
            "information_loss": None if self.information_loss is None else float(self.information_loss),
            "reconstruction_r2": None if self.reconstruction_r2 is None else float(self.reconstruction_r2),
            "compute_cost": None if self.compute_cost is None else float(self.compute_cost),
        }


@dataclass
class EvaluationResult:
    """Result of evaluating a spike representation with an SNN backend."""

    backend: str
    accuracy: float
    macro_f1: float
    n_train: int
    n_test: int
    train_time_s: float
    inference_time_s: float
    n_parameters: int
    train_accuracy: float | None = None
    predictions: np.ndarray | None = None
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.accuracy = float(self.accuracy)
        self.macro_f1 = float(self.macro_f1)
        if not 0.0 <= self.accuracy <= 1.0:
            raise DataValidationError(f"EvaluationResult.accuracy must be in [0, 1]; got {self.accuracy}.")
        if not 0.0 <= self.macro_f1 <= 1.0:
            raise DataValidationError(f"EvaluationResult.macro_f1 must be in [0, 1]; got {self.macro_f1}.")
        if self.n_test <= 0:
            raise DataValidationError("EvaluationResult.n_test must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
            "train_time_s": float(self.train_time_s),
            "inference_time_s": float(self.inference_time_s),
            "n_parameters": int(self.n_parameters),
            "train_accuracy": None if self.train_accuracy is None else float(self.train_accuracy),
            "seed": None if self.seed is None else int(self.seed),
            "predictions": None if self.predictions is None else self.predictions.tolist(),
            "extra": dict(self.extra),
        }



# --------------------------------------------------------------------------------------
# Experiment contracts
# --------------------------------------------------------------------------------------
@dataclass
class TrialResult:
    """Metrics recorded for one candidate encoder configuration."""

    index: int
    params: dict[str, Any]
    accuracy: float = 0.0
    macro_f1: float = 0.0
    train_accuracy: float | None = None
    spike_count: int = 0
    spike_rate_hz: float = 0.0
    spikes_per_second_per_neuron: float = 0.0
    temporal_sparsity: float = 0.0
    information_loss: float = 0.0
    reconstruction_r2: float = 0.0
    compute_cost: float = 0.0
    inference_time_s: float = 0.0
    train_time_s: float = 0.0
    feasible: bool = True
    score: float = 0.0
    rank: int = 0
    is_pareto: bool = False
    status: str = "ok"
    error: str | None = None
    objectives: dict[str, float] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "params": dict(self.params),
            "accuracy": float(self.accuracy),
            "macro_f1": float(self.macro_f1),
            "train_accuracy": None if self.train_accuracy is None else float(self.train_accuracy),
            "spike_count": int(self.spike_count),
            "spike_rate_hz": float(self.spike_rate_hz),
            "spikes_per_second_per_neuron": float(self.spikes_per_second_per_neuron),
            "temporal_sparsity": float(self.temporal_sparsity),
            "information_loss": float(self.information_loss),
            "reconstruction_r2": float(self.reconstruction_r2),
            "compute_cost": float(self.compute_cost),
            "inference_time_s": float(self.inference_time_s),
            "train_time_s": float(self.train_time_s),
            "feasible": bool(self.feasible),
            "score": float(self.score),
            "rank": int(self.rank),
            "is_pareto": bool(self.is_pareto),
            "status": str(self.status),
            "error": self.error,
            "objectives": dict(self.objectives),
            "evaluation": dict(self.evaluation),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrialResult:
        return cls(
            index=int(payload["index"]),
            params=dict(payload.get("params", {})),
            accuracy=float(payload.get("accuracy", 0.0)),
            macro_f1=float(payload.get("macro_f1", 0.0)),
            train_accuracy=payload.get("train_accuracy"),
            spike_count=int(payload.get("spike_count", 0)),
            spike_rate_hz=float(payload.get("spike_rate_hz", 0.0)),
            spikes_per_second_per_neuron=float(payload.get("spikes_per_second_per_neuron", 0.0)),
            temporal_sparsity=float(payload.get("temporal_sparsity", 0.0)),
            information_loss=float(payload.get("information_loss", 0.0)),
            reconstruction_r2=float(payload.get("reconstruction_r2", 0.0)),
            compute_cost=float(payload.get("compute_cost", 0.0)),
            inference_time_s=float(payload.get("inference_time_s", 0.0)),
            train_time_s=float(payload.get("train_time_s", 0.0)),
            feasible=bool(payload.get("feasible", True)),
            score=float(payload.get("score", 0.0)),
            rank=int(payload.get("rank", 0)),
            is_pareto=bool(payload.get("is_pareto", False)),
            status=str(payload.get("status", "ok")),
            error=payload.get("error"),
            objectives=dict(payload.get("objectives", {})),
            evaluation=dict(payload.get("evaluation", {})),
        )


@dataclass
class ExperimentResult:
    """Complete, serialisable record of one optimisation experiment."""

    experiment_id: str
    dataset: dict[str, Any]
    encoder: dict[str, Any]
    search_space: dict[str, Any]
    optimizer: str
    n_trials: int
    seed: int
    objective: str
    weights: dict[str, float]
    accuracy_target: float | None
    spike_budget: int | None
    backend: dict[str, Any]
    preprocess: dict[str, Any]
    window: dict[str, Any]
    trials: list[TrialResult]
    status: str = "completed"
    created_at: str = ""
    duration_s: float = 0.0
    software: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    normalization: dict[str, float] = field(default_factory=dict)
    best_index: int | None = None
    split: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ accessors
    def trial_by_index(self, index: int) -> TrialResult | None:
        for trial in self.trials:
            if trial.index == index:
                return trial
        return None

    @property
    def feasible_trials(self) -> list[TrialResult]:
        return [trial for trial in self.trials if trial.feasible and trial.status == "ok"]

    @property
    def pareto_front(self) -> list[TrialResult]:
        return [trial for trial in self.trials if trial.is_pareto and trial.status == "ok"]

    @property
    def best(self) -> TrialResult | None:
        if self.best_index is None:
            return None
        return self.trial_by_index(self.best_index)

    def top(self, n: int = 5) -> list[TrialResult]:
        """Return the ``n`` highest-ranked trials (feasible first)."""
        ordered = sorted(
            [trial for trial in self.trials if trial.status == "ok"],
            key=lambda trial: (
                0 if trial.feasible else 1,
                trial.rank if trial.rank else 10**9,
                -trial.score,
            ),
        )
        return ordered[:n]

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "created_at": self.created_at,
            "duration_s": self.duration_s,
            "dataset": self.dataset,
            "encoder": self.encoder,
            "search_space": self.search_space,
            "optimizer": self.optimizer,
            "n_trials": self.n_trials,
            "seed": self.seed,
            "objective": self.objective,
            "weights": self.weights,
            "accuracy_target": self.accuracy_target,
            "spike_budget": self.spike_budget,
            "backend": self.backend,
            "preprocess": self.preprocess,
            "window": self.window,
            "split": self.split,
            "normalization": self.normalization,
            "software": self.software,
            "notes": self.notes,
            "best_index": self.best_index,
            "extras": self.extras,
            "trials": [trial.to_dict() for trial in self.trials],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentResult:
        return cls(
            experiment_id=str(payload["experiment_id"]),
            dataset=dict(payload.get("dataset", {})),
            encoder=dict(payload.get("encoder", {})),
            search_space=dict(payload.get("search_space", {})),
            optimizer=str(payload.get("optimizer", "unknown")),
            n_trials=int(payload.get("n_trials", 0)),
            seed=int(payload.get("seed", 0)),
            objective=str(payload.get("objective", "multi")),
            weights=dict(payload.get("weights", {})),
            accuracy_target=payload.get("accuracy_target"),
            spike_budget=payload.get("spike_budget"),
            backend=dict(payload.get("backend", {})),
            preprocess=dict(payload.get("preprocess", {})),
            window=dict(payload.get("window", {})),
            trials=[TrialResult.from_dict(item) for item in payload.get("trials", [])],
            status=str(payload.get("status", "completed")),
            created_at=str(payload.get("created_at", "")),
            duration_s=float(payload.get("duration_s", 0.0)),
            software=dict(payload.get("software", {})),
            notes=str(payload.get("notes", "")),
            normalization=dict(payload.get("normalization", {})),
            best_index=payload.get("best_index"),
            split=dict(payload.get("split", {})),
            extras=dict(payload.get("extras", {})),
        )

    # ------------------------------------------------------------------ export
    def export(self, path: str | Path, *, fmt: str | None = None) -> Path:
        """Export the experiment to JSON, CSV or Markdown."""
        from .reporting import export_result

        return export_result(self, path, fmt=fmt)

    def export_config(self, path: str | Path) -> Path:
        """Export the best encoder configuration as a standalone JSON file."""
        from .reporting import export_config

        return export_config(self, path)
