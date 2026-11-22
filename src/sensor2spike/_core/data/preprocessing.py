"""Signal preprocessing: resampling, detrending, filtering, normalisation.

All operations are deterministic and implemented with NumPy/SciPy only, so
preprocessing never becomes a hidden source of run-to-run variation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from typing import Any

import numpy as np
from scipy import signal

from ..contracts import SensorDataset
from ..exceptions import ConfigurationError
from ..utils.validation import reject_unknown_keys, require_choice, require_range

__all__ = ["PreprocessConfig", "preprocess"]

_NORMALIZATION_MODES = ("none", "minmax", "zscore")


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for the preprocessing stage.

    Attributes
    ----------
    enabled:
        When ``False`` the dataset is returned unchanged.
    detrend:
        Remove a linear trend from each channel.
    highpass_hz, lowpass_hz:
        Butterworth cut-off frequencies.  Set both for a band-pass response.
    filter_order:
        Butterworth filter order (per edge).
    normalize:
        ``"none"``, ``"minmax"`` (scale each channel to ``[0, 1]``) or
        ``"zscore"``.
    resample_hz:
        Target sample rate.  Uses polyphase resampling with an anti-aliasing
        filter, so decimation is safe.
    """

    enabled: bool = True
    detrend: bool = True
    highpass_hz: float | None = None
    lowpass_hz: float | None = None
    filter_order: int = 4
    normalize: str = "minmax"
    resample_hz: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("PreprocessConfig.enabled must be a boolean.")
        if not isinstance(self.detrend, bool):
            raise ConfigurationError("PreprocessConfig.detrend must be a boolean.")
        require_range(self.filter_order, "PreprocessConfig.filter_order", low=1, high=12)
        require_choice(self.normalize, "PreprocessConfig.normalize", _NORMALIZATION_MODES)
        for name in ("highpass_hz", "lowpass_hz", "resample_hz"):
            value = getattr(self, name)
            if value is not None:
                require_range(value, f"PreprocessConfig.{name}", low=1e-6)
        if self.highpass_hz is not None and self.lowpass_hz is not None:
            if self.highpass_hz >= self.lowpass_hz:
                raise ConfigurationError(
                    "PreprocessConfig.highpass_hz must be lower than lowpass_hz."
                )

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any] | None, *, strict: bool = True) -> PreprocessConfig:
        """Build a config from a mapping, rejecting unknown keys by default."""
        if mapping is None:
            return cls()
        if not isinstance(mapping, Mapping):
            raise ConfigurationError("PreprocessConfig.from_dict expects a mapping.")
        if strict:
            reject_unknown_keys(mapping, [f.name for f in fields(cls)], name="preprocess")
        return cls(**dict(mapping))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preprocess(dataset: SensorDataset, config: PreprocessConfig | Mapping[str, Any] | None = None) -> SensorDataset:
    """Apply the preprocessing pipeline and return a new dataset.

    The input dataset is never modified.  Labels and group identifiers follow
    the resampled signal using nearest-neighbour index mapping, so window labels
    stay aligned with the signal.
    """
    config = config if isinstance(config, PreprocessConfig) else PreprocessConfig.from_dict(config)
    if not config.enabled:
        return dataset.copy()

    data = dataset.data.astype(np.float64)
    fs = float(dataset.sampling_frequency)
    labels = None if dataset.labels is None else dataset.labels.copy()
    groups = None if dataset.groups is None else dataset.groups.copy()
    timestamps = None if dataset.timestamps is None else dataset.timestamps.copy()

    if config.resample_hz is not None and abs(config.resample_hz - fs) > 1e-9:
        data, fs, indices = _resample(data, fs, config.resample_hz)
        labels = None if labels is None else labels[indices]
        groups = None if groups is None else groups[indices]
        timestamps = None if timestamps is None else timestamps[indices]

    if config.detrend:
        data = signal.detrend(data, axis=0, type="linear")

    if config.highpass_hz is not None or config.lowpass_hz is not None:
        data = _butter_filter(
            data,
            fs,
            highpass_hz=config.highpass_hz,
            lowpass_hz=config.lowpass_hz,
            order=config.filter_order,
        )

    data = _normalize(data, config.normalize)

    metadata = dict(dataset.metadata)
    metadata["preprocessing"] = {
        **config.to_dict(),
        "effective_sampling_frequency": fs,
        "n_samples": int(data.shape[0]),
    }
    return SensorDataset(
        data=data.astype(np.float32),
        sampling_frequency=fs,
        channel_names=dataset.channel_names,
        timestamps=timestamps,
        labels=labels,
        groups=groups,
        name=dataset.name,
        sensor_type=dataset.sensor_type,
        metadata=metadata,
        version=dataset.version,
    )


# --------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------
def _resample(data: np.ndarray, fs: float, target_fs: float) -> tuple[np.ndarray, float, np.ndarray]:
    ratio = Fraction(target_fs / fs).limit_denominator(10_000)
    up, down = ratio.numerator, ratio.denominator
    resampled = signal.resample_poly(data, up=up, down=down, axis=0)
    n_out = resampled.shape[0]
    # Nearest-neighbour mapping of per-sample labels onto the new time base.
    if n_out <= 1:
        indices = np.zeros(n_out, dtype=np.int64)
    else:
        source_positions = np.linspace(0, data.shape[0] - 1, n_out)
        indices = np.rint(source_positions).astype(np.int64)
    return resampled, float(target_fs), indices


def _butter_filter(
    data: np.ndarray,
    fs: float,
    *,
    highpass_hz: float | None,
    lowpass_hz: float | None,
    order: int,
) -> np.ndarray:
    nyquist = fs / 2.0
    if highpass_hz is not None and highpass_hz >= nyquist:
        raise ConfigurationError(
            f"highpass_hz ({highpass_hz}) must be below the Nyquist frequency ({nyquist:g} Hz)."
        )
    if lowpass_hz is not None and lowpass_hz >= nyquist:
        raise ConfigurationError(
            f"lowpass_hz ({lowpass_hz}) must be below the Nyquist frequency ({nyquist:g} Hz)."
        )
    if highpass_hz is not None and lowpass_hz is not None:
        sos = signal.butter(order, [highpass_hz, lowpass_hz], btype="bandpass", fs=fs, output="sos")
    elif highpass_hz is not None:
        sos = signal.butter(order, highpass_hz, btype="highpass", fs=fs, output="sos")
    else:
        sos = signal.butter(order, lowpass_hz, btype="lowpass", fs=fs, output="sos")  # type: ignore[arg-type]
    return signal.sosfiltfilt(sos, data, axis=0)


def _normalize(data: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return data
    if mode == "minmax":
        low = data.min(axis=0, keepdims=True)
        high = data.max(axis=0, keepdims=True)
        span = high - low
        span = np.where(span < 1e-12, 1.0, span)
        return (data - low) / span
    if mode == "zscore":
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        return (data - mean) / std
    raise ConfigurationError(f"Unknown normalization mode {mode!r}.")
