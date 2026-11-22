"""Abstract encoder interface and shared utilities.

Every encoder follows the same contract::

    encoder = RateEncoder(rate_max_hz=200, dt_ms=1.0)
    encoder.fit(signal, fs=1000)          # learn per-channel scaling
    spikes  = encoder.transform(signal)   # -> SpikeTrain(n_steps, n_neurons)
    loss    = encoder.information_loss(spikes, signal)

Encoders are *stateful* only with respect to the signal range observed during
``fit``; all other behaviour is deterministic for a fixed seed.
"""

from __future__ import annotations

import functools
import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from ..analysis.dsp import exp_smooth, reconstruction_r2, resample_to_steps
from ..contracts import SpikeTrain, WindowSet
from ..exceptions import ConfigurationError, EncodingError
from ..spaces import SearchSpace
from ..utils.validation import as_2d_float_array, reject_unknown_keys, require_positive

__all__ = [
    "SpikeEncoder",
    "exp_smooth",
    "reconstruction_r2",
    "resample_to_steps",
    "encode_windows",
]



def encode_windows(encoder: SpikeEncoder, windows: WindowSet | np.ndarray) -> list[SpikeTrain]:
    """Encode every window of a :class:`WindowSet` (or 3-D array) into a train.

    A stacked array of shape ``(n_windows, n_samples, n_channels)`` is encoded
    with the sampling rate supplied at the last ``fit`` call.
    """
    if isinstance(windows, np.ndarray):
        if windows.ndim == 2:
            windows = windows[None, ...]
        if windows.ndim != 3:
            raise EncodingError(
                "transform_windows expects a 3-D array (n_windows, n_samples, n_channels) "
                f"or a WindowSet; got shape {windows.shape}."
            )
        if not encoder.is_fitted:
            raise EncodingError(
                "transform_windows needs a fitted encoder; call fit(data, fs) first."
            )
        return [encoder.transform(window) for window in windows]
    return [encoder.transform(window, fs=windows.sampling_frequency) for window in windows]


_MISSING = object()


def _constructor_defaults(cls: type) -> dict[str, Any]:
    """Return the default value of every keyword parameter of ``cls.__init__``."""
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return {}
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


# --------------------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------------------
class SpikeEncoder(ABC):
    """Base class for all spike encoders.

    Subclasses implement :meth:`_fit` (optional) and :meth:`_transform`.
    """

    name: ClassVar[str] = "base"
    description: ClassVar[str] = ""
    PARAM_NAMES: ClassVar[tuple[str, ...]] = ()

    #: Keyword arguments explicitly supplied to the constructor.  Used by
    #: :meth:`to_space` so only committed choices narrow the search space.
    _explicit: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, *, dt_ms: float = 1.0, seed: int | None = None) -> None:
        dt_ms = float(dt_ms)
        if not math.isfinite(dt_ms) or dt_ms <= 0:
            raise ConfigurationError("dt_ms must be a positive number.")
        self.dt = dt_ms / 1000.0
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fs_: float | None = None
        self.n_channels_: int | None = None
        self.n_neurons_: int | None = None
        self.scale_min_: np.ndarray | None = None
        self.scale_max_: np.ndarray | None = None

    # ------------------------------------------------------------------ properties
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        original = cls.__init__
        if getattr(original, "_records_explicit", False):
            return

        @functools.wraps(original)
        def _wrapped(self, *args: Any, **call_kwargs: Any) -> None:
            named = {
                parameter.name
                for parameter in inspect.signature(original).parameters.values()
                if parameter.name != "self"
            }
            object.__setattr__(self, "_explicit", frozenset(call_kwargs) & named)
            original(self, *args, **call_kwargs)

        _wrapped._records_explicit = True  # type: ignore[attr-defined]
        cls.__init__ = _wrapped

    @property
    def dt_ms(self) -> float:
        return self.dt * 1000.0

    @property
    def is_fitted(self) -> bool:
        return self.n_channels_ is not None

    @property
    def n_neurons(self) -> int:
        """Number of output neurons (available after ``fit``)."""
        self._require_fitted()
        return int(self.n_neurons_)

    def __repr__(self) -> str:
        params = ", ".join(f"{key}={getattr(self, key)!r}" for key in self.PARAM_NAMES)
        return f"{type(self).__name__}(dt_ms={self.dt_ms:g}, {params})"

    # ------------------------------------------------------------------ fit / transform
    def fit(self, data: Any, fs: float | None = None) -> SpikeEncoder:
        """Learn the per-channel value range from ``data``.

        Parameters
        ----------
        data:
            Array of shape ``(n_samples, n_channels)``.
        fs:
            Sampling rate in Hz.  May be omitted when it was already supplied
            by a previous ``fit`` call.
        """
        array = as_2d_float_array(data, name=f"{self.name}.data")
        if fs is not None:
            self.fs_ = float(require_positive(fs, f"{self.name}.fs"))
        if self.fs_ is None:
            raise EncodingError(
                f"{self.name}: sampling frequency is required on the first fit() call."
            )
        self.n_channels_ = int(array.shape[1])
        self.scale_min_ = array.min(axis=0).astype(np.float64)
        self.scale_max_ = array.max(axis=0).astype(np.float64)
        if self.n_neurons_ is not None:
            # Multi-window batches concatenate along the neuron axis, so the
            # fitted neuron count must be reset before every fit.
            self.n_neurons_ = None
        self._fit(array)
        if self.n_neurons_ is None:
            self.n_neurons_ = self.n_channels_
        return self

    @abstractmethod
    def _fit(self, data: np.ndarray) -> None:
        """Encoder-specific initialisation.  Must set ``self.n_neurons_``."""

    def _transform(self, data: np.ndarray, fs: float, n_steps: int) -> np.ndarray:
        """Encode ``data`` into a binary matrix of shape ``(n_steps, n_neurons)``."""
        raise NotImplementedError

    def fit_transform(self, data: Any, fs: float | None = None) -> SpikeTrain:
        return self.fit(data, fs=fs).transform(data)

    def transform(self, data: Any, fs: float | None = None) -> SpikeTrain:
        """Encode ``data`` into a :class:`SpikeTrain`."""
        self._require_fitted()
        array = as_2d_float_array(data, name=f"{self.name}.data")
        if array.shape[1] != self.n_channels_:
            raise EncodingError(
                f"{self.name} was fitted on {self.n_channels_} channel(s) but received "
                f"{array.shape[1]}."
            )
        effective_fs = self.fs_ if fs is None else float(require_positive(fs, f"{self.name}.fs"))
        n_steps = max(1, int(round(array.shape[0] / effective_fs / self.dt)))
        spikes = np.asarray(self._transform(array, effective_fs, n_steps))
        if spikes.ndim == 1:
            spikes = spikes.reshape(-1, 1)
        if spikes.shape[0] != n_steps:
            raise EncodingError(
                f"{self.name} produced {spikes.shape[0]} steps but {n_steps} were expected."
            )
        spikes = (spikes > 0).astype(np.uint8)
        if spikes.shape[1] != self.n_neurons_:
            raise EncodingError(
                f"{self.name} produced {spikes.shape[1]} neurons but {self.n_neurons_} were expected."
            )
        return SpikeTrain(
            spikes=spikes,
            dt=self.dt,
            channel_names=self._channel_names(),
            encoder=self.name,
            metadata={
                "fs": float(effective_fs),
                "n_channels_in": int(self.n_channels_),
                "n_steps": int(n_steps),
                "dt_ms": self.dt_ms,
            },
        )

    def transform_windows(self, windows: WindowSet) -> list[SpikeTrain]:
        """Encode every window in a :class:`WindowSet` (one train per window)."""
        return encode_windows(self, windows)

    # ------------------------------------------------------------------ decoding / metrics
    def decode(self, spikes: SpikeTrain) -> np.ndarray | None:
        """Reconstruct a continuous estimate of the input signal from spikes.

        The default implementation is an exponential low-pass of the spike
        matrix.  Subclasses override this when a better inverse exists (for
        example cumulative summation for delta modulation).
        """
        smoothed = exp_smooth(spikes.spikes.astype(np.float64), spikes.dt, tau=0.02)
        return smoothed

    def information_loss(self, spikes: SpikeTrain, reference: Any) -> float | None:
        """Return ``1 - R^2`` between the decoded signal and ``reference``.

        A value of 0 means the spike train carries a linearly recoverable copy of
        the reference; 1 means no linear information is preserved.
        """
        decoded = self.decode(spikes)
        if decoded is None:
            return None
        array = as_2d_float_array(reference, name=f"{self.name}.reference")
        if self.fs_ is None:  # pragma: no cover - transform always sets this
            return None
        target = resample_to_steps(array, self.fs_, spikes.dt, spikes.n_steps)
        r2 = reconstruction_r2(decoded, target)
        return float(1.0 - np.clip(r2, 0.0, 1.0))

    # ------------------------------------------------------------------ configuration
    def get_config(self) -> dict[str, Any]:
        """Return a JSON-safe configuration (encoder name, dt, seed, params)."""
        config: dict[str, Any] = {
            "encoder": self.name,
            "dt_ms": self.dt_ms,
            "seed": self.seed,
        }
        for key in self.PARAM_NAMES:
            value = getattr(self, key)
            if isinstance(value, np.generic):
                value = value.item()
            config[key] = value
        return config

    def get_state(self) -> dict[str, Any]:
        """Configuration plus fitted scaling statistics (fully reproducible)."""
        state = self.get_config()
        state["n_channels"] = self.n_channels_
        state["n_neurons"] = self.n_neurons_
        state["scale_min"] = None if self.scale_min_ is None else self.scale_min_.tolist()
        state["scale_max"] = None if self.scale_max_ is None else self.scale_max_.tolist()
        return state

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> SpikeEncoder:
        """Build an unfitted encoder from a configuration mapping."""
        data = dict(config)
        encoder_name = data.pop("encoder", None)
        if encoder_name is not None and encoder_name != cls.name:
            raise ConfigurationError(
                f"Cannot build encoder {cls.name!r} from config for {encoder_name!r}."
            )
        for key in ("n_channels", "scale_min", "scale_max"):
            data.pop(key, None)
        allowed = {"dt_ms", "seed", *cls.PARAM_NAMES}
        reject_unknown_keys(data, allowed, name=f"{cls.name} config")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ConfigurationError(f"Invalid configuration for {cls.name}: {exc}") from exc

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> SpikeEncoder:
        """Restore a fitted encoder, including its scaling statistics."""
        data = dict(state)
        n_channels = data.pop("n_channels", None)
        n_neurons = data.pop("n_neurons", None)
        scale_min = data.pop("scale_min", None)
        scale_max = data.pop("scale_max", None)
        # Anything left after removing the config keys and the scalar fitted
        # statistics belongs to a subclass (e.g. the delta encoder's baseline).
        # ``_restore_state`` lets it restore those without the base class knowing.
        config_keys = {"encoder", "dt_ms", "seed", *cls.PARAM_NAMES}
        extras = {key: data.pop(key) for key in list(data) if key not in config_keys}
        encoder = cls.from_config(data)
        if n_channels is not None and scale_min is not None and scale_max is not None:
            encoder.n_channels_ = int(n_channels)
            encoder.scale_min_ = np.asarray(scale_min, dtype=np.float64)
            encoder.scale_max_ = np.asarray(scale_max, dtype=np.float64)
            encoder.n_neurons_ = (
                encoder._neuron_count(int(n_channels)) if n_neurons is None else int(n_neurons)
            )
        if extras:
            encoder._restore_state(extras)
        return encoder

    def _restore_state(self, extras: Mapping[str, Any]) -> None:
        """Hook for subclasses to restore fitted attributes saved in ``get_state``.

        The default implementation ignores extras; subclasses that persist their
        own fitted attributes override it.
        """
        del extras

    def _neuron_count(self, n_channels: int) -> int:
        """Number of output neurons for ``n_channels`` input channels."""
        return n_channels

    # ------------------------------------------------------------------ search space
    def param_space(self) -> SearchSpace:
        """Default search space for this encoder (overridable per instance)."""
        return SearchSpace()

    def to_space(self, overrides: Mapping[str, Any] | None = None) -> SearchSpace:
        """Return the default space merged with ``overrides``.

        Parameters whose value was explicitly set on this instance (i.e. differs
        from the constructor default) act as *fixed* values: the caller has
        already committed to them, so they narrow the space before ``overrides``
        are applied.
        """
        space = self.param_space()
        explicit = self._explicit
        fixed = {
            key: getattr(self, key)
            for key in space.keys()
            if key in explicit and hasattr(self, key)
        }
        return space.merge(fixed).merge(overrides)

    # ------------------------------------------------------------------ helpers
    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise EncodingError(f"{self.name}.transform() called before fit().")

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        assert self.scale_min_ is not None and self.scale_max_ is not None
        low = self.scale_min_
        span = self.scale_max_ - low
        span = np.where(span < 1e-12, 1.0, span)
        normalized = (data - low) / span
        return np.clip(normalized, 0.0, 1.0)

    def _steps(self, data: np.ndarray, fs: float) -> np.ndarray:
        return resample_to_steps(data, fs, self.dt, self._n_steps(data, fs))

    def _n_steps(self, data: np.ndarray, fs: float) -> int:
        return max(1, int(round(data.shape[0] / fs / self.dt)))

    def _channel_names(self) -> tuple[str, ...]:
        return tuple(f"{self.name}_ch{index:02d}" for index in range(self.n_neurons_ or 0))

    def _reset_rng(self) -> None:
        self.rng = np.random.default_rng(self.seed)
