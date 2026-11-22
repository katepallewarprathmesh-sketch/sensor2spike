"""Latency coding: information is carried by the time of the first spike."""

from __future__ import annotations

import numpy as np

from ..exceptions import ConfigurationError
from ..spaces import Parameter, SearchSpace
from ..utils.validation import require_range
from .base import SpikeEncoder, resample_to_steps
from .registry import register_encoder

__all__ = ["LatencyEncoder"]


@register_encoder
class LatencyEncoder(SpikeEncoder):
    """Population latency coding on fixed frames.

    The signal is split into frames of ``frame_ms``.  Within each frame every
    neuron emits exactly one spike; the delay from the frame start encodes the
    frame's mean amplitude through a Gaussian tuning curve.  Neurons tuned to
    the observed value fire earliest, so the first spike in the frame identifies
    the amplitude.

    ``t_max_ms`` must not exceed ``frame_ms`` so that every spike lands inside
    the frame that produced it (this makes decoding unambiguous).
    """

    name = "latency"
    description = "Population latency coding (first-spike timing on fixed frames)"
    PARAM_NAMES = ("neurons_per_channel", "sigma", "t_max_ms", "frame_ms")

    def __init__(
        self,
        *,
        dt_ms: float = 1.0,
        seed: int | None = None,
        neurons_per_channel: int = 16,
        sigma: float = 0.25,
        t_max_ms: float = 25.0,
        frame_ms: float = 50.0,
    ) -> None:
        super().__init__(dt_ms=dt_ms, seed=seed)
        self.neurons_per_channel = int(
            require_range(neurons_per_channel, "neurons_per_channel", low=2, high=256)
        )
        self.sigma = float(require_range(sigma, "sigma", low=0.01, high=1.0))
        self.t_max_ms = float(require_range(t_max_ms, "t_max_ms", low=1e-3))
        self.frame_ms = float(require_range(frame_ms, "frame_ms", low=1e-3))
        if self.frame_ms + 1e-9 < self.dt_ms:
            raise ConfigurationError("frame_ms must be greater than or equal to dt_ms.")
        if self.t_max_ms > self.frame_ms + 1e-9:
            raise ConfigurationError("t_max_ms must not exceed frame_ms.")
        self.centers_: np.ndarray | None = None

    # ------------------------------------------------------------------ internals
    @property
    def _frame_steps(self) -> int:
        return max(1, int(round(self.frame_ms / 1000.0 / self.dt)))

    @property
    def _t_max_steps(self) -> int:
        return max(1, int(round(self.t_max_ms / 1000.0 / self.dt)))

    def _fit(self, data: np.ndarray) -> None:
        base = self.n_neurons_ if self.n_neurons_ is not None else int(data.shape[1])
        self.n_neurons_ = base * self.neurons_per_channel
        self._neurons_per_channel_ = self.neurons_per_channel
        self._base_channels_ = base
        self.centers_ = np.linspace(0.0, 1.0, self.neurons_per_channel)

    def _transform(self, data: np.ndarray, fs: float, n_steps: int) -> np.ndarray:
        assert self.centers_ is not None
        values = self._normalize(resample_to_steps(data, fs, self.dt, n_steps))
        n_channels = values.shape[1]
        per_channel = self._neurons_per_channel_
        frame = self._frame_steps
        spikes = np.zeros((n_steps, self.n_neurons_), dtype=np.uint8)
        for start in range(0, n_steps, frame):
            stop = min(n_steps, start + frame)
            frame_value = values[start:stop].mean(axis=0)
            squared = (frame_value[:, None] - self.centers_[None, :]) ** 2
            activation = np.exp(-squared / (2.0 * self.sigma**2))
            delays = np.rint((1.0 - activation) * self._t_max_steps).astype(np.int64)
            indices = np.clip(start + delays, start, stop - 1)
            columns = np.arange(per_channel)
            for channel in range(n_channels):
                spikes[
                    indices[channel], channel * per_channel + columns
                ] = 1
        return spikes

    def decode(self, spikes):
        assert self.centers_ is not None and self.n_channels_ is not None
        n_channels = int(self._base_channels_)
        per_channel = self._neurons_per_channel_
        frame = self._frame_steps
        decoded = np.zeros((spikes.n_steps, n_channels), dtype=np.float64)
        for start in range(0, spikes.n_steps, frame):
            stop = min(spikes.n_steps, start + frame)
            for channel in range(n_channels):
                block = spikes.spikes[start:stop, channel * per_channel : (channel + 1) * per_channel]
                if not block.any():
                    continue
                first_step = int(np.argmax(block.any(axis=1)))
                winner = int(np.argmax(block[first_step]))
                decoded[start:stop, channel] = self.centers_[winner]
        return decoded

    def _channel_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for channel in range(self._base_channels_ or 0):
            for neuron in range(self._neurons_per_channel_):
                names.append(f"latency_ch{channel:02d}_n{neuron:02d}")
        return tuple(names)

    def param_space(self) -> SearchSpace:
        return SearchSpace(
            {
                "neurons_per_channel": Parameter(
                    "neurons_per_channel",
                    "int",
                    4,
                    32,
                    description="Neurons per channel (tuning curves)",
                ),
                "sigma": Parameter(
                    "sigma", "float", 0.05, 0.5, log=True, description="Tuning curve width"
                ),
                # Kept disjoint so that ``t_max_ms <= frame_ms`` holds for every
                # combination the search space can generate.
                "t_max_ms": Parameter(
                    "t_max_ms", "float", 10.0, 25.0, description="Maximum spike delay"
                ),
                "frame_ms": Parameter(
                    "frame_ms", "float", 25.0, 100.0, description="Frame length"
                ),
                "dt_ms": Parameter("dt_ms", "choice", choices=(0.5, 1.0, 2.0, 5.0), description="Spike bin width (ms)"),
            }
        )
