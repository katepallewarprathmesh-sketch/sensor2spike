"""Threshold coding: spikes are emitted when a signal crosses a threshold."""

from __future__ import annotations

import numpy as np

from ..spaces import Parameter, SearchSpace
from ..utils.validation import require_choice, require_range
from .base import SpikeEncoder, exp_smooth, resample_to_steps
from .registry import register_encoder

__all__ = ["ThresholdEncoder"]


@register_encoder
class ThresholdEncoder(SpikeEncoder):
    """Emit a spike on upward threshold crossings or whenever above threshold.

    ``mode="crossing"`` produces one spike per upward crossing (event-like,
    sparse) on a single neuron per channel.  ``mode="signed"`` adds a second
    neuron per channel that fires on downward crossings, so the sign of the
    change is preserved.  ``mode="level"`` produces a spike for every step above
    threshold (dense, high-activity).  ``refractory_ms`` enforces a minimum
    spacing between spikes on the same neuron.
    """

    name = "threshold"
    description = "Threshold-crossing (up/signed) or level coding with refractory period"
    PARAM_NAMES = ("threshold", "mode", "refractory_ms")

    def __init__(
        self,
        *,
        dt_ms: float = 1.0,
        seed: int | None = None,
        threshold: float = 0.5,
        mode: str = "crossing",
        refractory_ms: float = 0.0,
    ) -> None:
        super().__init__(dt_ms=dt_ms, seed=seed)
        self.threshold = float(require_range(threshold, "threshold", low=0.0, high=1.0))
        self.mode = str(require_choice(mode, "mode", ("crossing", "signed", "level")))
        self.refractory_ms = float(require_range(refractory_ms, "refractory_ms", low=0.0))

    def _fit(self, data: np.ndarray) -> None:
        base = self.n_neurons_ if self.n_neurons_ is not None else int(data.shape[1])
        self._base_channels_ = base
        self.n_neurons_ = base * (2 if self.mode in {"signed", "level"} else 1)

    def _transform(self, data: np.ndarray, fs: float, n_steps: int) -> np.ndarray:
        # An excursion starts wherever the signal sits at or above the threshold,
        # regardless of the absolute offset introduced by the previous sample.
        values = self._normalize(resample_to_steps(data, fs, self.dt, n_steps))
        above = values >= self.threshold
        previous = np.vstack([np.zeros((1, values.shape[1])), values[:-1]])
        if self.mode == "level":
            raw = np.concatenate([above, ~above], axis=1)
        elif self.mode == "signed":
            raw = np.concatenate(
                [above & (previous < self.threshold), (previous >= self.threshold) & ~above],
                axis=1,
            )
        else:
            raw = above & (previous < self.threshold)

        refractory_steps = int(round(self.refractory_ms / 1000.0 / self.dt))
        if refractory_steps <= 0:
            return raw.astype(np.uint8)

        spikes = np.zeros(raw.shape, dtype=np.uint8)
        last_fired = np.full(raw.shape[1], -(refractory_steps + 1), dtype=np.int64)
        for step in range(raw.shape[0]):
            allowed = raw[step] & ((step - last_fired) >= refractory_steps)
            last_fired = np.where(allowed, step, last_fired)
            spikes[step] = allowed.astype(np.uint8)
        return spikes

    def decode(self, spikes):
        matrix = spikes.spikes.astype(np.float64)
        n_channels = int(getattr(self, "_base_channels_", matrix.shape[1]))
        if matrix.shape[1] == 2 * n_channels:
            # Symmetric banks: recombine up/down activity into one signal.
            up = exp_smooth(matrix[:, 0::2], spikes.dt, tau=0.02)
            down = exp_smooth(matrix[:, 1::2], spikes.dt, tau=0.02)
            matrix = up - down
        else:
            matrix = exp_smooth(matrix, spikes.dt, tau=0.02)
        peak = float(np.abs(matrix).max()) if matrix.size else 0.0
        if peak <= 1e-12:
            return matrix
        return np.clip(matrix / peak, 0.0, 1.0)

    def _channel_names(self) -> tuple[str, ...]:
        base = getattr(self, "_base_channels_", self.n_channels_ or 0)
        names: list[str] = []
        for channel in range(base):
            if self.mode == "crossing":
                names.append(f"thr_ch{channel:02d}")
            else:
                names.extend([f"thr_ch{channel:02d}_up", f"thr_ch{channel:02d}_down"])
        return tuple(names)

    def param_space(self) -> SearchSpace:
        return SearchSpace(
            {
                "threshold": Parameter(
                    "threshold",
                    "float",
                    0.1,
                    0.9,
                    description="Normalised threshold level",
                ),
                "mode": Parameter(
                    "mode",
                    "choice",
                    choices=("crossing", "signed", "level"),
                    description="Spike rule",
                ),
                "refractory_ms": Parameter(
                    "refractory_ms",
                    "float",
                    0.0,
                    50.0,
                    description="Minimum spacing between spikes",
                ),
                "dt_ms": Parameter("dt_ms", "choice", choices=(0.5, 1.0, 2.0, 5.0), description="Spike bin width (ms)"),
            }
        )
