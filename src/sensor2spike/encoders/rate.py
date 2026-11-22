"""Rate coding: signal amplitude is mapped to a firing rate."""

from __future__ import annotations

import numpy as np

from ..contracts import SpikeTrain
from ..spaces import Parameter, SearchSpace
from ..utils.validation import require_choice, require_range
from .base import SpikeEncoder, exp_smooth, resample_to_steps
from .registry import register_encoder

__all__ = ["RateEncoder"]


@register_encoder
class RateEncoder(SpikeEncoder):
    """Encode normalised amplitude as a firing rate.

    Each channel drives one neuron.  With ``mode="poisson"`` spikes are drawn
    from a Bernoulli process; with ``mode="deterministic"`` (the default) the
    fractional rate is carried over between steps with error diffusion, which
    keeps the output reproducible and free of shot noise.
    """

    name = "rate"
    description = "Amplitude-to-rate coding (Poisson or deterministic error diffusion)"
    PARAM_NAMES = ("rate_max_hz", "mode", "gain")

    def __init__(
        self,
        *,
        dt_ms: float = 1.0,
        seed: int | None = None,
        rate_max_hz: float = 100.0,
        mode: str = "deterministic",
        gain: float = 1.0,
    ) -> None:
        super().__init__(dt_ms=dt_ms, seed=seed)
        self.rate_max_hz = float(require_range(rate_max_hz, "rate_max_hz", low=1e-6))
        self.mode = str(require_choice(mode, "mode", ("poisson", "deterministic")))
        self.gain = float(require_range(gain, "gain", low=1e-6))

    def _fit(self, data: np.ndarray) -> None:
        self.n_neurons_ = int(data.shape[1])

    def _transform(self, data: np.ndarray, fs: float, n_steps: int) -> np.ndarray:
        values = self._normalize(resample_to_steps(data, fs, self.dt, n_steps))
        probability = np.clip(values * self.rate_max_hz * self.gain * self.dt, 0.0, 1.0)
        if self.mode == "poisson":
            return (self.rng.random(values.shape) < probability).astype(np.uint8)

        spikes = np.zeros(values.shape, dtype=np.uint8)
        accumulator = np.zeros(values.shape[1], dtype=np.float64)
        for step in range(values.shape[0]):
            accumulator = accumulator + probability[step]
            fired = np.floor(accumulator + 1e-9)
            accumulator -= fired
            spikes[step] = fired.astype(np.uint8)
        return spikes

    def decode(self, spikes: SpikeTrain) -> np.ndarray:
        smoothed = exp_smooth(spikes.spikes.astype(np.float64), spikes.dt, tau=0.02)
        expected = self.rate_max_hz * self.gain * spikes.dt
        if expected <= 0:  # pragma: no cover - guarded by construction
            return smoothed
        return np.clip(smoothed / expected, 0.0, 1.0)

    def param_space(self) -> SearchSpace:
        return SearchSpace(
            {
                "rate_max_hz": Parameter(
                    "rate_max_hz",
                    "float",
                    20.0,
                    500.0,
                    log=True,
                    description="Maximum firing rate at full-scale amplitude",
                ),
                "dt_ms": Parameter(
                    "dt_ms", "choice", choices=(0.5, 1.0, 2.0, 5.0), description="Spike bin width (ms)"
                ),
                "mode": Parameter(
                    "mode", "choice", choices=("deterministic", "poisson"), description="Spike generation"
                ),
                "gain": Parameter(
                    "gain", "float", 0.5, 2.0, description="Linear gain applied before clipping"
                ),
            }
        )
