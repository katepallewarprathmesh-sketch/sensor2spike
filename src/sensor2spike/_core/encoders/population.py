"""Population coding: each channel is represented by a bank of tuned neurons."""

from __future__ import annotations

import numpy as np

from ..contracts import SpikeTrain
from ..spaces import Parameter, SearchSpace
from ..utils.validation import require_choice, require_range
from .base import SpikeEncoder, exp_smooth, resample_to_steps
from .registry import register_encoder

__all__ = ["PopulationEncoder"]


@register_encoder
class PopulationEncoder(SpikeEncoder):
    """Gaussian tuning-curve population coding.

    Each channel drives ``neurons_per_channel`` neurons with Gaussian receptive fields
    centred on ``[0, 1]``.  Neurons fire at a rate proportional to their tuning
    response, so the pattern across the population encodes the amplitude.  This
    is more spike-hungry than rate coding but far more robust to spike noise.
    """

    name = "population"
    description = "Gaussian tuning-curve population coding"
    PARAM_NAMES = ("neurons_per_channel", "sigma", "rate_max_hz", "mode")

    def __init__(
        self,
        *,
        dt_ms: float = 1.0,
        seed: int | None = None,
        neurons_per_channel: int = 16,
        sigma: float = 0.2,
        rate_max_hz: float = 200.0,
        mode: str = "deterministic",
    ) -> None:
        super().__init__(dt_ms=dt_ms, seed=seed)
        self.neurons_per_channel = int(
            require_range(neurons_per_channel, "neurons_per_channel", low=2, high=256)
        )
        self.sigma = float(require_range(sigma, "sigma", low=0.01, high=1.0))
        self.rate_max_hz = float(require_range(rate_max_hz, "rate_max_hz", low=1e-6))
        self.mode = str(require_choice(mode, "mode", ("poisson", "deterministic")))
        self.centers_: np.ndarray | None = None

    def _fit(self, data: np.ndarray) -> None:
        base = self.n_neurons_ if self.n_neurons_ is not None else int(data.shape[1])
        self._base_channels_ = base
        self._neurons_per_channel_ = self.neurons_per_channel
        self.n_neurons_ = base * self.neurons_per_channel
        self.centers_ = np.linspace(0.0, 1.0, self.neurons_per_channel)

    def _transform(self, data: np.ndarray, fs: float, n_steps: int) -> np.ndarray:
        assert self.centers_ is not None
        values = self._normalize(resample_to_steps(data, fs, self.dt, n_steps))
        squared = (values[:, :, None] - self.centers_[None, None, :]) ** 2
        activation = np.exp(-squared / (2.0 * self.sigma**2))
        probability = np.clip(activation * self.rate_max_hz * self.dt, 0.0, 1.0)
        flat = probability.reshape(probability.shape[0], -1)
        if self.mode == "poisson":
            return (self.rng.random(flat.shape) < flat).astype(np.uint8)

        spikes = np.zeros(flat.shape, dtype=np.uint8)
        accumulator = np.zeros(flat.shape[1], dtype=np.float64)
        for step in range(flat.shape[0]):
            accumulator = accumulator + flat[step]
            fired = np.floor(accumulator + 1e-9)
            accumulator -= fired
            spikes[step] = fired.astype(np.uint8)
        return spikes

    def decode(self, spikes: SpikeTrain) -> np.ndarray:
        assert self.centers_ is not None and self.n_channels_ is not None
        n_channels = int(self.n_channels_)
        smoothed = exp_smooth(spikes.spikes.astype(np.float64), spikes.dt, tau=0.02)
        rates = smoothed.reshape(spikes.n_steps, n_channels, self._neurons_per_channel_)
        numerator = (rates * self.centers_[None, None, :]).sum(axis=2)
        denominator = rates.sum(axis=2)
        return numerator / np.maximum(denominator, 1e-9)

    def _channel_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for channel in range(self.n_channels_ or 0):
            for neuron in range(self._neurons_per_channel_):
                names.append(f"pop_ch{channel:02d}_n{neuron:02d}")
        return tuple(names)

    def param_space(self) -> SearchSpace:
        return SearchSpace(
            {
                "neurons_per_channel": Parameter(
                    "neurons_per_channel", "int", 4, 32, description="Neurons per channel"
                ),
                "sigma": Parameter(
                    "sigma", "float", 0.05, 0.5, log=True, description="Tuning curve width"
                ),
                "rate_max_hz": Parameter(
                    "rate_max_hz", "float", 20.0, 500.0, log=True, description="Peak firing rate"
                ),
                "mode": Parameter(
                    "mode", "choice", choices=("deterministic", "poisson"), description="Spike generation"
                ),
                "dt_ms": Parameter("dt_ms", "choice", choices=(0.5, 1.0, 2.0, 5.0), description="Spike bin width (ms)"),
            }
        )
