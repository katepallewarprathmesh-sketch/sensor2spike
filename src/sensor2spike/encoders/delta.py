"""Delta (step-forward) modulation: spikes encode signal changes."""

from __future__ import annotations

import numpy as np

from ..spaces import Parameter, SearchSpace
from ..utils.validation import require_range
from .base import SpikeEncoder, resample_to_steps
from .registry import register_encoder

__all__ = ["DeltaEncoder"]


@register_encoder
class DeltaEncoder(SpikeEncoder):
    """Step-forward / step-back (sigma-delta) modulation.

    Each channel drives two neurons: an *up* neuron that fires when the tracked
    error becomes positive and a *down* neuron when it becomes negative.  The
    encoder keeps a running reconstruction and only emits a spike when the
    difference between the signal and that reconstruction reaches ``threshold``,
    then moves the reconstruction by exactly one step.

    This error-feedback form (rather than open-loop thresholding of the raw
    derivative) is what makes the promise in the README true: decoding is a
    cumulative sum of the spikes, which reproduces the signal up to the
    quantisation step and an additive constant.  Open-loop thresholding loses the
    sub-threshold increments and drifts, because the decoder integrates errors the
    encoder never corrects.
    """

    name = "delta"
    description = "Delta modulation with up/down neurons per channel"
    PARAM_NAMES = ("threshold", "max_steps_per_sample")

    def __init__(
        self,
        *,
        dt_ms: float = 1.0,
        seed: int | None = None,
        threshold: float = 0.1,
        max_steps_per_sample: int = 4,
    ) -> None:
        super().__init__(dt_ms=dt_ms, seed=seed)
        self.threshold = float(require_range(threshold, "threshold", low=1e-4, high=1.0))
        self.max_steps_per_sample = int(
            require_range(max_steps_per_sample, "max_steps_per_sample", low=1, high=16)
        )

    def _fit(self, data: np.ndarray) -> None:
        self.n_neurons_ = int(data.shape[1]) * 2
        # The decoder rebuilds an absolute level from cumulative spikes, so both
        # sides must agree on where integration starts.  Learning it from the data
        # (rather than assuming 0 or 0.5) makes a flat signal cost zero spikes and
        # removes the offset error from the reconstruction.
        normalized = self._normalize(data)
        self.baseline_ = normalized[0].astype(np.float64)

    def _transform(self, data: np.ndarray, fs: float, n_steps: int) -> np.ndarray:
        values = self._normalize(resample_to_steps(data, fs, self.dt, n_steps))
        n_channels = values.shape[1]
        spikes = np.zeros((values.shape[0], n_channels * 2), dtype=np.uint8)

        # Error-feedback loop: ``tracked`` is the value the decoder will rebuild
        # from the spikes emitted so far.
        tracked = self.baseline_.copy()
        budget = self.max_steps_per_sample
        for step in range(values.shape[0]):
            for _ in range(budget):
                error = values[step] - tracked
                index = error >= self.threshold
                decrement = error <= -self.threshold
                if not (index.any() or decrement.any()):
                    break
                spikes[step, 0::2] |= index.astype(np.uint8)
                spikes[step, 1::2] |= decrement.astype(np.uint8)
                tracked = tracked + (index.astype(np.float64) - decrement.astype(np.float64)) * self.threshold
        return spikes

    def decode(self, spikes):
        """Rebuild the signal: cumulative sum of net (up - down) steps.

        The learned baseline is added back, so the result is on the encoder's
        normalised [0, 1] scale — the same scale ``information_loss`` compares
        against.
        """
        up = spikes.spikes[:, 0::2].astype(np.float64)
        down = spikes.spikes[:, 1::2].astype(np.float64)
        return self.baseline_[None, :] + np.cumsum((up - down) * self.threshold, axis=0)

    def get_state(self) -> dict[str, object]:
        state = dict(super().get_state())
        state["baseline"] = None if self.baseline_ is None else self.baseline_.tolist()
        return state

    def _restore_state(self, extras: dict[str, object]) -> None:
        baseline = extras.get("baseline")
        self.baseline_ = None if baseline is None else np.asarray(baseline, dtype=np.float64)

    def _neuron_count(self, n_channels: int) -> int:
        return n_channels * 2

    def _channel_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for channel in range(self.n_channels_ or 0):
            names.append(f"delta_ch{channel:02d}_up")
            names.append(f"delta_ch{channel:02d}_down")
        return tuple(names)

    def param_space(self) -> SearchSpace:
        return SearchSpace(
            {
                # The step size is the accuracy/sparsity knob: a small step tracks
                # the signal closely but costs spikes, a large step is sparse but
                # coarse.  The budget must be large enough to follow the signal's
                # slew rate, otherwise the modulator saturates and the
                # reconstruction lags (which the information-loss term will show).
                "threshold": Parameter(
                    "threshold",
                    "float",
                    0.002,
                    0.5,
                    log=True,
                    description="Quantisation step: error that triggers a spike",
                ),
                "max_steps_per_sample": Parameter(
                    "max_steps_per_sample",
                    "int",
                    1,
                    8,
                    description="Spike budget per sample (tracks fast changes)",
                ),
                "dt_ms": Parameter("dt_ms", "choice", choices=(0.5, 1.0, 2.0, 5.0), description="Spike bin width (ms)"),
            }
        )
