#!/usr/bin/env python3
"""Extend Sensor2Spike with your own encoder — and optimise it like any other.

The product claims "add new encoding strategies without changing the core".
This example holds that claim to account: a Ben's-spike (moving-threshold)
encoder is defined in the file, registered through the public API, and then
searched by ``AutoTuner`` without editing a single line of the engine.

Ben's-spike coding: an integrator accumulates a moving baseline plus a
threshold term. A spike is emitted and the accumulator reset once the running
value crosses the threshold. It adapts to slowly varying signals in a way fixed
thresholds cannot, which is exactly the sort of encoder a practitioner would
want to test against the built-ins.

Run with::

    python examples/custom_encoder.py
"""

from __future__ import annotations

import numpy as np

from sensor2spike import (
    AutoTuner,
    SensorDataset,
    SpikeEncoder,
    build_encoder,
    list_encoders,
    register_encoder,
)
from sensor2spike.demo import generate_demo_dataset


@register_encoder
class BensSpikeEncoder(SpikeEncoder):
    """Moving-threshold integrator coding (Ben's spike algorithm).

    A leaky running average tracks the signal's baseline. The difference between
    the signal and that baseline drives an accumulator; when the accumulator
    crosses ``threshold`` a spike fires and the accumulator resets. The leak
    ``adaptation_ms`` controls how quickly the threshold follows a slow trend.
    """

    name = "bens_spike"
    description = "Ben's moving-threshold integrator (adaptive threshold)"
    PARAM_NAMES = ("threshold", "adaptation_ms")

    def __init__(
        self,
        *,
        dt_ms: float = 1.0,
        seed: int | None = None,
        threshold: float = 0.1,
        adaptation_ms: float = 50.0,
    ) -> None:
        super().__init__(dt_ms=dt_ms, seed=seed)
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if adaptation_ms <= 0:
            raise ValueError("adaptation_ms must be positive")
        self.threshold = float(threshold)
        self.adaptation_ms = float(adaptation_ms)

    # -- encoder contract ---------------------------------------------------------
    def _fit(self, data: np.ndarray) -> None:
        self.n_neurons_ = int(data.shape[1])

    def _transform(self, data: np.ndarray, fs: float, n_steps: int) -> np.ndarray:
        # ``_steps`` resamples onto this encoder's time base (one row per step)
        # and ``_normalize`` maps values onto [0, 1] using the range seen in fit.
        values = self._normalize(self._steps(data, fs))
        n_channels = values.shape[1]
        spikes = np.zeros((n_steps, n_channels), dtype=np.uint8)

        baseline = values[0].copy()
        accumulator = np.zeros(n_channels, dtype=np.float64)
        tau = self.adaptation_ms / 1000.0

        for step in range(n_steps):
            baseline = baseline + (values[step] - baseline) * min(1.0, self.dt / tau)
            accumulator += values[step] - baseline
            fired = np.abs(accumulator) >= self.threshold
            if fired.any():
                spikes[step] = fired.astype(np.uint8)
                accumulator[fired] = 0.0
        return spikes

    def decode(self, spikes):
        """Rate-based estimate: a first-order leaky integral of the spikes."""
        alpha = min(1.0, spikes.dt / 0.02)
        smoothed = np.zeros_like(spikes.spikes, dtype=np.float64)
        state = np.zeros(spikes.n_neurons, dtype=np.float64)
        for step, row in enumerate(spikes.spikes):
            state = state + (row - state) * alpha
            smoothed[step] = state
        peak = float(np.abs(smoothed).max()) if smoothed.size else 0.0
        return smoothed if peak <= 1e-12 else np.clip(smoothed / peak, 0.0, 1.0)

    # -- search space -------------------------------------------------------------
    def param_space(self):
        space = super().param_space()
        return space.merge(
            {
                "threshold": {"kind": "float", "low": 0.01, "high": 0.5, "log": True},
                "adaptation_ms": {"kind": "float", "low": 5.0, "high": 200.0, "log": True},
            }
        )


def main() -> int:
    print("Registry before:", [entry["name"] for entry in list_encoders()])

    dataset: SensorDataset = generate_demo_dataset(n_trials=24, seed=3, duration_seconds=0.5)
    print(f"Dataset: {dataset.n_samples:,} samples, {dataset.n_groups} trials")

    # 1. Use it directly, like any built-in encoder.
    encoder = build_encoder("bens_spike", dt_ms=1.0, threshold=0.08, adaptation_ms=40.0)
    encoder.fit(dataset.data, fs=dataset.sampling_frequency)
    spikes = encoder.transform(dataset.data, fs=dataset.sampling_frequency)
    print(
        f"Direct use: {spikes.spike_count:,} spikes, "
        f"{spikes.spike_rate_hz:,.1f}/s, info loss "
        f"{encoder.information_loss(spikes, dataset.data):.3f}"
    )

    # 2. Optimise it exactly like a built-in.
    result = AutoTuner(
        objective="multi_objective",
        encoder="bens_spike",
        n_trials=16,
        seed=0,
        window_seconds=0.25,
    ).optimize(dataset)

    print()
    print(f"Optimised: {result.experiment_id} ({result.status})")
    best = result.best
    print(f"  best params : {best.params}")
    print(f"  accuracy    : {best.accuracy:.4f}")
    print(f"  spikes      : {best.spike_count}")
    print(f"  info loss   : {best.information_loss:.4f}")
    print(f"  pareto front: {len(result.pareto_front)} of {len(result.trials)} trials")
    print()
    print("Registry after :", [entry["name"] for entry in list_encoders()])
    print("No engine code was modified to make this work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
