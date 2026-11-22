"""Synthetic radar demo dataset: human presence detection.

The specification recommends starting with a single application, preferably
radar-based human presence detection (section 23).  This generator produces a
reproducible stand-in for such a recording so the whole pipeline can be
exercised without hardware:

* **Class 0 — empty room**: low-frequency clutter (HVAC / fan vibration) plus
  sensor noise.
* **Class 1 — presence**: the same clutter plus a gait-like micro-Doppler
  signature (fundamental 1.6–2.6 Hz with harmonics) that is amplitude modulated
  by a breathing rate of 0.25–0.45 Hz.

Four channels represent range bins with independent phase and gain.  Every trial
is a separate group so window splits stay leakage-free.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import SensorDataset

__all__ = ["generate_demo_dataset", "DEMO_CLASS_NAMES", "DEMO_SENSOR_TYPE"]

DEMO_CLASS_NAMES = ("empty", "presence")
DEMO_SENSOR_TYPE = "radar"


def generate_demo_dataset(
    *,
    n_trials: int = 120,
    seed: int = 7,
    noise: float = 0.15,
    fs: float = 1000.0,
    duration_seconds: float = 1.0,
    n_channels: int = 4,
) -> SensorDataset:
    """Generate a two-class synthetic radar recording.

    Parameters
    ----------
    n_trials:
        Total number of trials; classes alternate so the dataset is balanced.
    seed:
        Seed for the deterministic random generator.
    noise:
        Sensor-noise scale (0 disables noise).
    fs, duration_seconds, n_channels:
        Sampling rate, trial length and number of range bins.
    """
    if n_trials < 2:
        raise ValueError("n_trials must be >= 2.")
    if n_channels < 1:
        raise ValueError("n_channels must be >= 1.")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")

    rng = np.random.default_rng(seed)
    n_samples = int(round(fs * duration_seconds))
    t = np.arange(n_samples, dtype=np.float64) / fs

    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []

    for trial in range(n_trials):
        label = trial % 2
        clutter_phase = rng.uniform(0.0, 2.0 * np.pi, size=2)
        clutter = (
            0.22 * np.sin(2.0 * np.pi * 0.15 * t + clutter_phase[0])
            + 0.12 * np.sin(2.0 * np.pi * 0.42 * t + clutter_phase[1])
        )

        signal = np.zeros((n_samples, n_channels), dtype=np.float64)
        for channel in range(n_channels):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            if label == 1:
                gait = rng.uniform(1.6, 2.6)
                breathing = rng.uniform(0.25, 0.45)
                gain = rng.uniform(0.65, 1.0)
                micro_doppler = gain * (
                    np.sin(2.0 * np.pi * gait * t + phase)
                    + 0.5 * np.sin(2.0 * np.pi * 2.0 * gait * t + 1.7 * phase + 0.4 * channel)
                    + 0.28 * np.sin(2.0 * np.pi * 3.0 * gait * t + 0.6 * phase + 0.9 * channel)
                )
                envelope = 1.0 + 0.30 * np.sin(2.0 * np.pi * breathing * t + phase * 0.3)
                signal[:, channel] = clutter + micro_doppler * envelope
            else:
                signal[:, channel] = clutter + 0.05 * np.sin(
                    2.0 * np.pi * 0.08 * t + 0.7 * phase + 0.2 * channel
                )
            signal[:, channel] += noise * rng.standard_normal(n_samples)

        # Normalise each trial so amplitude jitter does not make the task trivial.
        peak = float(np.max(np.abs(signal)))
        if peak > 0:
            signal = signal / peak

        blocks.append(signal.astype(np.float32))
        labels.append(np.full(n_samples, label, dtype=np.int64))
        groups.append(np.full(n_samples, trial, dtype=np.int64))

    return SensorDataset(
        data=np.concatenate(blocks, axis=0),
        sampling_frequency=fs,
        channel_names=tuple(f"range_bin_{index}" for index in range(n_channels)),
        labels=np.concatenate(labels),
        groups=np.concatenate(groups),
        name="demo_radar_presence",
        sensor_type=DEMO_SENSOR_TYPE,
        metadata={
            "synthetic": True,
            "description": (
                "Synthetic radar micro-Doppler recording for human presence detection. "
                "Class 0: empty room clutter. Class 1: gait + breathing modulation."
            ),
            "class_names": list(DEMO_CLASS_NAMES),
            "n_trials": int(n_trials),
            "noise": float(noise),
            "generator_seed": int(seed),
        },
        version="1",
    )


def demo_summary(dataset: SensorDataset) -> dict[str, Any]:
    """Return a compact description used by the CLI and the benchmark report."""
    counts = {int(label): int(np.sum(dataset.labels == label)) for label in dataset.classes}
    return {
        "name": dataset.name,
        "sensor_type": dataset.sensor_type,
        "n_samples": dataset.n_samples,
        "n_channels": dataset.n_channels,
        "n_trials": dataset.n_groups,
        "sampling_frequency": dataset.sampling_frequency,
        "duration_seconds_per_trial": dataset.duration_seconds / max(1, dataset.n_groups),
        "class_counts": counts,
        "class_names": list(DEMO_CLASS_NAMES),
        "fingerprint": dataset.fingerprint(),
    }
