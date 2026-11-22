"""Shared fixtures for the Sensor2Spike test suite."""

from __future__ import annotations

import numpy as np
import pytest

from sensor2spike.contracts import EvaluationResult, ExperimentResult, SpikeTrain, TrialResult
from sensor2spike.demo import generate_demo_dataset


@pytest.fixture(scope="session")
def demo_dataset():
    """Small, deterministic two-class radar-like dataset (12 trials)."""
    return generate_demo_dataset(n_trials=12, seed=11, duration_seconds=0.5)


@pytest.fixture()
def tiny_dataset():
    """Very small dataset: 6 trials x 0.25 s at 500 Hz, 2 channels."""
    rng = np.random.default_rng(3)
    fs = 500.0
    n_samples = int(0.25 * fs)
    t = np.arange(n_samples) / fs
    blocks, labels, groups = [], [], []
    for trial in range(6):
        label = trial % 2
        signal = np.zeros((n_samples, 2))
        for channel in range(2):
            signal[:, channel] = (
                (1.0 + 0.4 * label) * np.sin(2 * np.pi * (3.0 + channel) * t)
                + 0.05 * rng.standard_normal(n_samples)
            )
        blocks.append(signal.astype(np.float32))
        labels.append(np.full(n_samples, label, dtype=np.int64))
        groups.append(np.full(n_samples, trial, dtype=np.int64))
    from sensor2spike.contracts import SensorDataset

    return SensorDataset(
        data=np.concatenate(blocks),
        sampling_frequency=fs,
        channel_names=("c0", "c1"),
        labels=np.concatenate(labels),
        groups=np.concatenate(groups),
        name="tiny",
        sensor_type="synthetic",
    )


@pytest.fixture()
def spike_train() -> SpikeTrain:
    """Deterministic 50-step x 4-neuron spike train."""
    spikes = np.zeros((50, 4), dtype=np.uint8)
    spikes[::5, 0] = 1
    spikes[::7, 1] = 1
    spikes[::11, 2] = 1
    spikes[3, 3] = 1
    return SpikeTrain(spikes=spikes, dt=0.001, channel_names=("a", "b", "c", "d"), encoder="rate")


@pytest.fixture()
def trial_factory():
    """Build a ``TrialResult`` with sensible defaults."""

    def _make(index: int = 0, **kwargs) -> TrialResult:
        payload = {
            "index": index,
            "params": {"rate_max_hz": 100.0},
            "accuracy": 0.8,
            "macro_f1": 0.8,
            "spike_count": 1000,
            "information_loss": 0.5,
            "compute_cost": 10_000.0,
        }
        payload.update(kwargs)
        return TrialResult(**payload)

    return _make


@pytest.fixture()
def experiment_result() -> ExperimentResult:
    """A small, fully-populated ``ExperimentResult`` for reporting tests."""
    trials = [
        TrialResult(
            index=0,
            params={"rate_max_hz": 50.0, "dt_ms": 1.0},
            accuracy=0.70,
            macro_f1=0.69,
            spike_count=800,
            information_loss=0.9,
            compute_cost=5_000.0,
            rank=1,
            is_pareto=True,
            score=-1.5,
        ),
        TrialResult(
            index=1,
            params={"rate_max_hz": 200.0, "dt_ms": 1.0},
            accuracy=0.95,
            macro_f1=0.95,
            spike_count=4_000,
            information_loss=0.4,
            compute_cost=20_000.0,
            rank=1,
            is_pareto=True,
            score=-1.2,
        ),
        TrialResult(
            index=2,
            params={"rate_max_hz": 400.0, "dt_ms": 1.0},
            accuracy=0.50,
            macro_f1=0.48,
            spike_count=9_000,
            information_loss=0.3,
            compute_cost=40_000.0,
            rank=2,
            is_pareto=False,
            score=-2.4,
        ),
    ]
    return ExperimentResult(
        experiment_id="exp_test000",
        dataset={"name": "tiny", "n_samples": 750, "sampling_frequency": 500.0},
        encoder={"encoder": "rate", "seed": 0},
        search_space={"rate_max_hz": {"kind": "float", "low": 10.0, "high": 500.0}},
        optimizer="random",
        n_trials=3,
        seed=0,
        objective="multi",
        weights={"accuracy": 1.0, "spikes": 1.0, "information": 1.0, "compute": 1.0},
        accuracy_target=None,
        spike_budget=None,
        backend={"backend": "lif_ridge", "tau_ms": 20.0},
        preprocess={"enabled": True},
        window={"window_seconds": 0.25, "window_samples": 125, "stride_samples": 125},
        trials=trials,
        notes="unit test",
        best_index=1,
        split={"n_train_windows": 4, "n_test_windows": 2},
        software={"python": "3.13", "numpy": "2.x"},
    )


@pytest.fixture()
def evaluation_result() -> EvaluationResult:
    return EvaluationResult(
        backend="lif_ridge",
        accuracy=0.9,
        macro_f1=0.89,
        n_train=40,
        n_test=20,
        train_time_s=0.05,
        inference_time_s=0.01,
        n_parameters=123,
        extra={"confusion": [[9, 1], [1, 9]]},
    )
