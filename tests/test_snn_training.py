"""Tests for surrogate-gradient SNN training.

A hard threshold has zero derivative almost everywhere, so the surrogate is a
deliberate fiction and finite differences can never validate it: they return
exactly 0.0.  What *can* be verified is that the surrogate is the derivative of
the smooth proxy it claims to be, that it peaks at the threshold, and — the test
that actually matters — that gradient descent reduces the loss and the network
learns a separable task.

These tests are the reason the NumPy SGD backend is trustworthy: without them the
training loop would be an unverified claim, exactly like the PyTorch backend was
before torch was available.
"""

from __future__ import annotations

import numpy as np
import pytest

import sensor2spike
from sensor2spike import SpikeTrain, build_backend
from sensor2spike._core.snn.numpy_sgd import _surrogate
from sensor2spike.demo import generate_demo_dataset


def _toy_spikes(seed: int = 0, n: int = 60, steps: int = 50, channels: int = 4):
    """A trivially separable task: class 0 fires early, class 1 fires late."""
    rng = np.random.default_rng(seed)
    trains, labels = [], []
    for i in range(n):
        label = i % 2
        matrix = (rng.random((steps, channels)) < 0.05).astype(np.uint8)
        if label == 0:
            matrix[:10, 0] = 1
        else:
            matrix[-10:, channels - 1] = 1
        trains.append(
            SpikeTrain(
                spikes=matrix,
                dt=0.001,
                channel_names=tuple(f"ch{i}" for i in range(channels)),
            )
        )
        labels.append(label)
    return trains, np.array(labels)


# ------------------------------------------------------------- the surrogate
def test_surrogate_is_the_derivative_of_its_smooth_proxy():
    """sigma(x) = 1/(1+exp(-k x));  d sigma / d x = k s (1 - s)."""
    k = 4.0
    x = np.linspace(-3.0, 3.0, 4001)
    proxy = 1.0 / (1.0 + np.exp(-k * x))
    numeric = np.gradient(proxy, x)
    analytic = _surrogate(x, k)
    # Interior points only: np.gradient is one-sided at the edges.
    assert np.max(np.abs(numeric[10:-10] - analytic[10:-10])) < 5e-3


def test_surrogate_peaks_at_the_threshold_and_decays():
    values = {x: float(_surrogate(np.array([x]), 4.0)[0]) for x in (-2, -1, 0, 1, 2)}
    assert values[0] == max(values.values())
    assert values[-2] < 0.01 and values[2] < 0.01
    assert values[0] > 0.9


def test_surrogate_is_non_negative():
    x = np.linspace(-20, 20, 1000)
    assert (_surrogate(x, 4.0) >= 0).all()


# ------------------------------------------- finite differences are NOT the test
def test_hard_threshold_has_no_numeric_gradient():
    """Documents *why* finite differences cannot check a spiking network.

    The forward pass emits ``membrane > 0``.  Perturbing a weight by a tiny
    epsilon does not move a neuron across the threshold, so the numeric
    gradient is exactly zero while the surrogate reports a real slope.  This
    test exists so nobody "fixes" the surrogate by trusting finite differences.
    """
    backend = build_backend("numpy_sgd", hidden=4, seed=0)
    backend._build(1, 2, np.random.default_rng(0))
    backend._params["b_h"][:] = 1.0  # fire unconditionally, far from threshold

    features = np.ones((1, 5, 1))
    labels = np.array([0])
    base = backend._loss(features, labels)

    eps = 1e-7
    original = backend._params["b_h"].copy()
    backend._params["b_h"] = original + eps
    up = backend._loss(features, labels)
    backend._params["b_h"] = original - eps
    down = backend._loss(features, labels)
    backend._params["b_h"] = original
    assert up == base == down, "a saturated neuron should be completely insensitive"


# ----------------------------------------------------- gradient descent works
def test_backprop_reduces_loss_on_a_separable_task():
    """The real verification: if the gradients pointed the wrong way, loss rises."""
    trains, labels = _toy_spikes(n=40)
    backend = build_backend("numpy_sgd", hidden=16, lr=0.1, sharpness=4.0, seed=0)
    backend.n_classes_ = 2
    features = backend.features(trains)
    backend._build(features.shape[2], 2, np.random.default_rng(0))

    before = backend._loss(features, labels)
    for _ in range(30):
        logits, cache = backend._forward(features, cache=True)
        grads = backend._backward(cache, logits, labels)
        for key in backend._params:
            backend._params[key] = backend._params[key] - 0.1 * grads[key]
    after = backend._loss(features, labels)

    assert after < before, f"loss rose from {before:.4f} to {after:.4f}"


def test_network_learns_a_separable_task():
    """End to end through the public backend API."""
    trains, labels = _toy_spikes(n=60)
    backend = build_backend("numpy_sgd", hidden=16, epochs=60, lr=0.05, seed=0)
    result = backend.train_evaluate(trains, labels, trains, labels, seed=0)
    assert result.accuracy > 0.85, f"only reached {result.accuracy:.3f} on a separable task"


def test_more_epochs_does_not_make_it_worse():
    trains, labels = _toy_spikes(n=60)
    scores = []
    for epochs in (5, 40):
        backend = build_backend("numpy_sgd", hidden=16, epochs=epochs, lr=0.05, seed=0)
        scores.append(backend.train_evaluate(trains, labels, trains, labels, seed=0).accuracy)
    assert scores[1] >= scores[0] - 0.15, f"training regressed: {scores}"


# ----------------------------------------------------------- dynamics sanity
def test_neurons_actually_fire():
    """A silent network cannot be trained; this catches the dead-recurrence bug."""
    trains, labels = _toy_spikes(n=20)
    backend = build_backend("numpy_sgd", hidden=16, seed=0)
    backend.n_classes_ = 2
    features = backend.features(trains)
    backend._build(features.shape[2], 2, np.random.default_rng(0))
    _, cache = backend._forward(features, cache=True)
    firing = cache[1].mean()
    assert 0.02 < firing < 0.98, f"firing rate {firing:.3f} is degenerate"


def test_decay_follows_the_timestep():
    """A 5 ms frame must leak faster than a 1 ms frame for tau to stay fixed."""
    coarse = build_backend("numpy_sgd", tau_ms=20.0)
    coarse._dt_seconds = 0.001
    fine_decay = coarse._decay()
    coarse._dt_seconds = 0.005
    coarse_decay = coarse._decay()
    assert coarse_decay < fine_decay
    # exp(-dt/tau): 5 ms against a 20 ms constant leaves ~0.78 of the membrane.
    assert coarse_decay == pytest.approx(np.exp(-5.0 / 20.0), rel=1e-6)


def test_mixed_timesteps_are_rejected():
    trains, _ = _toy_spikes(n=4)
    trains[1] = SpikeTrain(
        spikes=trains[1].spikes, dt=0.005, channel_names=trains[1].channel_names
    )
    backend = build_backend("numpy_sgd")
    with pytest.raises(Exception, match="one timestep"):
        backend.features(trains)


# ------------------------------------------------------------- reproducibility
def test_training_is_reproducible():
    trains, labels = _toy_spikes(n=40)
    first = build_backend("numpy_sgd", hidden=12, epochs=15, seed=3).train_evaluate(
        trains, labels, trains, labels, seed=3
    )
    second = build_backend("numpy_sgd", hidden=12, epochs=15, seed=3).train_evaluate(
        trains, labels, trains, labels, seed=3
    )
    assert first.accuracy == second.accuracy
    assert first.n_parameters == second.n_parameters


def test_backend_is_public_api():
    assert "NumpySGDBackend" in sensor2spike.__all__
    assert "numpy_sgd" in {entry["name"] for entry in sensor2spike.list_backends()}


# ------------------------------------------- the finding the tool exists to show
def test_temporal_encoders_outperform_rate_coding_on_spikes():
    """Rate coding discards timing; a spiking network can use it.

    This is a measured property of the stack, not an assertion of superiority:
    on the bundled dataset, delta modulation and latency coding reach markedly
    higher accuracy through this backend than rate coding does, because rate
    coding throws away the temporal structure the recurrence is built to exploit.
    """
    dataset = generate_demo_dataset(n_trials=24, seed=7)
    windows = dataset.to_windows(250)
    labels = windows.labels[:24]

    scores = {}
    for name in ("rate", "delta", "latency"):
        encoder = sensor2spike.build_encoder(name, dt_ms=5.0)
        encoder.fit(windows.samples[0], fs=dataset.sampling_frequency)
        trains = [
            encoder.transform(sample, fs=dataset.sampling_frequency)
            for sample in windows.samples[:24]
        ]
        backend = build_backend("numpy_sgd", hidden=32, epochs=60, lr=0.02, seed=0)
        scores[name] = backend.train_evaluate(trains, labels, trains, labels, seed=0).accuracy

    assert scores["delta"] > scores["rate"], scores
    assert scores["latency"] > scores["rate"], scores
