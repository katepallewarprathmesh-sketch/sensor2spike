"""Tests for the pluggable SNN evaluation backends."""

from __future__ import annotations

import numpy as np
import pytest

from sensor2spike._core.contracts import EvaluationResult, SpikeTrain
from sensor2spike._core.encoders import build_encoder
from sensor2spike._core.exceptions import (
    ConfigurationError,
    DependencyMissingError,
    EvaluationError,
    Sensor2SpikeError,
)
from sensor2spike._core.snn import (
    BACKENDS,
    SNNBackend,
    build_backend,
    list_backends,
    register_backend,
)


def _spike_dataset(n_per_class: int = 6, n_steps: int = 40, n_neurons: int = 8, seed: int = 0):
    """Two linearly separable spike classes."""
    rng = np.random.default_rng(seed)
    trains, labels = [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            spikes = (rng.random((n_steps, n_neurons)) < (0.1 if label == 0 else 0.5)).astype(np.uint8)
            trains.append(SpikeTrain(spikes=spikes, dt=0.001, encoder="test"))
            labels.append(label)
    return trains, np.array(labels)


# ------------------------------------------------------------------------- registry
def test_registry_contains_reference_backend():
    assert "lif_ridge" in BACKENDS
    names = {entry["name"] for entry in list_backends()}
    assert {"lif_ridge", "torch_lif"} <= names


def test_build_backend_by_name_and_config():
    backend = build_backend("lif_ridge", tau_ms=10.0)
    assert isinstance(backend, SNNBackend)
    assert backend.tau_ms == pytest.approx(10.0)
    assert backend.get_config()["backend"] == "lif_ridge"
    from_mapping = build_backend({"backend": "lif_ridge", "alpha": 0.5})
    assert from_mapping.alpha == pytest.approx(0.5)
    with pytest.raises(ConfigurationError):
        build_backend("nope")
    with pytest.raises(ConfigurationError):
        build_backend("lif_ridge", bogus_parameter=1)


def test_register_backend_rejects_duplicates_and_incomplete_classes():
    class MissingHooks(SNNBackend):
        name = "incomplete"

    with pytest.raises(ConfigurationError):
        register_backend(MissingHooks)

    class Good(SNNBackend):
        name = "good_test_backend"

        def _features(self, spikes):
            return spikes.spikes.sum(axis=0, keepdims=True).astype(np.float64)

        def _fit(self, features, labels):
            self.centroids_ = np.stack(
                [features[labels == label].mean(axis=0) for label in np.unique(labels)]
            )

        def _predict_scores(self, features):
            distances = ((features[:, None, :] - self.centroids_[None]) ** 2).sum(axis=2)
            return -distances

        @property
        def n_parameters(self):
            return int(self.centroids_.size)

    registered = register_backend(Good)
    assert registered is Good
    assert "good_test_backend" in BACKENDS
    with pytest.raises(ConfigurationError):
        register_backend(Good)


# --------------------------------------------------------------------- reference LIF
def test_lif_backend_trains_and_evaluates():
    trains, labels = _spike_dataset()
    train_idx, test_idx = np.arange(0, len(trains), 2), np.arange(1, len(trains), 2)
    backend = build_backend("lif_ridge", seed=1)
    result = backend.train_evaluate(
        [trains[i] for i in train_idx],
        labels[train_idx],
        [trains[i] for i in test_idx],
        labels[test_idx],
        seed=1,
    )
    assert isinstance(result, EvaluationResult)
    assert result.backend == "lif_ridge"
    assert 0.0 <= result.accuracy <= 1.0
    assert result.n_train == len(train_idx)
    assert result.n_test == len(test_idx)
    assert result.train_time_s >= 0.0
    assert result.n_parameters > 0


def test_lif_backend_learns_separable_classes():
    trains, labels = _spike_dataset(n_per_class=8)
    backend = build_backend("lif_ridge", alpha=0.1)
    result = backend.train_evaluate(trains, labels, trains, labels, seed=0)
    assert result.accuracy >= 0.9


def test_lif_backend_is_deterministic():
    trains, labels = _spike_dataset()
    first = build_backend("lif_ridge", seed=7).train_evaluate(trains, labels, trains, labels, seed=7)
    second = build_backend("lif_ridge", seed=7).train_evaluate(trains, labels, trains, labels, seed=7)
    assert first.accuracy == second.accuracy
    np.testing.assert_array_equal(first.predictions, second.predictions)


def test_lif_backend_requires_fit_before_predict():
    backend = build_backend("lif_ridge")
    trains, _ = _spike_dataset(n_per_class=1)
    with pytest.raises(EvaluationError):
        backend.predict(trains)


def test_lif_backend_rejects_mismatched_labels():
    trains, labels = _spike_dataset()
    backend = build_backend("lif_ridge")
    with pytest.raises(EvaluationError):
        backend.fit(trains, labels[:-1])


def test_lif_backend_rejects_single_class():
    trains, _ = _spike_dataset()
    backend = build_backend("lif_ridge")
    with pytest.raises(EvaluationError):
        backend.fit(trains, np.zeros(len(trains), dtype=int))


def test_lif_backend_supports_feature_sets():
    trains, labels = _spike_dataset()
    for feature_set in ("mean_max", "membrane", "mean_max_rate"):
        backend = build_backend("lif_ridge", feature_set=feature_set)
        result = backend.train_evaluate(trains, labels, trains, labels)
        assert result.accuracy >= 0.5
    with pytest.raises(ConfigurationError):
        build_backend("lif_ridge", feature_set="magic")


def test_lif_backend_temporal_bins_change_feature_dimension():
    trains, _ = _spike_dataset(n_neurons=4, n_steps=50)
    few = build_backend("lif_ridge", temporal_bins=2)
    many = build_backend("lif_ridge", temporal_bins=10)
    assert many.features(trains[:1]).shape[1] > few.features(trains[:1]).shape[1]


# ------------------------------------------------------------------- end-to-end path
def test_encoder_backend_pipeline(tiny_dataset):
    """Full path: window -> encode -> train -> evaluate, leakage-free split."""
    from sensor2spike._core.data import split_windows

    windows = tiny_dataset.to_windows(window_samples=64, stride=64)
    train_windows, test_windows = split_windows(windows, test_fraction=0.34, seed=0)
    assert set(train_windows.labels) == {0, 1}
    assert set(test_windows.labels) == {0, 1}

    encoder = build_encoder("rate", rate_max_hz=200.0, seed=0).fit(
        tiny_dataset.data, fs=tiny_dataset.sampling_frequency
    )
    backend = build_backend("lif_ridge")
    result = backend.train_evaluate(
        encoder.transform_windows(train_windows),
        train_windows.labels,
        encoder.transform_windows(test_windows),
        test_windows.labels,
    )
    assert 0.0 <= result.accuracy <= 1.0
    assert result.n_test == test_windows.n_windows
    assert result.n_train == train_windows.n_windows


# ------------------------------------------------------------------------- torch hook
def test_torch_backend_reports_missing_dependency_with_install_hint():
    try:
        build_backend("torch_lif").train_evaluate(*([], [], [], []))
    except DependencyMissingError as exc:
        assert "torch" in str(exc).lower()
    except Sensor2SpikeError as exc:  # torch installed: backend should still work
        assert not isinstance(exc, DependencyMissingError)
