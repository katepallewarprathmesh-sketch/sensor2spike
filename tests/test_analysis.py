"""Tests for spike statistics, classification metrics and the search space."""

from __future__ import annotations

import numpy as np
import pytest

from sensor2spike.analysis import (
    accuracy,
    classification_summary,
    compute_cost,
    compute_spike_stats,
    confusion_matrix,
    macro_f1,
    per_class_accuracy,
)
from sensor2spike.contracts import SpikeTrain
from sensor2spike.exceptions import ConfigurationError, DataValidationError
from sensor2spike.spaces import Parameter, SearchSpace


# ------------------------------------------------------------------------ metrics
def test_accuracy_and_f1_on_perfect_and_imperfect_predictions():
    y_true = np.array([0, 1, 1, 0])
    assert accuracy(y_true, y_true) == pytest.approx(1.0)
    assert macro_f1(y_true, y_true) == pytest.approx(1.0)

    y_pred = np.array([0, 1, 0, 1])
    assert accuracy(y_true, y_pred) == pytest.approx(0.5)
    f1 = macro_f1(y_true, y_pred)
    assert 0.0 <= f1 <= 0.5


def test_confusion_matrix_layout():
    matrix = confusion_matrix(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), n_classes=2)
    np.testing.assert_array_equal(matrix, np.array([[1, 1], [0, 2]]))


def test_per_class_accuracy_handles_absent_classes():
    values = per_class_accuracy(np.array([0, 0, 0]), np.array([0, 0, 0]), n_classes=2)
    np.testing.assert_allclose(values, np.array([1.0, 0.0]))


def test_classification_summary_contains_confusion():
    summary = classification_summary(np.array([0, 1]), np.array([0, 1]))
    assert summary["accuracy"] == pytest.approx(1.0)
    assert np.asarray(summary["confusion_matrix"]).shape == (2, 2)


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(DataValidationError):
        accuracy(np.array([0, 1]), np.array([0]))


# ------------------------------------------------------------------- spike stats
def test_spike_stats_are_consistent(spike_train):
    stats = compute_spike_stats(spike_train, information_loss=0.25, reconstruction_r2=0.75)
    assert stats.spike_count == int(spike_train.spikes.sum())
    assert stats.n_neurons == spike_train.n_neurons
    assert stats.n_steps == spike_train.n_steps
    assert stats.duration_seconds == pytest.approx(spike_train.n_steps * spike_train.dt)
    assert stats.spike_rate_hz == pytest.approx(stats.spike_count / stats.duration_seconds)
    assert 0.0 < stats.active_fraction <= 1.0
    assert 0.0 <= stats.temporal_sparsity <= 1.0
    assert stats.information_loss == pytest.approx(0.25)
    assert stats.compute_cost is not None and stats.compute_cost > 0
    dumped = stats.to_dict()
    assert dumped["spike_count"] == stats.spike_count


def test_spike_stats_pool_multiple_trains(spike_train):
    total = compute_spike_stats([spike_train, spike_train])
    assert total.spike_count == 2 * spike_train.spike_count
    assert total.n_steps == 2 * spike_train.n_steps


def test_spike_stats_rejects_mixed_neuron_counts(spike_train):
    other = SpikeTrain(spikes=np.zeros((10, 3), dtype=np.uint8), dt=0.001)
    with pytest.raises(ValueError):
        compute_spike_stats([spike_train, other])


def test_empty_train_has_no_isi(spike_train):
    silent = SpikeTrain(spikes=np.zeros_like(spike_train.spikes), dt=spike_train.dt)
    stats = compute_spike_stats(silent)
    assert stats.spike_count == 0
    assert stats.mean_isi_ms is None
    assert stats.active_fraction == 0.0


def test_compute_cost_scales_with_spikes_and_steps(spike_train):
    cost = compute_cost(spike_train)
    doubled = SpikeTrain(
        spikes=np.concatenate([spike_train.spikes, spike_train.spikes], axis=0), dt=spike_train.dt
    )
    assert compute_cost(doubled) > cost
    assert compute_cost([spike_train, spike_train]) == pytest.approx(2 * cost)


def test_spike_train_validation():
    with pytest.raises(DataValidationError):
        SpikeTrain(spikes=np.zeros((4, 2)) + 3, dt=0.001)
    with pytest.raises(DataValidationError):
        SpikeTrain(spikes=np.zeros((0, 2)), dt=0.001)
    with pytest.raises(DataValidationError):
        SpikeTrain(spikes=np.zeros((4, 2)), dt=0.001, channel_names=("only_one",))


def test_firing_rates_and_sparsity(spike_train):
    rates = spike_train.firing_rates_hz
    assert rates.shape == (spike_train.n_neurons,)
    assert np.all(rates >= 0)
    assert 0.0 <= spike_train.temporal_sparsity <= 1.0
    assert spike_train.to_float().dtype == np.float32


# --------------------------------------------------------------------- search space
def test_parameter_validation_errors():
    with pytest.raises(ConfigurationError):
        Parameter("x", "weird")
    with pytest.raises(ConfigurationError):
        Parameter("x", "float", 5.0, 1.0)
    with pytest.raises(ConfigurationError):
        Parameter("x", "choice", choices=())
    with pytest.raises(ConfigurationError):
        Parameter("x", "choice", choices=(1, 1))
    with pytest.raises(ConfigurationError):
        Parameter("x", "float", -1.0, 1.0, log=True)
    with pytest.raises(ConfigurationError):
        Parameter("x", "int", 0.0, 1.5)


def test_parameter_sampling_is_within_bounds():
    rng = np.random.default_rng(0)
    numeric = Parameter("f", "float", 0.1, 10.0, log=True)
    integer = Parameter("i", "int", 2, 8)
    choice = Parameter("c", "choice", choices=("a", "b", "c"))
    for _ in range(50):
        value = numeric.sample(rng)
        assert 0.1 <= value <= 10.0
        assert 2 <= integer.sample(rng) <= 8
        assert choice.sample(rng) in {"a", "b", "c"}


def test_parameter_grid_shapes():
    assert len(Parameter("f", "float", 0.0, 1.0).grid(5)) == 5
    assert Parameter("c", "choice", choices=("a", "b")).grid(5) == ["a", "b"]
    assert Parameter("i", "int", 0, 4).grid(10) == [0, 1, 2, 3, 4]


def test_search_space_merge_and_grid_cap():
    space = SearchSpace(
        {
            "a": Parameter("a", "float", 0.0, 1.0),
            "b": Parameter("b", "int", 1, 3),
            "c": Parameter("c", "choice", choices=(True, False)),
        }
    )
    combos = space.grid(3, max_combinations=10)
    assert len(combos) <= 10
    merged = space.merge({"a": 0.5})
    assert merged["a"].kind == "choice"
    assert merged["a"].choices == (0.5,)


def test_search_space_validate_coerces_and_rejects():
    space = SearchSpace({"a": Parameter("a", "float", 0.0, 1.0)})
    assert space.validate({"a": 0.25})["a"] == pytest.approx(0.25)
    # Out-of-range values are clipped, unknown keys rejected.
    assert space.validate({"a": 5.0})["a"] == pytest.approx(1.0)
    with pytest.raises(ConfigurationError):
        space.validate({"nope": 1})
    with pytest.raises(ConfigurationError):
        space.validate({})


def test_search_space_round_trip():
    space = SearchSpace({"a": Parameter("a", "float", 0.0, 1.0), "b": Parameter("b", "int", 1, 2)})
    payload = space.to_dict()
    rebuilt = SearchSpace.from_dict(payload)
    assert set(rebuilt.keys()) == set(space.keys())
    assert rebuilt["a"].kind == "float"
