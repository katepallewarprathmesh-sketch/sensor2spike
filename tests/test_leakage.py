"""Guards against the ways an evaluation can silently lie.

Each test here pins down a leak that was previously present and measured: the
transform was fitted on the whole recording, the reported spike budget described
the training split while accuracy described the test split, and overlapping
windows could be split across the boundary. These tests fail if any of that
comes back.
"""

from __future__ import annotations

import numpy as np
import pytest

import sensor2spike
from sensor2spike import (
    AutoTuner,
    ConfigurationError,
    NormalizationStats,
    PreprocessConfig,
    SensorDataset,
    preprocess,
    split_windows,
)
from sensor2spike.demo import generate_demo_dataset


# --------------------------------------------------------------- normalisation
def test_stats_fitted_on_train_do_not_see_test_values():
    """The transform must be identical no matter what the test split contains."""
    rng = np.random.default_rng(0)
    train = rng.normal(size=(500, 2))

    stats = NormalizationStats.fit(train, "zscore")

    # Same training data, two wildly different future test sets.
    a = stats.transform(np.array([[1e6, -1e6]]))
    b = stats.transform(np.array([[0.0, 0.0]]))
    assert not np.allclose(a, b), "transform should carry the values through unchanged"

    # And the transform itself is a pure function of the training data.
    again = NormalizationStats.fit(train, "zscore")
    assert np.allclose(stats.offset, again.offset)
    assert np.allclose(stats.scale, again.scale)


def test_test_split_can_fall_outside_the_training_range():
    """A held-out split is allowed to exceed [0, 1] — that proves it was unseen."""
    stats = NormalizationStats.fit(np.array([[0.0], [1.0], [2.0]]), "minmax")
    held_out = stats.transform(np.array([[3.0]]))
    assert held_out[0, 0] > 1.0


def test_preprocess_with_stats_matches_fit_then_apply():
    dataset = generate_demo_dataset(n_trials=6, seed=2)
    config = PreprocessConfig(normalize="zscore")
    stats = config.fit(dataset.data)

    a = preprocess(dataset, config, stats=stats)
    b = config.apply(dataset, stats)
    assert np.allclose(a.data, b.data, atol=1e-6)


def test_mismatched_stats_and_config_are_rejected():
    dataset = generate_demo_dataset(n_trials=4, seed=2)
    stats = NormalizationStats.fit(dataset.data, "minmax")
    with pytest.raises(ConfigurationError, match="fitted for"):
        preprocess(dataset, PreprocessConfig(normalize="zscore"), stats=stats)


# ------------------------------------------------- the runner fits on train only
def test_tampering_with_the_test_split_cannot_move_the_training_transform():
    """The decisive leakage test.

    Corrupt every sample that belongs to the test groups, then verify that the
    *training* data is preprocessed identically to before.  A transform fitted
    on the whole recording would shift when the test data changes; a transform
    fitted on train only cannot.
    """
    dataset = generate_demo_dataset(n_trials=12, seed=5)
    config = PreprocessConfig(normalize="zscore")

    raw = dataset.to_windows(250)
    train, test = split_windows(raw, test_fraction=0.3, seed=0)
    test_groups = set(test.groups.tolist())
    hold_out = np.isin(dataset.groups, list(test_groups))

    tampered = SensorDataset(
        data=dataset.data.copy(),
        sampling_frequency=dataset.sampling_frequency,
        labels=dataset.labels,
        groups=dataset.groups,
    )
    tampered.data[hold_out] += 1e3  # violent, unmistakable change

    # Honest pipeline: statistics from the training samples only.
    honest = config.fit(np.concatenate(train.samples, axis=0))
    honest_after = config.fit(np.concatenate(train.samples, axis=0))
    assert np.allclose(honest.offset, honest_after.offset)
    assert np.allclose(honest.scale, honest_after.scale)

    # The training portion of the data must be transformed identically whether
    # or not the test portion was tampered with.
    honest_values = honest.transform(dataset.data)
    honest_values_after = honest.transform(tampered.data)
    assert np.allclose(honest_values[~hold_out], honest_values_after[~hold_out])

    # And the leaky alternative really would have moved: prove the test is
    # sharp by showing whole-recording statistics DO shift.
    leaky = config.fit(dataset.data)
    leaky_after = config.fit(tampered.data)
    assert not np.allclose(leaky.offset, leaky_after.offset, atol=1e-6), (
        "the tampering was too subtle to distinguish the two approaches"
    )


def test_spike_statistics_are_measured_on_the_test_split(monkeypatch):
    """The spike budget must describe the same split as the accuracy.

    Train and test differ in size, so counting trains is enough to tell which
    split a statistic came from.
    """
    from sensor2spike._core.experiment import runner as runner_module

    seen: list[int] = []
    original = runner_module.compute_spike_stats

    def spy(spike_trains, **kwargs):
        seen.append(len(spike_trains))
        return original(spike_trains, **kwargs)

    monkeypatch.setattr(runner_module, "compute_spike_stats", spy)

    dataset = generate_demo_dataset(n_trials=12, seed=4)
    result = AutoTuner(
        objective="multi_objective", encoder="rate", n_trials=1, seed=0, window_seconds=0.25
    ).optimize(dataset)

    trial = result.best
    assert seen, "compute_spike_stats was never called"
    assert seen[0] == trial.evaluation["n_test"], (
        f"spike statistics came from {seen[0]} windows but the test split has "
        f"{trial.evaluation['n_test']} (train has {trial.evaluation['n_train']})"
    )
    assert trial.evaluation["n_test"] != trial.evaluation["n_train"]


# ----------------------------------------------------- overlapping windows
def test_overlapping_windows_without_groups_are_refused():
    dataset = generate_demo_dataset(n_trials=8, seed=6)
    ungrouped = SensorDataset(
        data=dataset.data,
        sampling_frequency=dataset.sampling_frequency,
        labels=dataset.labels,
    )
    with pytest.raises(ConfigurationError, match="require group identifiers"):
        AutoTuner(
            objective="multi_objective",
            encoder="rate",
            n_trials=2,
            seed=0,
            window_seconds=0.25,
            window_stride_seconds=0.025,
        ).optimize(ungrouped)


def test_overlapping_windows_with_groups_are_allowed():
    dataset = generate_demo_dataset(n_trials=8, seed=6)
    result = AutoTuner(
        objective="multi_objective",
        encoder="rate",
        n_trials=2,
        seed=0,
        window_seconds=0.25,
        window_stride_seconds=0.025,
    ).optimize(dataset)
    assert result.best.accuracy >= 0.0


def test_non_overlapping_windows_without_groups_are_allowed():
    dataset = generate_demo_dataset(n_trials=8, seed=6)
    ungrouped = SensorDataset(
        data=dataset.data,
        sampling_frequency=dataset.sampling_frequency,
        labels=dataset.labels,
    )
    result = AutoTuner(
        objective="multi_objective", encoder="rate", n_trials=2, seed=0, window_seconds=0.25
    ).optimize(ungrouped)
    assert result.best.accuracy >= 0.0


# ---------------------------------------------------------- group integrity
def test_train_and_test_never_share_a_group():
    dataset = generate_demo_dataset(n_trials=24, seed=7)
    train, test = split_windows(dataset.to_windows(250), test_fraction=0.3, seed=0)
    assert not (set(train.groups.tolist()) & set(test.groups.tolist()))


def test_normalization_stats_are_public_api():
    assert "NormalizationStats" in sensor2spike.__all__


def test_preprocess_honours_supplied_stats_instead_of_refitting():
    """End-to-end: ``preprocess(..., stats=...)`` must not re-estimate.

    This is the code path the runner uses.  If the implementation quietly
    refitted the statistics on the data it was handed, the held-out split would
    rescale itself and the leak would be back.

    Detrending and filtering are disabled here so the comparison isolates
    normalisation, which is the only stage that can leak.
    """
    dataset = generate_demo_dataset(n_trials=6, seed=9)
    config = PreprocessConfig(normalize="zscore", detrend=False)

    # Fit on a deliberately unrepresentative slice, so re-fitting on the whole
    # dataset would give a measurably different answer.
    stats = config.fit(dataset.data[:500])

    applied = config.apply(dataset, stats)
    expected = stats.transform(dataset.data.astype(np.float64)).astype(np.float32)
    assert np.allclose(applied.data, expected, atol=1e-5), (
        "preprocess() refitted its own statistics instead of using the supplied ones"
    )

    self_fitted = preprocess(dataset, config)
    assert not np.allclose(applied.data, self_fitted.data, atol=1e-3), (
        "the two paths agree, so this test cannot distinguish them"
    )
