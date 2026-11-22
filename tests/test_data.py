"""Tests for dataset ingestion, preprocessing and splitting."""

from __future__ import annotations

import numpy as np
import pytest

from sensor2spike._core.contracts import MAX_SAMPLES, SensorDataset, WindowSet
from sensor2spike._core.data import (
    PreprocessConfig,
    load_csv,
    load_dataset,
    load_npz,
    preprocess,
    save_csv,
    save_dataset,
    save_npz,
    split_windows,
)
from sensor2spike._core.exceptions import (
    ConfigurationError,
    DataValidationError,
    Sensor2SpikeError,
)


# --------------------------------------------------------------------------- contract
def test_dataset_rejects_non_finite_and_1d_expands():
    with pytest.raises(DataValidationError):
        SensorDataset(data=np.array([[0.0, np.nan], [1.0, 2.0]]), sampling_frequency=10.0)
    single = SensorDataset(data=np.arange(10.0), sampling_frequency=10.0)
    assert single.n_channels == 1


def test_dataset_rejects_bad_sampling_frequency():
    with pytest.raises(Sensor2SpikeError):
        SensorDataset(data=np.zeros((10, 1)), sampling_frequency=0.0)
    with pytest.raises(Sensor2SpikeError):
        SensorDataset(data=np.zeros((10, 1)), sampling_frequency=float("nan"))


def test_dataset_rejects_mismatched_labels_and_groups():
    data = np.zeros((10, 2))
    with pytest.raises(DataValidationError):
        SensorDataset(data=data, sampling_frequency=10.0, labels=np.zeros(5, dtype=int))
    with pytest.raises(DataValidationError):
        SensorDataset(data=data, sampling_frequency=10.0, groups=np.zeros(4, dtype=int))


def test_dataset_enforces_size_guardrail(monkeypatch):
    # Guardrail: refuse recordings above MAX_SAMPLES.  Patch the limit instead of
    # allocating 50 M samples.
    import sensor2spike._core.contracts as contracts

    monkeypatch.setattr(contracts, "MAX_SAMPLES", 5)
    with pytest.raises(DataValidationError):
        contracts.SensorDataset(data=np.zeros((10, 1)), sampling_frequency=10.0)
    assert MAX_SAMPLES > 0


def test_timestamps_must_be_increasing():
    with pytest.raises(DataValidationError):
        SensorDataset(
            data=np.zeros((5, 1)),
            sampling_frequency=10.0,
            timestamps=np.array([0.0, 0.2, 0.1, 0.3, 0.4]),
        )


def test_fingerprint_is_stable_and_content_sensitive():
    a = SensorDataset(data=np.zeros((10, 1)), sampling_frequency=10.0, name="a")
    b = SensorDataset(data=np.zeros((10, 1)), sampling_frequency=10.0, name="a")
    c = SensorDataset(data=np.ones((10, 1)), sampling_frequency=10.0, name="a")
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_summary_and_describe_are_json_safe():
    dataset = SensorDataset(
        data=np.zeros((10, 2)), sampling_frequency=10.0, labels=np.zeros(10, dtype=int), name="s"
    )
    summary = dataset.summary()
    assert summary["name"] == "s"
    assert summary["n_samples"] == 10
    assert "s" in dataset.describe()


# ---------------------------------------------------------------------------- windows
def test_to_windows_never_crosses_groups():
    data = np.arange(200, dtype=float)[:, None]
    groups = np.repeat(np.arange(4), 50)
    dataset = SensorDataset(data=data, sampling_frequency=10.0, groups=groups)
    windows = dataset.to_windows(window_samples=20)
    assert windows.n_windows == 8  # 2 windows per group, none straddling a boundary
    for window, start in zip(windows.samples, windows.starts, strict=False):
        owning_group = groups[int(start)]
        assert np.all(np.equal(window[:, 0], data[start : start + 20, 0]))
        assert groups[int(start) : int(start) + 20].tolist() == [int(owning_group)] * 20


def test_to_windows_validates_window_size():
    dataset = SensorDataset(data=np.zeros((20, 1)), sampling_frequency=10.0)
    with pytest.raises(DataValidationError):
        dataset.to_windows(window_samples=50)
    with pytest.raises(ConfigurationError):
        dataset.to_windows(window_samples=10, stride=0)


def test_split_windows_is_deterministic_and_disjoint():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((40, 20, 2)).astype(np.float32)
    windows = WindowSet(
        samples=samples,
        starts=np.arange(40),
        sampling_frequency=100.0,
        window_samples=20,
        stride=20,
        labels=np.tile([0, 1], 20),
        groups=np.repeat(np.arange(20), 2),
    )
    first = split_windows(windows, test_fraction=0.25, seed=3)
    second = split_windows(windows, test_fraction=0.25, seed=3)
    np.testing.assert_array_equal(first[0].groups, second[0].groups)
    assert set(first[0].groups).isdisjoint(set(first[1].groups))
    assert first[0].n_windows + first[1].n_windows == windows.n_windows


def test_split_windows_rejects_single_group():
    windows = WindowSet(
        samples=np.zeros((4, 10, 1), dtype=np.float32),
        starts=np.arange(4),
        sampling_frequency=10.0,
        window_samples=10,
        stride=10,
        groups=np.zeros(4, dtype=int),
    )
    with pytest.raises(ConfigurationError):
        split_windows(windows)


# ------------------------------------------------------------------------ preprocessing
def test_preprocessing_normalises_and_detrends():
    t = np.linspace(0, 1, 500)
    ramp = (10 * t)[:, None]
    dataset = SensorDataset(data=ramp, sampling_frequency=500.0)
    processed = preprocess(dataset, PreprocessConfig(detrend=True, normalize="minmax"))
    assert processed.data.max() <= 1.0 + 1e-6
    assert processed.data.min() >= -1e-6
    # The linear trend should be gone.
    assert abs(np.corrcoef(np.arange(processed.n_samples), processed.data[:, 0])[0, 1]) < 0.2


def test_preprocessing_resamples_with_anti_aliasing():
    fs = 1000.0
    t = np.arange(int(fs)) / fs
    data = np.sin(2 * np.pi * 20 * t)[:, None]
    dataset = SensorDataset(data=data, sampling_frequency=fs)
    processed = preprocess(dataset, PreprocessConfig(resample_hz=250.0, normalize="none"))
    assert processed.sampling_frequency == pytest.approx(250.0)
    assert processed.n_samples == pytest.approx(250, abs=2)
    # A 20 Hz tone survives decimation by 4.
    assert np.std(processed.data) > 0.5


def test_preprocessing_config_rejects_unknown_keys():
    with pytest.raises(ConfigurationError):
        PreprocessConfig.from_dict({"normalize": "minmax", "bogus": 1})
    assert PreprocessConfig.from_dict({"normalize": "minmax"}).normalize == "minmax"


def test_preprocessing_bandpass_requires_ordered_cutoffs():
    with pytest.raises(ConfigurationError):
        PreprocessConfig(highpass_hz=100.0, lowpass_hz=10.0)


def test_preprocessing_can_be_disabled():
    dataset = SensorDataset(data=np.arange(10.0)[:, None], sampling_frequency=10.0)
    processed = preprocess(dataset, {"enabled": False})
    np.testing.assert_array_equal(processed.data, dataset.data)


# ------------------------------------------------------------------------------ I/O
def test_save_and_load_npz_round_trip(tmp_path, tiny_dataset):
    path = save_dataset(tiny_dataset, tmp_path / "data.npz")
    assert path.exists()
    reloaded = load_dataset(path)
    assert reloaded.fingerprint() == tiny_dataset.fingerprint()
    np.testing.assert_array_equal(reloaded.labels, tiny_dataset.labels)
    np.testing.assert_array_equal(reloaded.groups, tiny_dataset.groups)


def test_save_and_load_csv_round_trip(tmp_path, tiny_dataset):
    path = save_csv(tiny_dataset, tmp_path / "data.csv")
    reloaded = load_csv(
        path,
        label_column="label",
        group_column="group",
        sampling_frequency=tiny_dataset.sampling_frequency,
    )
    np.testing.assert_allclose(reloaded.data, tiny_dataset.data, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(reloaded.labels, tiny_dataset.labels)


def test_load_dataset_dispatches_on_suffix(tmp_path, tiny_dataset):
    npz_path = save_dataset(tiny_dataset, tmp_path / "x.npz")
    assert load_dataset(npz_path).n_samples == tiny_dataset.n_samples
    npy_path = tmp_path / "x.npy"
    np.save(npy_path, tiny_dataset.data)
    loaded = load_dataset(npy_path, sampling_frequency=500.0)
    assert loaded.n_samples == tiny_dataset.n_samples
    with pytest.raises(DataValidationError):
        load_dataset(tmp_path / "x.parquet")


def test_load_csv_infers_sampling_rate_from_time_column(tmp_path):
    rng = np.random.default_rng(0)
    fs = 250.0
    n = 500
    t = np.arange(n) / fs
    rows = ["time,c0,c1,label,group"]
    for index in range(n):
        rows.append(f"{t[index]:.6f},{rng.standard_normal():.6f},{rng.standard_normal():.6f},{index // 250},{index // 250}")
    path = tmp_path / "ts.csv"
    path.write_text("\n".join(rows), encoding="utf-8")
    dataset = load_csv(path, time_column="time", label_column="label", group_column="group")
    assert dataset.sampling_frequency == pytest.approx(fs, rel=1e-3)
    assert dataset.channel_names == ("c0", "c1")


def test_load_csv_requires_sampling_rate_without_time_column(tmp_path):
    path = tmp_path / "plain.csv"
    path.write_text("a,b\n0.0,1.0\n0.1,1.1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_csv(path)
    dataset = load_csv(path, sampling_frequency=100.0)
    assert dataset.n_channels == 2


@pytest.mark.parametrize(
    "policy,expected_finite",
    [("interpolate", True), ("zero", True), ("drop", True), ("raise", None)],
)
def test_nan_policies(tmp_path, policy, expected_finite):
    path = tmp_path / "nan.csv"
    path.write_text("a,b\n0.0,1.0\n0.1,\n0.2,1.2\n0.3,1.3\n", encoding="utf-8")
    if expected_finite is None:
        with pytest.raises(DataValidationError):
            load_csv(path, sampling_frequency=10.0, nan_policy=policy)
        return
    dataset = load_csv(path, sampling_frequency=10.0, nan_policy=policy)
    assert np.all(np.isfinite(dataset.data))
    if policy == "drop":
        assert dataset.n_samples == 3


def test_load_rejects_missing_and_non_numeric_files(tmp_path):
    with pytest.raises(DataValidationError):
        load_dataset(tmp_path / "nope.npz")
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,x\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_csv(bad, sampling_frequency=10.0)


def test_npz_loading_disallows_pickles(tmp_path):
    # An object array must never be unpickled: loading has to fail loudly rather
    # than execute arbitrary code from a crafted archive.
    path = tmp_path / "obj.npz"
    np.savez(
        path,
        data=np.array([object()], dtype=object),
        sampling_frequency=10.0,
    )
    with pytest.raises(DataValidationError) as excinfo:
        load_npz(path)
    assert "pickle" in str(excinfo.value).lower() or isinstance(
        excinfo.value, DataValidationError
    )


def test_save_npz_and_reload_metadata(tmp_path, tiny_dataset):
    path = save_npz(tiny_dataset, tmp_path / "m.npz")
    reloaded = load_npz(path)
    assert reloaded.channel_names == tiny_dataset.channel_names
    assert reloaded.name == tiny_dataset.name
