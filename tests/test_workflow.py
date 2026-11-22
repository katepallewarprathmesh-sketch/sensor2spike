"""End-to-end acceptance tests: the developer workflow from the specification.

Covers the success criterion (section 2): install, load a dataset, choose an
encoder, optimise, evaluate, compare, export a configuration and reproduce the
experiment — plus the security and reliability requirements of sections 13/14.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sensor2spike
from sensor2spike import (
    AutoTuner,
    ExperimentRegistry,
    SensorDataset,
    build_backend,
    build_encoder,
    compare_results,
    load_dataset,
    save_dataset,
)
from sensor2spike._core.exceptions import DataValidationError
from sensor2spike._core.utils.security import redact_paths, safe_name
from sensor2spike.demo import generate_demo_dataset


# ---------------------------------------------------------------- public API surface
def test_public_api_is_importable():
    """Everything a developer needs is reachable from the package root."""
    for name in (
        "AutoTuner",
        "OptimizeConfig",
        "SensorDataset",
        "ExperimentRegistry",
        "build_encoder",
        "build_backend",
        "load_dataset",
        "save_dataset",
        "generate_report",
        "export_config",
        "compare_results",
    ):
        assert hasattr(sensor2spike, name), name
    assert sensor2spike.__version__


def test_documented_sdk_snippet_works(tmp_path):
    """The README's headline example runs as written."""
    dataset = generate_demo_dataset(n_trials=16, seed=1, duration_seconds=0.5)
    path = save_dataset(dataset, tmp_path / "radar.npz")

    dataset = SensorDataset.load(path)
    result = (
        AutoTuner(objective="multi_objective", encoder="rate", n_trials=2, seed=0, window_seconds=0.25)
        .optimize(dataset)
        .export(tmp_path / "result.json")
    )
    assert Path(result).exists()
    payload = json.loads(Path(result).read_text())
    assert payload["status"] == "completed"
    assert payload["trials"]


# ------------------------------------------------------------------ reproduce exactly
def test_experiment_can_be_reproduced_bit_for_bit(tmp_path):
    dataset = generate_demo_dataset(n_trials=12, seed=2, duration_seconds=0.4)
    registry = ExperimentRegistry(tmp_path / "runs.db")
    config = dict(
        objective="multi",
        encoder="rate",
        optimizer="random",
        n_trials=3,
        seed=11,
        window_seconds=0.2,
        test_fraction=0.34,
    )
    first = AutoTuner(registry=registry, **config).optimize(dataset)
    stored = registry.list_experiments()[0]
    original = registry.load_experiment(stored["experiment_id"])

    # Re-running the identical configuration reproduces every measurement.
    again = AutoTuner(**config).optimize(dataset)
    assert [t.accuracy for t in again.trials] == pytest.approx([t.accuracy for t in original.trials])
    assert [t.spike_count for t in again.trials] == [t.spike_count for t in original.trials]
    assert again.best.params == original.best.params

    # The stored record carries everything needed to reproduce it.
    assert first.software["python"]
    assert first.dataset["fingerprint"]
    assert first.seed == 11
    assert first.backend["backend"] == "lif_ridge"
    assert first.window["window_samples"] > 0


def test_exported_config_rebuilds_the_encoder(experiment_result, tmp_path):
    from sensor2spike._core.reporting import export_config

    path = export_config(experiment_result, tmp_path / "best.json")
    payload = json.loads(path.read_text())
    encoder = build_encoder(payload["encoder_config"])
    assert encoder.name == payload["encoder"]
    for key, value in payload["params"].items():
        assert getattr(encoder, key) == pytest.approx(value) if isinstance(value, float) else getattr(encoder, key) == value


# --------------------------------------------------------------------- compare path
def test_compare_two_encoders_through_the_sdk(tmp_path):
    dataset = generate_demo_dataset(n_trials=12, seed=3, duration_seconds=0.4)
    registry = ExperimentRegistry(tmp_path / "runs.db")
    common = dict(objective="multi", optimizer="random", n_trials=2, seed=0, window_seconds=0.2)
    rate = AutoTuner(registry=registry, encoder="rate", **common).optimize(dataset)
    delta = AutoTuner(registry=registry, encoder="delta", **common).optimize(dataset)

    comparison = compare_results([rate, delta])
    assert [entry["encoder"] for entry in comparison["experiments"]] == ["rate", "delta"]
    assert comparison["experiments"][1]["delta_vs_baseline"] is not None
    assert comparison["baseline_experiment_id"] == rate.experiment_id


# ------------------------------------------------------------------------ security
def test_redact_paths_hides_directories():
    text = "failed to read /home/alice/private/data.csv and C:\\Users\\alice\\data.csv"
    redacted = redact_paths(text)
    assert "/home/alice" not in redacted
    assert "C:\\Users" not in redacted
    assert "data.csv" in redacted


def test_safe_name_strips_unsafe_characters():
    assert safe_name("../../etc/passwd") == "etc_passwd"
    assert safe_name("report 2026/11") == "report_2026_11"
    assert safe_name("") == "artifact"


def test_error_messages_do_not_leak_paths(tiny_dataset, tmp_path, monkeypatch):
    import sensor2spike._core.experiment.runner as runner_module

    secret = tmp_path / "very_secret_folder" / "data.npz"

    def leaky(*args, **kwargs):
        raise RuntimeError(f"could not read {secret}")

    monkeypatch.setattr(runner_module, "build_encoder", leaky)
    registry = ExperimentRegistry(tmp_path / "runs.db")
    result = AutoTuner(registry=registry, encoder="rate", n_trials=1, window_seconds=0.2).optimize(
        tiny_dataset
    )
    trial = result.trials[0]
    assert trial.status == "failed"
    assert str(tmp_path) not in (trial.error or "")
    assert "data.npz" in (trial.error or "")


def test_registry_does_not_store_home_directory(tmp_path, tiny_dataset):
    registry = ExperimentRegistry(tmp_path / "runs.db")
    AutoTuner(registry=registry, encoder="rate", n_trials=1, window_seconds=0.2).optimize(tiny_dataset)
    experiment_id = registry.list_experiments()[0]["experiment_id"]
    loaded = registry.load_experiment(experiment_id)
    serialised = json.dumps(loaded.to_dict(), default=str)
    assert "/home/" not in serialised
    assert str(tmp_path) not in serialised


def test_outputs_are_written_away_from_inputs(tmp_path, tiny_dataset):
    """Reports default to their own directory; the dataset is never modified."""
    dataset_path = save_dataset(tiny_dataset, tmp_path / "data" / "tiny.npz")
    before = dataset_path.read_bytes()
    dataset = load_dataset(dataset_path)
    result = AutoTuner(encoder="rate", n_trials=1, window_seconds=0.2).optimize(dataset)
    written = result.export(tmp_path / "reports" / "run.json")
    assert Path(written).parent.name == "reports"
    assert dataset_path.read_bytes() == before


def test_no_arbitrary_code_execution_on_load(tmp_path):
    """Pickled payloads are rejected outright (allow_pickle=False)."""
    import pickle

    class Payload:
        """A trivially picklable object: loading must never unpickle it."""

        def __reduce__(self):  # pragma: no cover - would execute on load
            return (print, ("code executed",))

    path = tmp_path / "evil.npz"
    np.savez(path, data=np.array([pickle.dumps(Payload())], dtype=object))
    with pytest.raises(DataValidationError):
        load_dataset(path)


# --------------------------------------------------------------------- reliability
def test_failed_experiment_is_recorded_not_lost(tmp_path, tiny_dataset, monkeypatch):
    import sensor2spike._core.optimization.search as search_module

    def boom(self, space, evaluate, n_trials=None):
        raise RuntimeError("simulated backend crash")

    monkeypatch.setattr(search_module.RandomSearch, "search", boom)
    registry = ExperimentRegistry(tmp_path / "runs.db")
    result = AutoTuner(registry=registry, encoder="rate", n_trials=2, window_seconds=0.2).optimize(
        tiny_dataset
    )
    assert result.status == "failed"
    assert registry.experiment_count(status="failed") == 1
    stored = registry.list_experiments()[0]
    assert stored["status"] == "failed"


def test_partial_trial_failures_do_not_break_the_run(tiny_dataset, monkeypatch):
    import sensor2spike._core.experiment.runner as runner_module

    original = runner_module.build_encoder
    state = {"n": 0}

    def flaky(*args, **kwargs):
        state["n"] += 1
        if state["n"] % 2 == 0:
            raise ValueError("bad configuration")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "build_encoder", flaky)
    result = AutoTuner(encoder="rate", n_trials=4, window_seconds=0.2).optimize(tiny_dataset)
    assert result.status == "completed"
    assert any(trial.status == "failed" for trial in result.trials)
    assert result.best is not None


def test_reports_only_contain_measured_values(experiment_result):
    """Spec: never claim unmeasured performance — report only recorded metrics."""
    from sensor2spike._core.reporting import render_markdown

    markdown = render_markdown(experiment_result)
    for word in ("state-of-the-art", "SOTA", "superior to", "outperforms"):
        assert word.lower() not in markdown.lower()
    # The accuracy that appears is exactly the measured accuracy of the best trial.
    best_accuracy = experiment_result.best.accuracy
    assert f"{best_accuracy:.4f}" in markdown or f"{best_accuracy:g}" in markdown


# ----------------------------------------------------------------- notebook-style run
def test_full_manual_pipeline(tiny_dataset):
    """Do it by hand: windows -> split -> encode -> train -> evaluate."""
    from sensor2spike._core.analysis import compute_spike_stats
    from sensor2spike._core.data import PreprocessConfig, preprocess, split_windows

    processed = preprocess(tiny_dataset, PreprocessConfig(detrend=True, normalize="zscore"))
    windows = processed.to_windows(window_samples=64, stride=64)
    train, test = split_windows(windows, test_fraction=0.34, seed=0)

    encoder = build_encoder("population", neurons_per_channel=6, sigma=0.3, seed=0)
    encoder.fit(np.concatenate(list(train.samples), axis=0), fs=processed.sampling_frequency)

    train_spikes = encoder.transform_windows(train)
    test_spikes = encoder.transform_windows(test)
    stats = compute_spike_stats(train_spikes)

    backend = build_backend("lif_ridge", tau_ms=15.0, alpha=0.5)
    evaluation = backend.train_evaluate(train_spikes, train.labels, test_spikes, test.labels)

    assert 0.0 <= evaluation.accuracy <= 1.0
    assert stats.spike_count > 0
    assert stats.active_fraction > 0
    assert evaluation.n_parameters > 0
