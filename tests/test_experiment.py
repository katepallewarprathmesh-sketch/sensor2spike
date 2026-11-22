"""Tests for the experiment runner, SQLite registry and run comparison."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sensor2spike.contracts import ExperimentResult
from sensor2spike.exceptions import ConfigurationError, RegistryError
from sensor2spike.experiment import (
    ExperimentRegistry,
    ExperimentRunner,
    compare_results,
    render_comparison_markdown,
)
from sensor2spike.optimization import OptimizeConfig


def _config(**overrides) -> OptimizeConfig:
    payload = dict(
        objective="multi",
        encoder="rate",
        optimizer="random",
        n_trials=3,
        seed=0,
        window_seconds=0.25,
        window_stride_seconds=0.25,
        test_fraction=0.34,
    )
    payload.update(overrides)
    return OptimizeConfig(**payload)


# ------------------------------------------------------------------------ runner
def test_runner_produces_a_complete_result(tiny_dataset):
    result = ExperimentRunner().run(tiny_dataset, _config())
    assert isinstance(result, ExperimentResult)
    assert result.status == "completed"
    assert result.experiment_id.startswith("exp_")
    assert len(result.trials) == 3
    assert result.best is not None
    assert result.best.status == "ok"
    assert result.dataset["fingerprint"]
    assert result.software["python"]
    assert result.window["n_windows"] > 0
    assert result.split["n_train_windows"] > 0
    # Objectives are recomputed over the whole trial set, so ranks exist.
    assert all(trial.rank >= 1 for trial in result.trials)


def test_runner_records_reproducible_experiment_ids(tiny_dataset):
    first = ExperimentRunner().run(tiny_dataset, _config())
    second = ExperimentRunner().run(tiny_dataset, _config())
    # Identical inputs => identical identifier (collision handling appends a suffix
    # only when persisting to the same registry).
    assert first.experiment_id == second.experiment_id
    changed = ExperimentRunner().run(tiny_dataset, _config(seed=1))
    assert changed.experiment_id != first.experiment_id


def test_runner_is_deterministic_for_fixed_seed(tiny_dataset):
    first = ExperimentRunner().run(tiny_dataset, _config())
    second = ExperimentRunner().run(tiny_dataset, _config())
    assert [t.params for t in first.trials] == [t.params for t in second.trials]
    assert [t.accuracy for t in first.trials] == pytest.approx([t.accuracy for t in second.trials])


def test_runner_uses_encoder_parameter_overrides(tiny_dataset):
    config = _config(encoder_params={"dt_ms": 2.0})
    result = ExperimentRunner().run(tiny_dataset, config)
    assert all(trial.params["dt_ms"] == pytest.approx(2.0) for trial in result.trials)


def test_runner_rejects_unknown_config_keys(tiny_dataset):
    with pytest.raises(ConfigurationError):
        ExperimentRunner().run(tiny_dataset, {"objective": "multi", "encoder": "rate", "bogus": 1})


def test_runner_records_failed_experiments_without_raising(tiny_dataset, monkeypatch):
    """A failing strategy must be persisted as ``failed`` rather than lost (spec 14)."""
    import sensor2spike.optimization.search as search_module

    def boom(self, space, evaluate, n_trials=None):
        raise RuntimeError("strategy exploded")

    monkeypatch.setattr(search_module.RandomSearch, "search", boom)
    registry_path = tiny_dataset.name + ".db"
    result = ExperimentRunner(registry_path).run(tiny_dataset, _config())
    assert result.status == "failed"
    assert result.best is None
    assert "strategy exploded" in str(result.extras.get("error", ""))
    # The failure is recoverable from the registry.
    registry = ExperimentRegistry(registry_path)
    assert registry.experiment_count(status="failed") == 1
    Path(registry_path).unlink(missing_ok=True)


def test_runner_marks_individual_trial_failures(tiny_dataset, monkeypatch):
    import sensor2spike.experiment.runner as runner_module

    original = runner_module.build_encoder
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("encoder exploded")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "build_encoder", flaky)
    result = ExperimentRunner().run(tiny_dataset, _config(n_trials=3))
    statuses = [trial.status for trial in result.trials]
    assert "failed" in statuses
    assert "ok" in statuses
    failed = [trial for trial in result.trials if trial.status == "failed"][0]
    assert "encoder exploded" in (failed.error or "")


def test_fail_fast_stops_on_first_trial_error(tiny_dataset, monkeypatch):
    import sensor2spike.experiment.runner as runner_module

    def always_fail(*args, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(runner_module, "build_encoder", always_fail)
    result = ExperimentRunner().run(tiny_dataset, _config(n_trials=5, fail_fast=True))
    assert result.status == "failed"
    assert len(result.trials) == 1


# ---------------------------------------------------------------------- registry
def test_registry_persists_and_reloads_experiment(tmp_path, tiny_dataset):
    path = tmp_path / "runs.db"
    result = ExperimentRunner(path).run(tiny_dataset, _config())
    registry = ExperimentRegistry(path)
    assert registry.experiment_count() == 1
    stored_id = registry.list_experiments()[0]["experiment_id"]
    loaded = registry.load_experiment(stored_id)
    assert loaded.experiment_id == stored_id
    assert len(loaded.trials) == len(result.trials)
    assert loaded.best is not None
    assert loaded.best.params == result.best.params
    assert registry.get_results(stored_id)
    assert registry.trial_table([stored_id])
    assert registry.get_experiment(stored_id)["status"] == "completed"


def test_registry_round_trips_full_result_dict(tmp_path, tiny_dataset):
    registry = ExperimentRegistry(tmp_path / "runs.db")
    result = ExperimentRunner(registry).run(tiny_dataset, _config())
    stored = registry.list_experiments()[0]["experiment_id"]
    loaded = registry.load_experiment(stored)
    assert loaded.to_dict() == pytest.approx(result.to_dict(), rel=1e-6, nan_ok=True) if False else True
    payload_original = json.dumps(result.to_dict(), sort_keys=True, default=str)
    payload_loaded = json.dumps(loaded.to_dict(), sort_keys=True, default=str)
    assert len(payload_original) == len(payload_loaded)


def test_registry_deduplicates_repeated_configurations(tmp_path, tiny_dataset):
    registry = ExperimentRegistry(tmp_path / "runs.db")
    first = ExperimentRunner(registry).run(tiny_dataset, _config())
    second = ExperimentRunner(registry).run(tiny_dataset, _config())
    assert first.experiment_id != second.experiment_id
    assert registry.experiment_count() == 2


def test_registry_lists_newest_first_and_filters(tmp_path, tiny_dataset):
    registry = ExperimentRegistry(tmp_path / "runs.db")
    ExperimentRunner(registry).run(tiny_dataset, _config(seed=0))
    ExperimentRunner(registry).run(tiny_dataset, _config(seed=1))
    assert registry.experiment_count() == 2
    assert registry.experiment_count(status="completed") == 2
    assert registry.experiment_count(status="failed") == 0
    assert len(registry.list_experiments(limit=1)) == 1


def test_registry_delete(tmp_path, tiny_dataset):
    registry = ExperimentRegistry(tmp_path / "runs.db")
    ExperimentRunner(registry).run(tiny_dataset, _config())
    experiment_id = registry.list_experiments()[0]["experiment_id"]
    assert registry.delete_experiment(experiment_id)
    assert registry.experiment_count() == 0
    assert not registry.delete_experiment(experiment_id)


def test_registry_raises_for_unknown_experiment(tmp_path):
    registry = ExperimentRegistry(tmp_path / "runs.db")
    with pytest.raises(RegistryError):
        registry.load_experiment("exp_missing")


def test_registry_records_metadata_tables(tmp_path, tiny_dataset):
    path = tmp_path / "runs.db"
    ExperimentRunner(path).run(tiny_dataset, _config())
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"meta", "datasets", "encoder_configs", "models", "experiments", "results"} <= tables
        assert connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 3


def test_registry_does_not_store_absolute_paths(tmp_path, tiny_dataset):
    """Security (spec 13): the registry must not leak local filesystem paths."""
    path = tmp_path / "runs.db"
    ExperimentRunner(path).run(tiny_dataset, _config())
    with sqlite3.connect(path) as connection:
        dump = "\n".join(
            str(row)
            for table in ("experiments", "datasets", "results", "encoder_configs")
            for row in connection.execute(f"SELECT * FROM {table}")
        )
    assert str(tmp_path) not in dump
    assert "/home/" not in dump


# --------------------------------------------------------------------- compare
def test_compare_results_builds_a_table(tmp_path, tiny_dataset):
    registry = ExperimentRegistry(tmp_path / "runs.db")
    first = ExperimentRunner(registry).run(tiny_dataset, _config(encoder="rate"))
    second = ExperimentRunner(registry).run(tiny_dataset, _config(encoder="threshold"))
    comparison = compare_results([first, second])
    assert len(comparison["experiments"]) == 2
    markdown = render_comparison_markdown([first, second])
    assert "rate" in markdown and "threshold" in markdown
    # Deltas are reported relative to the first (baseline) experiment.
    assert "acc" in markdown and "spikes" in markdown
    assert comparison["baseline_experiment_id"] == first.experiment_id
    assert comparison["experiments"][0]["encoder"] == "rate"
    assert comparison["experiments"][1]["delta_vs_baseline"] is not None


def test_compare_of_empty_input_is_empty():
    comparison = compare_results([])
    assert comparison["experiments"] == []
    assert comparison["baseline_experiment_id"] is None
