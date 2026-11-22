"""End-to-end tests for the command-line interface."""

from __future__ import annotations

import json

import pytest

from sensor2spike.cli import main


@pytest.fixture()
def csv_dataset(tmp_path, tiny_dataset):
    from sensor2spike.data import save_csv

    return save_csv(tiny_dataset, tmp_path / "tiny.csv")


@pytest.fixture()
def npz_dataset(tmp_path, tiny_dataset):
    from sensor2spike.data import save_dataset

    return save_dataset(tiny_dataset, tmp_path / "tiny.npz")


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ------------------------------------------------------------------------- basics
def test_version_and_help(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "sensor2spike" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        main(["--help"])


def test_encoders_command_lists_every_strategy(capsys):
    code, out, _ = run(capsys, "encoders")
    assert code == 0
    for name in ("rate", "latency", "threshold", "delta", "population"):
        assert name in out
    assert "lif_ridge" in out


def test_no_command_is_an_error(capsys):
    with pytest.raises(SystemExit):
        main([])
    assert capsys.readouterr().err


# ----------------------------------------------------------------------- validate
def test_validate_csv(npz_dataset, capsys):
    code, out, err = run(
        capsys,
        "validate",
        str(npz_dataset),
    )
    assert code == 0, err
    assert "Dataset validation: OK" in out
    assert "sampling rate" in out


def test_validate_csv_with_columns(csv_dataset, capsys):
    code, out, err = run(
        capsys,
        "validate",
        str(csv_dataset),
        "--label-column",
        "label",
        "--group-column",
        "group",
        "--sampling-frequency",
        "500",
    )
    assert code == 0, err
    assert "labels" in out


def test_validate_reports_errors_without_traceback(tmp_path, capsys):
    code, _, err = run(capsys, "validate", str(tmp_path / "missing.csv"))
    assert code != 0
    assert "error:" in err
    assert "Traceback" not in err


def test_error_output_redacts_absolute_paths(tmp_path, capsys):
    missing = tmp_path / "secret_project" / "data.csv"
    code, _, err = run(capsys, "validate", str(missing))
    assert code != 0
    assert str(tmp_path) not in err


# ------------------------------------------------------------------------- encode
def test_encode_writes_spike_file(npz_dataset, tmp_path, capsys):
    out_file = tmp_path / "spikes.npz"
    code, out, err = run(
        capsys,
        "encode",
        str(npz_dataset),
        "--method",
        "rate",
        "--out",
        str(out_file),
        "--param",
        "rate_max_hz=150",
    )
    assert code == 0, err
    assert out_file.exists()
    assert "Spike count" in out
    from sensor2spike.data import load_spike_train

    spikes = load_spike_train(out_file)
    assert spikes.encoder == "rate"
    assert spikes.spike_count > 0


def test_encode_with_plot(npz_dataset, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out_file = tmp_path / "spikes.npz"
    code, _, err = run(
        capsys, "encode", str(npz_dataset), "--method", "delta", "--out", str(out_file), "--plot"
    )
    assert code == 0, err
    assert out_file.with_suffix(".png").exists()


def test_encode_rejects_unknown_parameter(npz_dataset, tmp_path, capsys):
    code, _, err = run(
        capsys,
        "encode",
        str(npz_dataset),
        "--method",
        "rate",
        "--out",
        str(tmp_path / "s.npz"),
        "--param",
        "not_a_param=1",
    )
    assert code != 0
    assert "error:" in err


def test_encode_rejects_malformed_param(npz_dataset, capsys):
    code, _, err = run(capsys, "encode", str(npz_dataset), "--param", "oops")
    assert code != 0
    assert "key=value" in err


# ----------------------------------------------------------------------- optimize
def test_optimize_end_to_end(npz_dataset, tmp_path, capsys):
    registry = tmp_path / "runs.db"
    out_dir = tmp_path / "results"
    code, out, err = run(
        capsys,
        "optimize",
        str(npz_dataset),
        "--encoder",
        "rate",
        "--optimizer",
        "random",
        "--trials",
        "2",
        "--window",
        "0.25",
        "--registry",
        str(registry),
        "--out",
        str(out_dir),
        "--no-plots",
    )
    assert code == 0, err
    assert "Experiment:" in out
    assert "Status: completed" in out
    assert registry.exists()
    assert list(out_dir.glob("*.json"))
    assert list(out_dir.glob("*.md"))


def test_optimize_records_experiment_in_registry(npz_dataset, tmp_path, capsys):
    registry = tmp_path / "runs.db"
    code, _, err = run(
        capsys,
        "optimize",
        str(npz_dataset),
        "--trials",
        "2",
        "--window",
        "0.25",
        "--registry",
        str(registry),
        "--out",
        str(tmp_path / "out"),
        "--no-plots",
    )
    assert code == 0, err
    code, out, err = run(capsys, "list", "--registry", str(registry))
    assert code == 0, err
    assert "exp_" in out
    assert "completed" in out


def test_optimize_supports_grid_and_weighted(npz_dataset, tmp_path, capsys):
    code, _, err = run(
        capsys,
        "optimize",
        str(npz_dataset),
        "--encoder",
        "delta",
        "--optimizer",
        "grid",
        "--trials",
        "2",
        "--objective",
        "weighted",
        "--window",
        "0.25",
        "--registry",
        str(tmp_path / "r.db"),
        "--out",
        str(tmp_path / "o"),
        "--no-plots",
    )
    assert code == 0, err


# ------------------------------------------------------- report / compare / export
@pytest.fixture()
def stored_experiment(npz_dataset, tmp_path):
    registry = tmp_path / "runs.db"
    code = main(
        [
            "optimize",
            str(npz_dataset),
            "--encoder",
            "rate",
            "--trials",
            "2",
            "--window",
            "0.25",
            "--registry",
            str(registry),
            "--out",
            str(tmp_path / "out"),
            "--no-plots",
        ]
    )
    assert code == 0
    from sensor2spike.experiment import ExperimentRegistry

    experiment_id = ExperimentRegistry(registry).list_experiments()[0]["experiment_id"]
    return registry, experiment_id


def test_report_regenerates_artefacts(stored_experiment, tmp_path, capsys):
    registry, experiment_id = stored_experiment
    out_dir = tmp_path / "reports"
    code, out, err = run(
        capsys,
        "report",
        "--registry",
        str(registry),
        "--experiment",
        experiment_id,
        "--out",
        str(out_dir),
        "--formats",
        "json,md",
        "--no-plots",
    )
    assert code == 0, err
    assert sorted(path.suffix for path in out_dir.iterdir()) == [".json", ".md"]


def test_export_and_compare(stored_experiment, tmp_path, capsys):
    registry, experiment_id = stored_experiment
    config_path = tmp_path / "config.json"
    code, _, err = run(
        capsys,
        "export",
        "--registry",
        str(registry),
        "--experiment",
        experiment_id,
        "--out",
        str(config_path),
    )
    assert code == 0, err
    payload = json.loads(config_path.read_text())
    assert payload["encoder"] == "rate"
    assert payload["encoder_config"]["encoder"] == "rate"

    comparison_path = tmp_path / "comparison.md"
    code, out, err = run(
        capsys,
        "compare",
        "--registry",
        str(registry),
        experiment_id,
        experiment_id,
        "--out",
        str(comparison_path),
    )
    assert code == 0, err
    assert "Experiment comparison" in out
    assert comparison_path.exists()


def test_report_on_missing_experiment_fails_cleanly(stored_experiment, tmp_path, capsys):
    registry, _ = stored_experiment
    code, _, err = run(
        capsys, "report", "--registry", str(registry), "--experiment", "exp_nope"
    )
    assert code != 0
    assert "error:" in err


# --------------------------------------------------------------------------- demo
def test_demo_generates_dataset(tmp_path, capsys):
    out_file = tmp_path / "demo.npz"
    code, out, err = run(capsys, "demo", "--out", str(out_file), "--trials", "8", "--csv")
    assert code == 0, err
    assert out_file.exists()
    assert out_file.with_suffix(".csv").exists()
    assert "class" in out.lower()
