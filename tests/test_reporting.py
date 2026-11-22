"""Tests for report generation, export formats and plotting."""

from __future__ import annotations

import json

import pytest

from sensor2spike.contracts import ExperimentResult
from sensor2spike.exceptions import ConfigurationError, ReportError
from sensor2spike.reporting import (
    export_config,
    export_result,
    generate_report,
    render_html,
    render_markdown,
    trial_rows,
)

cv2 = pytest.importorskip("matplotlib")


# ------------------------------------------------------------------------ markdown
def test_markdown_report_contains_measured_results(experiment_result):
    markdown = render_markdown(experiment_result)
    assert experiment_result.experiment_id in markdown
    # Measured numbers appear; nothing is invented.
    assert "0.95" in markdown  # accuracy of the best trial
    assert "4000" in markdown  # spike count of the best trial
    assert "Recommended configuration" in markdown
    assert "Pareto" in markdown


def test_markdown_report_has_no_absolute_paths(experiment_result, tmp_path):
    markdown = render_markdown(experiment_result)
    assert str(tmp_path) not in markdown
    assert "/home/" not in markdown


def test_html_report_is_self_contained(experiment_result):
    html = render_html(experiment_result)
    assert html.startswith("<!DOCTYPE html>")
    # No external stylesheet/script/font references: previews stay offline-safe.
    assert "http://" not in html
    assert "https://" not in html
    assert "<link" not in html


def test_trial_rows_are_tabular(experiment_result):
    rows = trial_rows(experiment_result)
    assert len(rows) == len(experiment_result.trials)
    assert {"index", "accuracy", "spike_count", "params_json"} <= set(rows[0])
    assert json.loads(rows[0]["params_json"]) == experiment_result.trials[0].params


# -------------------------------------------------------------------------- export
def test_export_result_json_round_trips(experiment_result, tmp_path):
    path = export_result(experiment_result, tmp_path / "result.json")
    payload = json.loads(path.read_text())
    assert payload["experiment_id"] == experiment_result.experiment_id
    rebuilt = ExperimentResult.from_dict(payload)
    assert len(rebuilt.trials) == len(experiment_result.trials)
    assert rebuilt.best_index == experiment_result.best_index
    assert rebuilt.best is not None


def test_export_result_csv_and_markdown(experiment_result, tmp_path):
    csv_path = export_result(experiment_result, tmp_path / "result.csv")
    header = csv_path.read_text().splitlines()[0]
    assert "accuracy" in header and "spike_count" in header
    md_path = export_result(experiment_result, tmp_path / "result.md")
    assert md_path.read_text().startswith("#")


def test_export_result_rejects_unknown_suffix(experiment_result, tmp_path):
    with pytest.raises(ConfigurationError):
        export_result(experiment_result, tmp_path / "result.xyz")


def test_export_config_is_directly_reusable(experiment_result, tmp_path):
    path = export_config(experiment_result, tmp_path / "config.json")
    config = json.loads(path.read_text())
    assert config["encoder"] == "rate"
    assert config["params"]
    # The exported config must rebuild a working encoder.
    from sensor2spike.encoders import build_encoder

    encoder = build_encoder({"encoder": config["encoder"], **config["params"]})
    assert encoder.name == "rate"


def test_export_refuses_to_overwrite_input_files(experiment_result, tmp_path):
    """Security (spec 13): reports are written next to, never over, the input."""
    protected = tmp_path / "dataset.npz"
    protected.write_bytes(b"PK\x03\x04 not really a dataset")
    with pytest.raises(ReportError):
        export_result(experiment_result, protected)
    # ... while re-exporting an existing report file is allowed.
    md = export_result(experiment_result, tmp_path / "report.md")
    export_result(experiment_result, md)


# ------------------------------------------------------------------------ bundles
def test_generate_report_writes_every_format(experiment_result, tmp_path):
    written = generate_report(experiment_result, tmp_path, plots=True)
    assert {"json", "csv", "md", "html"} <= set(written)
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0
    assert any(name.startswith("plot_") for name in written)


def test_generate_report_can_skip_plots(experiment_result, tmp_path):
    written = generate_report(experiment_result, tmp_path, formats=("json",), plots=False)
    assert set(written) == {"json"}


def test_generate_report_survives_plot_failure(experiment_result, tmp_path, monkeypatch):
    import sensor2spike.reporting.plots as plots_module

    def boom(*args, **kwargs):
        raise RuntimeError("no display")

    monkeypatch.setattr(plots_module, "plot_pareto", boom)
    written = generate_report(experiment_result, tmp_path, plots=True)
    # Reports still produced; the failure is recorded rather than raised.
    assert "json" in written
    assert "plot_error" in experiment_result.extras


def test_generate_report_never_leaks_paths(experiment_result, tmp_path):
    written = generate_report(experiment_result, tmp_path)
    for fmt in ("md", "html", "csv"):
        text = written[fmt].read_text(encoding="utf-8")
        assert str(tmp_path) not in text


# ------------------------------------------------------------------------- plots
def test_pareto_plot_is_a_png(experiment_result, tmp_path):
    from sensor2spike.reporting.plots import plot_pareto

    path = plot_pareto(experiment_result, tmp_path / "pareto.png")
    assert path.exists()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_spike_raster_plot(spike_train, tmp_path):
    from sensor2spike.reporting.plots import plot_spike_raster

    path = plot_spike_raster(spike_train, tmp_path / "raster.png")
    assert path.exists()
    assert path.stat().st_size > 0
