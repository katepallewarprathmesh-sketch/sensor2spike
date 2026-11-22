"""Report generation: JSON, CSV, Markdown and a self-contained HTML page.

Every report carries a reproducibility block (seed, dataset fingerprint,
dependency versions) so a stored run can be reconstructed later.
"""

from __future__ import annotations

import base64
import csv
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..contracts import ExperimentResult
from ..exceptions import ConfigurationError, ReportError
from ..utils.security import redact_paths

__all__ = [
    "export_result",
    "export_config",
    "generate_report",
    "render_markdown",
    "render_html",
    "trial_rows",
    "write_csv",
]

CSV_COLUMNS = (
    "index",
    "rank",
    "status",
    "feasible",
    "accuracy",
    "macro_f1",
    "train_accuracy",
    "spike_count",
    "spike_rate_hz",
    "spikes_per_second_per_neuron",
    "temporal_sparsity",
    "information_loss",
    "reconstruction_r2",
    "compute_cost",
    "inference_time_s",
    "train_time_s",
    "score",
    "is_pareto",
    "params_json",
)


# --------------------------------------------------------------------------------------
# Serialisation helpers
# --------------------------------------------------------------------------------------
def trial_rows(result: ExperimentResult) -> list[dict[str, Any]]:
    """Flatten trials into row-oriented dictionaries (one per configuration)."""
    rows: list[dict[str, Any]] = []
    for trial in result.trials:
        rows.append(
            {
                "index": trial.index,
                "rank": trial.rank,
                "status": trial.status,
                "feasible": trial.feasible,
                "accuracy": trial.accuracy,
                "macro_f1": trial.macro_f1,
                "train_accuracy": "" if trial.train_accuracy is None else trial.train_accuracy,
                "spike_count": trial.spike_count,
                "spike_rate_hz": trial.spike_rate_hz,
                "spikes_per_second_per_neuron": trial.spikes_per_second_per_neuron,
                "temporal_sparsity": trial.temporal_sparsity,
                "information_loss": trial.information_loss,
                "reconstruction_r2": trial.reconstruction_r2,
                "compute_cost": trial.compute_cost,
                "inference_time_s": trial.inference_time_s,
                "train_time_s": trial.train_time_s,
                "score": trial.score,
                "is_pareto": trial.is_pareto,
                "params_json": json.dumps(trial.params, sort_keys=True, default=str),
                **{f"param_{key}": value for key, value in trial.params.items()},
            }
        )
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write rows to CSV, preserving column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ReportError("Cannot write a CSV report without any rows.")
    fieldnames: list[str] = list(CSV_COLUMNS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


_REPORT_FORMATS = frozenset({"json", "csv", "md", "markdown"})

_DATA_SUFFIXES = frozenset({"csv", "npz", "npy", "parquet", "h5", "hdf5", "mat", "bin", "dat", "tsv", "jsonl"})


def _guard_target(path: Path, formats: Sequence[str]) -> None:
    """Refuse to overwrite a file that this exporter did not create.

    Spec 13: generated outputs must stay separate from input data, so writing a
    report over an existing dataset is always a mistake.
    """
    if not path.exists():
        return
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {fmt.lower() for fmt in formats}:
        return
    if suffix not in _DATA_SUFFIXES:
        # Unknown extension: assume the caller knows what they are doing.
        return
    raise ReportError(
        f"Refusing to write a report to {path.name!r}: the existing file is not a "
        f"report artefact (expected one of {sorted(formats)}). Choose another path."
    )


def export_result(result: ExperimentResult, path: str | Path, *, fmt: str | None = None) -> Path:
    """Export an experiment to JSON, CSV or Markdown, chosen by suffix."""
    path = Path(path)
    suffix = (fmt or path.suffix.lstrip(".")).lower()
    # Refuse to clobber an input file first: when a caller points a report at a
    # dataset, that is the more important diagnosis than the format error.
    _guard_target(path, (suffix,) if suffix in _REPORT_FORMATS else _REPORT_FORMATS)
    if suffix not in _REPORT_FORMATS:
        raise ConfigurationError(
            f"Unsupported export format {suffix!r}. Use '.json', '.csv' or '.md'."
        )
    if suffix == "json":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")
        return path
    if suffix == "csv":
        return write_csv(trial_rows(result), path)
    if suffix in {"md", "markdown"}:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(result), encoding="utf-8")
        return path
    raise ConfigurationError(
        f"Unsupported export format {suffix!r}. Use '.json', '.csv' or '.md'."
    )


def export_config(result: ExperimentResult, path: str | Path) -> Path:
    """Export the recommended encoder configuration as a portable JSON file."""
    path = Path(path)
    _guard_target(path, ("json",))
    path.parent.mkdir(parents=True, exist_ok=True)
    best = result.best
    encoder_name = result.encoder.get("encoder") or result.encoder.get("name")
    params = {} if best is None else dict(best.params)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_id": result.experiment_id,
        "encoder": encoder_name,
        # Flat enough to feed straight into ``build_encoder(config)`` ...
        "params": params,
        "encoder_config": {"encoder": encoder_name, **params},
        "metrics": None if best is None else _best_metrics(best),
        "preprocess": result.preprocess,
        "window": result.window,
        "backend": result.backend,
        "seed": result.seed,
        "dataset": result.dataset,
        "reproduce": (
            f"AutoTuner(encoder={encoder_name!r}, seed={result.seed}, "
            f"encoder_params={params!r}).optimize(dataset)"
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _best_metrics(trial: Any) -> dict[str, float]:
    """Measured metrics for the recommended trial (no derived claims)."""
    return {
        "accuracy": float(trial.accuracy),
        "macro_f1": float(trial.macro_f1),
        "spike_count": int(trial.spike_count),
        "information_loss": float(trial.information_loss),
        "compute_cost": float(trial.compute_cost),
    }


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------
def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == 0:
            return "0"
        return f"{value:.{digits}g}"
    return str(value)


def _params_inline(params: Mapping[str, Any]) -> str:
    if not params:
        return "(defaults)"
    return ", ".join(f"{key}={_fmt(value)}" for key, value in sorted(params.items()))


def render_markdown(result: ExperimentResult) -> str:
    """Render a complete Markdown report."""
    best = result.best
    lines: list[str] = []
    lines.append("# Sensor2Spike experiment report")
    lines.append("")
    lines.append(f"- **Experiment ID**: `{result.experiment_id or 'not persisted'}`")
    lines.append(f"- **Status**: {result.status}")
    lines.append(f"- **Created**: {result.created_at or '-'}")
    lines.append(f"- **Duration**: {result.duration_s:.2f} s")
    lines.append(f"- **Encoder**: `{result.encoder.get('encoder')}`")
    lines.append(f"- **Search**: `{result.optimizer}`, {result.n_trials} candidate(s), seed {result.seed}")
    lines.append(f"- **Objective**: `{result.objective}`")
    if result.notes:
        lines.append(f"- **Notes**: {result.notes}")
    lines.append("")

    # Dataset
    lines.append("## Dataset")
    lines.append("")
    dataset = result.dataset
    lines.append("| Property | Value |")
    lines.append("|---|---|")
    for key in (
        "name",
        "sensor_type",
        "n_samples",
        "n_channels",
        "sampling_frequency",
        "duration_seconds",
        "n_classes",
        "n_groups",
        "fingerprint",
    ):
        if key in dataset:
            lines.append(f"| {key} | {_fmt(dataset[key])} |")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append("### Windowing")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|---|---|")
    for key, value in result.window.items():
        lines.append(f"| {key} | {_fmt(value)} |")
    lines.append("")
    lines.append("### Train/test split")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|---|---|")
    for key, value in result.split.items():
        lines.append(f"| {key} | {_fmt(value)} |")
    lines.append("")
    lines.append("### Backend")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(result.backend, indent=2, sort_keys=True, default=str))
    lines.append("```")
    lines.append("")
    lines.append("### Objective")
    lines.append("")
    lines.append("`J = w_a·(1 − accuracy) + w_s·spikes + w_l·information_loss + w_c·compute_cost`")
    lines.append("")
    lines.append("| Weight | Value |")
    lines.append("|---|---|")
    for key, value in result.weights.items():
        lines.append(f"| {key} | {_fmt(value)} |")
    lines.append("")
    if result.accuracy_target is not None:
        lines.append(f"Accuracy constraint: `accuracy >= {result.accuracy_target}`")
    if result.spike_budget is not None:
        lines.append(f"Spike budget: `spike_count <= {result.spike_budget}`")
    lines.append("")

    # Results
    lines.append("## Results")
    lines.append("")
    summary = result.extras.get("summary", {})
    if summary:
        lines.append(
            f"- Trials completed: **{summary.get('n_ok', 0)}/{summary.get('n_trials', 0)}**, "
            f"feasible: **{summary.get('n_feasible', 0)}**, Pareto-optimal: **{summary.get('n_pareto', 0)}**"
        )
        if "accuracy_max" in summary:
            lines.append(
                f"- Accuracy range: **{summary['accuracy_min']:.4f} – {summary['accuracy_max']:.4f}** "
                f"(mean {summary['accuracy_mean']:.4f})"
            )
        if "spike_count_min" in summary:
            lines.append(
                f"- Spike-count range: **{summary['spike_count_min']} – {summary['spike_count_max']}**"
            )
        lines.append("")
    lines.append("### Top configurations")
    lines.append("")
    lines.append("| Rank | Params | Accuracy | Macro F1 | Spikes | Info loss | Cost | Score | Pareto |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for trial in result.top(10):
        lines.append(
            "| {rank} | `{params}` | {acc} | {f1} | {spikes} | {loss} | {cost} | {score} | {pareto} |".format(
                rank=trial.rank if trial.rank else "-",
                params=_params_inline(trial.params),
                acc=_fmt(trial.accuracy),
                f1=_fmt(trial.macro_f1),
                spikes=trial.spike_count,
                loss=_fmt(trial.information_loss),
                cost=_fmt(trial.compute_cost),
                score=_fmt(trial.score),
                pareto="yes" if trial.is_pareto else "",
            )
        )
    lines.append("")

    # Pareto front
    front = sorted(result.pareto_front, key=lambda trial: (trial.spike_count, -trial.accuracy))
    if front:
        lines.append("### Pareto front")
        lines.append("")
        lines.append("| Accuracy | Spikes | Info loss | Cost | Params |")
        lines.append("|---|---|---|---|---|")
        for trial in front:
            lines.append(
                f"| {_fmt(trial.accuracy)} | {trial.spike_count} | {_fmt(trial.information_loss)} | {_fmt(trial.compute_cost)} | `{_params_inline(trial.params)}` |"
            )
        lines.append("")

    # Best configuration
    lines.append("## Recommended configuration")
    lines.append("")
    if best is None:
        lines.append("_No feasible configuration was found._")
    else:
        lines.append("```json")
        lines.append(
            json.dumps(
                {
                    "encoder": result.encoder.get("encoder"),
                    "params": best.params,
                    "metrics": {
                        "accuracy": best.accuracy,
                        "macro_f1": best.macro_f1,
                        "spike_count": best.spike_count,
                        "information_loss": best.information_loss,
                        "compute_cost": best.compute_cost,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        lines.append("```")
        lines.append("")
        lines.append(f"Selection rule: `{result.extras.get('selection', 'balanced')}`.")
    lines.append("")

    # Reproducibility
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- Dataset fingerprint: `{result.dataset.get('fingerprint', '-')}`")
    lines.append(f"- Seed: `{result.seed}`")
    lines.append("- Software: " + ", ".join(f"{key}={value}" for key, value in result.software.items()))
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Compute cost is an analytical proxy (operations per spike and per neuron-step), "
        "not a measurement on neuromorphic hardware."
    )
    lines.append(
        "- Objective normalisation is relative to the trials of this experiment; scores are "
        "comparable within an experiment, not across experiments."
    )
    lines.append(
        "- Accuracy is estimated on a single group-level train/test split; repeat with "
        "multiple seeds for a confidence interval."
    )
    lines.append(
        "- The default backend is a NumPy LIF readout. Validate the final configuration with a "
        "framework-specific backend before deployment."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------
def _image_data_uri(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_html(result: ExperimentResult, images: Mapping[str, str | Path] | None = None) -> str:
    """Render a self-contained HTML report (no external assets)."""
    images = images or {}
    best = result.best
    summary = result.extras.get("summary", {})
    cards = [
        ("Accuracy", _fmt(None if best is None else best.accuracy)),
        ("Spike count", _fmt(None if best is None else best.spike_count)),
        ("Information loss", _fmt(None if best is None else best.information_loss)),
        ("Pareto solutions", _fmt(summary.get("n_pareto"))),
        ("Trials", _fmt(result.n_trials)),
    ]
    card_html = "".join(
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div></div>'
        for label, value in cards
    )
    rows_html = []
    for trial in result.top(15):
        pareto = '<span class="badge">Pareto</span>' if trial.is_pareto else ""
        rows_html.append(
            "<tr>"
            f"<td>{trial.rank or '-'}</td>"
            f"<td><code>{_params_inline(trial.params)}</code></td>"
            f"<td class='num'>{_fmt(trial.accuracy)}</td>"
            f"<td class='num'>{trial.spike_count}</td>"
            f"<td class='num'>{_fmt(trial.information_loss)}</td>"
            f"<td class='num'>{_fmt(trial.compute_cost)}</td>"
            f"<td>{pareto}</td>"
            "</tr>"
        )
    figure_html = ""
    for key, path in images.items():
        uri = _image_data_uri(path)
        if uri:
            figure_html += (
                f'<figure><img alt="{key}" src="{uri}">'
                f"<figcaption>{key.replace('_', ' ')}</figcaption></figure>"
            )
    config_json = json.dumps(result.encoder, indent=2, sort_keys=True, default=str)
    best_json = "{}" if best is None else json.dumps(
        {"params": best.params, "accuracy": best.accuracy, "spike_count": best.spike_count},
        indent=2,
        sort_keys=True,
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sensor2Spike — {result.experiment_id or 'experiment'}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 32px; background: #f6f7f9; color: #1f2430; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; }}
  h2 {{ font-size: 16px; margin-top: 32px; border-bottom: 1px solid #d8dde6; padding-bottom: 6px; }}
  .subtle {{ color: #6b7280; font-size: 13px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 20px; }}
  .card {{ background: #fff; border: 1px solid #e3e7ee; border-radius: 10px; padding: 14px 16px; }}
  .card-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #6b7280; }}
  .card-value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 13px;
           border: 1px solid #e3e7ee; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #eef1f6; }}
  th {{ background: #fafbfc; font-weight: 600; }}
  tr:last-child td {{ border-bottom: none; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  code {{ background: #f0f2f6; padding: 1px 4px; border-radius: 4px; font-size: 12px; }}
  pre {{ background: #0f172a; color: #e2e8f0; padding: 14px; border-radius: 10px; overflow-x: auto; }}
  .badge {{ background: #dcfce7; color: #166534; padding: 1px 6px; border-radius: 999px; font-size: 11px; }}
  figure {{ margin: 18px 0; background: #fff; border: 1px solid #e3e7ee; border-radius: 10px; padding: 10px; }}
  img {{ max-width: 100%; height: auto; }}
  figcaption {{ font-size: 12px; color: #6b7280; margin-top: 6px; }}
  ul {{ line-height: 1.6; }}
</style>
</head>
<body>
  <h1>Sensor2Spike experiment report</h1>
  <div class="subtle">Experiment {result.experiment_id or 'not persisted'} · created {result.created_at or '-'} ·
    encoder <code>{result.encoder.get('encoder')}</code> · search <code>{result.optimizer}</code> ·
    objective <code>{result.objective}</code> · seed {result.seed}</div>
  <div class="cards">{card_html}</div>

  <h2>Recommended configuration</h2>
  <pre>{best_json}</pre>

  <h2>Dataset</h2>
  <table><tbody>
    <tr><th>Name</th><td>{result.dataset.get('name')}</td></tr>
    <tr><th>Sensor type</th><td>{result.dataset.get('sensor_type')}</td></tr>
    <tr><th>Shape</th><td>{result.dataset.get('n_samples')} samples × {result.dataset.get('n_channels')} channels</td></tr>
    <tr><th>Sampling rate</th><td>{_fmt(result.dataset.get('sampling_frequency'))} Hz</td></tr>
    <tr><th>Fingerprint</th><td><code>{result.dataset.get('fingerprint')}</code></td></tr>
  </tbody></table>

  <h2>Search space</h2>
  <pre>{json.dumps(result.search_space, indent=2, sort_keys=True)}</pre>

  <h2>Encoder configuration</h2>
  <pre>{config_json}</pre>

  <h2>Top configurations</h2>
  <table>
    <thead><tr><th>Rank</th><th>Parameters</th><th>Accuracy</th><th>Spikes</th><th>Info loss</th><th>Cost</th><th></th></tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>

  <h2>Charts</h2>
  {figure_html or '<p class="subtle">Charts were not generated for this report.</p>'}

  <h2>Reproducibility</h2>
  <ul>
    <li>Dataset fingerprint: <code>{result.dataset.get('fingerprint')}</code></li>
    <li>Random seed: <code>{result.seed}</code></li>
    <li>Software: {', '.join(f'{key}={value}' for key, value in result.software.items())}</li>
  </ul>

  <h2>Limitations</h2>
  <ul>
    <li>Compute cost is an analytical proxy, not a hardware measurement.</li>
    <li>Scores are normalised within this experiment and are not comparable across experiments.</li>
    <li>Accuracy comes from a single group-level split; repeat with several seeds for variance.</li>
    <li>The default backend is a NumPy LIF readout; validate on the target framework before deployment.</li>
  </ul>
</body>
</html>
"""


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------
def generate_report(
    result: ExperimentResult,
    out_dir: str | Path,
    *,
    formats: Sequence[str] = ("json", "csv", "md", "html"),
    plots: bool = True,
    spike_train: Any | None = None,
) -> dict[str, Path]:
    """Generate every requested report artefact inside ``out_dir``.

    Returns a mapping of format name to the written path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    stem = result.experiment_id or "experiment"

    formats = tuple(fmt.lower() for fmt in formats)
    images: dict[str, Path] = {}

    if plots:
        from .plots import plot_objective_tradeoffs, plot_pareto, plot_spike_raster

        try:
            images["accuracy_vs_spikes"] = plot_pareto(result, out_dir / f"{stem}_pareto.png")
            images["objective_tradeoffs"] = plot_objective_tradeoffs(
                result, out_dir / f"{stem}_tradeoffs.png"
            )
        except Exception as exc:  # noqa: BLE001 - plotting must never fail a report
            images = {}
            result.extras["plot_error"] = redact_paths(f"{type(exc).__name__}: {exc}")
        if spike_train is not None:
            try:
                images["spike_raster"] = plot_spike_raster(
                    spike_train, out_dir / f"{stem}_raster.png"
                )
            except Exception as exc:  # noqa: BLE001
                result.extras["raster_error"] = redact_paths(f"{type(exc).__name__}: {exc}")

    for fmt in formats:
        if fmt == "json":
            path = out_dir / f"{stem}.json"
            export_result(result, path, fmt="json")
        elif fmt == "csv":
            path = out_dir / f"{stem}.csv"
            write_csv(trial_rows(result), path)
        elif fmt == "md":
            path = out_dir / f"{stem}.md"
            path.write_text(render_markdown(result), encoding="utf-8")
        elif fmt == "html":
            path = out_dir / f"{stem}.html"
            path.write_text(render_html(result, images), encoding="utf-8")
        else:
            raise ConfigurationError(f"Unsupported report format {fmt!r}.")
        written[fmt] = path
    for name, path in images.items():
        written[f"plot_{name}"] = path
    return written


