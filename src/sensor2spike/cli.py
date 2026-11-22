"""Command-line interface.

The CLI is intentionally thin: every command maps onto the Python API, so the
two surfaces cannot drift apart.  Use ``sensor2spike --help`` for the full
command reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import (
    ExperimentRegistry,
    ExperimentRunner,
    OptimizeConfig,
    Sensor2SpikeError,
    SensorDataset,
    __version__,
    build_encoder,
    compare_results,
    export_config,
    generate_report,
    list_backends,
    list_encoders,
    load_dataset,
    render_comparison_markdown,
    save_dataset,
)
from ._core.data.io import save_spike_train
from ._core.utils.security import redact_paths

__all__ = ["main"]


# --------------------------------------------------------------------------------------
# Console helpers
# --------------------------------------------------------------------------------------
def _print(text: str = "") -> None:
    print(text)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], *, width: int = 14) -> None:
    """Render a simple fixed-width table (no external dependency)."""
    headers = list(headers)
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(str(cell)) for cell in [headers[i], *[row[i] for row in rows]]) if rows else len(headers[i]) for i in range(len(headers))]
    widths = [min(max(w, len(h)), width) for w, h in zip(widths, headers, strict=False)]
    line = "  ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers)))
    _print(line)
    _print("-" * len(line))
    for row in rows:
        cells = []
        for index, cell in enumerate(row):
            text = str(cell)
            if len(text) > widths[index]:
                text = text[: max(0, widths[index] - 1)] + "…"
            cells.append(text.ljust(widths[index]))
        _print("  ".join(cells))


def _error(message: str) -> None:
    print(f"error: {redact_paths(message)}", file=sys.stderr)


def _load_dataset(args: argparse.Namespace) -> SensorDataset:
    kwargs: dict[str, Any] = {}
    if getattr(args, "time_column", None):
        kwargs["time_column"] = args.time_column
    if getattr(args, "label_column", None):
        kwargs["label_column"] = args.label_column
    if getattr(args, "group_column", None):
        kwargs["group_column"] = args.group_column
    if getattr(args, "channel_columns", None):
        kwargs["channel_columns"] = args.channel_columns
    if getattr(args, "sampling_frequency", None):
        kwargs["sampling_frequency"] = args.sampling_frequency
    if getattr(args, "nan_policy", None):
        kwargs["nan_policy"] = args.nan_policy
    if getattr(args, "name", None):
        kwargs["name"] = args.name
    if getattr(args, "sensor_type", None):
        kwargs["sensor_type"] = args.sensor_type
    return load_dataset(args.data, **kwargs)


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------
def cmd_encoders(_: argparse.Namespace) -> int:
    _print("Available encoders")
    _print()
    rows = [[item["name"], item["description"], ", ".join(item["parameters"])] for item in list_encoders()]
    _table(["encoder", "description", "parameters"], rows, width=48)
    _print()
    _print("SNN backends")
    _print()
    _table(["backend", "description"], [[item["name"], item["description"]] for item in list_backends()], width=64)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args)
    _print("Dataset validation: OK")
    _print()
    _print(dataset.describe())
    return 0


def cmd_encode(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args)
    params = _parse_params(args.param)
    encoder = build_encoder(args.method, dt_ms=args.dt_ms, seed=args.seed, **params)
    spikes = encoder.fit_transform(dataset.data, fs=dataset.sampling_frequency)
    out = Path(args.out) if args.out else Path(f"{dataset.name}_{args.method}_spikes.npz")
    save_spike_train(spikes, out)
    _print(f"Encoded {dataset.n_samples} samples -> {spikes.n_steps} steps x {spikes.n_neurons} neurons")
    _print(f"Spike count: {spikes.spike_count} ({spikes.spike_rate_hz:.1f} spikes/s)")
    _print(f"Temporal sparsity: {spikes.temporal_sparsity:.4f}")
    if args.plot:
        from ._core.reporting.plots import plot_spike_raster

        path = Path(out).with_suffix(".png")
        plot_spike_raster(spikes, path, title=f"{args.method} encoding — {dataset.name}")
        _print(f"Raster plot: {path.name}")
    _print(f"Saved: {out.name}")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args)
    config = OptimizeConfig(
        encoder=args.encoder,
        encoder_params=_parse_params(args.set),
        optimizer=args.optimizer,
        n_trials=args.trials,
        objective=args.objective,
        accuracy_target=args.target,
        spike_budget=args.budget,
        seed=args.seed,
        backend=args.backend,
        window_seconds=args.window,
        test_fraction=args.test_fraction,
        selection=args.selection,
        notes=args.notes or "",
    )
    runner = ExperimentRunner(args.registry)
    result = runner.run(dataset, config)

    _print()
    _print(f"Experiment: {result.experiment_id}")
    _print(f"Status: {result.status}  trials={result.n_trials}  duration={result.duration_s:.2f}s")
    best = result.best
    if best is not None:
        _print()
        _print("Recommended configuration")
        _print(f"  params      : {best.params}")
        _print(f"  accuracy    : {best.accuracy:.4f}")
        _print(f"  macro F1    : {best.macro_f1:.4f}")
        _print(f"  spike count : {best.spike_count}")
        _print(f"  info loss   : {best.information_loss:.4f}")
        _print(f"  compute cost: {best.compute_cost:.4g}")
    _print()
    rows = [
        [
            trial.rank or "-",
            _params_inline(trial.params),
            f"{trial.accuracy:.4f}",
            trial.spike_count,
            f"{trial.information_loss:.4f}",
            "yes" if trial.is_pareto else "",
        ]
        for trial in result.top(10)
    ]
    _table(["rank", "params", "accuracy", "spikes", "info loss", "pareto"], rows, width=40)

    out_dir = Path(args.out) if args.out else Path("results")
    written = generate_report(result, out_dir, formats=("json", "csv", "md", "html"), plots=not args.no_plots)
    _print()
    _print(f"Reports written to {out_dir}/")
    for fmt, path in written.items():
        _print(f"  {fmt:24s} {path.name}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    registry = ExperimentRegistry(args.registry)
    result = registry.load_experiment(args.experiment)
    out_dir = Path(args.out) if args.out else Path("reports")
    formats = tuple(fmt.strip() for fmt in args.formats.split(",") if fmt.strip())
    written = generate_report(result, out_dir, formats=formats, plots=not args.no_plots)
    _print(f"Reports for {result.experiment_id} written to {out_dir}/")
    for fmt, path in written.items():
        _print(f"  {fmt:24s} {path.name}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    registry = ExperimentRegistry(args.registry)
    experiments = registry.list_experiments(limit=args.limit)
    if not experiments:
        _print("No experiments recorded yet.")
        return 0
    rows = [
        [
            item["experiment_id"],
            item["optimizer"],
            item["objective"],
            item["n_trials"],
            item["status"],
            "-" if item["best_accuracy"] is None else f"{item['best_accuracy']:.4f}",
            "-" if item["best_spike_count"] is None else item["best_spike_count"],
        ]
        for item in experiments
    ]
    _table(["experiment", "optimizer", "objective", "trials", "status", "best acc", "spikes"], rows, width=22)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    registry = ExperimentRegistry(args.registry)
    results = []
    for experiment_id in args.experiments:
        results.append(registry.load_experiment(experiment_id))
    comparison = compare_results(results)
    _print(render_comparison_markdown(results))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_comparison_markdown(results), encoding="utf-8")
        _print(f"Comparison written to {path}")
    _ = json.dumps(comparison, indent=2, default=str)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    registry = ExperimentRegistry(args.registry)
    result = registry.load_experiment(args.experiment)
    out = Path(args.out) if args.out else Path("optimized_config.json")
    export_config(result, out)
    _print(f"Exported recommended configuration to {out}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import demo_summary, generate_demo_dataset

    dataset = generate_demo_dataset(
        n_trials=args.trials,
        seed=args.seed,
        noise=args.noise,
    )
    out = Path(args.out) if args.out else Path("demo_radar.npz")
    save_dataset(dataset, out)
    if args.csv:
        csv_path = Path(out).with_suffix(".csv")
        save_dataset(dataset, csv_path)
        _print(f"CSV dataset: {csv_path.name}")
    _print(f"Demo dataset: {out.name}")
    _print(dataset.describe())
    summary = demo_summary(dataset)
    named = ", ".join(
        f"{summary['class_names'][label]}={count}"
        for label, count in sorted(summary["class_counts"].items())
    )
    _print(f"class balance    : {named}")
    _print()
    _print("Next:")
    _print(f"  sensor2spike optimize {out.name} --encoder rate --trials 20 --window 0.5")
    return 0


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------
def _params_inline(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items())) or "(defaults)"


def _parse_params(values: Sequence[str] | None) -> dict[str, Any]:
    """Parse ``key=value`` pairs, coercing numbers and booleans."""
    params: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise Sensor2SpikeError(f"Invalid parameter {item!r}; expected key=value.")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        try:
            params[key] = int(raw)
        except ValueError:
            try:
                params[key] = float(raw)
            except ValueError:
                if raw.lower() in {"true", "false"}:
                    params[key] = raw.lower() == "true"
                else:
                    params[key] = raw
    return params


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("data", help="Dataset file (.csv, .npz or .npy)")
    parser.add_argument("--time-column", help="Column with numeric seconds or ISO-8601 timestamps")
    parser.add_argument("--label-column", help="Column with integer class labels")
    parser.add_argument("--group-column", help="Column with trial/segment identifiers")
    parser.add_argument(
        "--channel-columns",
        help="Comma-separated channel columns (default: all unused columns)",
    )
    parser.add_argument("--sampling-frequency", type=float, help="Sample rate in Hz (CSV only)")
    parser.add_argument("--nan-policy", choices=("raise", "interpolate", "zero", "drop"), help="Missing-value policy")
    parser.add_argument("--name", help="Dataset name used in reports")
    parser.add_argument("--sensor-type", help="Sensor type metadata")


def _add_registry_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", default="sensor2spike.db", help="SQLite registry path (default: sensor2spike.db)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sensor2spike",
        description="Sensor-to-spike encoding optimisation for spiking neural networks.",
    )
    parser.add_argument("--version", action="version", version=f"sensor2spike {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encoders = subparsers.add_parser("encoders", help="List available encoders and backends")
    encoders.set_defaults(handler=cmd_encoders)

    validate = subparsers.add_parser("validate", help="Validate a dataset and print its metadata")
    _add_dataset_arguments(validate)
    validate.set_defaults(handler=cmd_validate)

    encode = subparsers.add_parser("encode", help="Encode a dataset into spikes")
    _add_dataset_arguments(encode)
    encode.add_argument("--method", default="rate", help="Encoder name (default: rate)")
    encode.add_argument("--dt-ms", type=float, default=1.0, help="Spike bin width in ms")
    encode.add_argument("--seed", type=int, default=0, help="Random seed")
    encode.add_argument("--param", action="append", help="Encoder parameter as key=value (repeatable)")
    encode.add_argument("--out", help="Output .npz path")
    encode.add_argument("--plot", action="store_true", help="Also write a raster plot")
    encode.set_defaults(handler=cmd_encode)

    optimize = subparsers.add_parser("optimize", help="Search for the best encoder configuration")
    _add_dataset_arguments(optimize)
    _add_registry_argument(optimize)
    optimize.add_argument("--encoder", default="rate", help="Encoder to optimise")
    optimize.add_argument("--optimizer", default="random", choices=("grid", "random", "bayes", "bayesian"))
    optimize.add_argument("--objective", default="multi", choices=("multi", "multi_objective", "weighted"))
    optimize.add_argument("--trials", type=int, default=20, help="Number of candidate configurations")
    optimize.add_argument("--target", type=float, help="Minimum accuracy constraint")
    optimize.add_argument("--budget", type=int, help="Maximum spike-count constraint")
    optimize.add_argument("--window", type=float, help="Window length in seconds")
    optimize.add_argument("--test-fraction", type=float, default=0.3)
    optimize.add_argument("--selection", default=None, choices=("balanced", "accuracy", "spikes", "score"), help="Best-configuration rule (default: balanced for multi, score for weighted)")
    optimize.add_argument("--backend", default="lif_ridge", help="SNN backend name")
    optimize.add_argument("--seed", type=int, default=0)
    optimize.add_argument("--set", action="append", help="Fixed encoder parameter as key=value (repeatable)")
    optimize.add_argument("--out", help="Output directory (default: results/)")
    optimize.add_argument("--no-plots", action="store_true", help="Skip chart generation")
    optimize.add_argument("--notes", help="Free-text note stored with the experiment")
    optimize.set_defaults(handler=cmd_optimize)

    report = subparsers.add_parser("report", help="Regenerate reports for a stored experiment")
    _add_registry_argument(report)
    report.add_argument("--experiment", required=True, help="Experiment identifier")
    report.add_argument("--out", help="Output directory (default: reports/)")
    report.add_argument("--formats", default="json,csv,md,html", help="Comma-separated formats")
    report.add_argument("--no-plots", action="store_true")
    report.set_defaults(handler=cmd_report)

    list_cmd = subparsers.add_parser("list", help="List recorded experiments")
    _add_registry_argument(list_cmd)
    list_cmd.add_argument("--limit", type=int, default=50)
    list_cmd.set_defaults(handler=cmd_list)

    compare = subparsers.add_parser("compare", help="Compare stored experiments")
    _add_registry_argument(compare)
    compare.add_argument("experiments", nargs="+", help="Experiment identifiers (first = baseline)")
    compare.add_argument("--out", help="Write the comparison to a Markdown file")
    compare.set_defaults(handler=cmd_compare)

    export = subparsers.add_parser("export", help="Export the recommended configuration")
    _add_registry_argument(export)
    export.add_argument("--experiment", required=True)
    export.add_argument("--out", help="Output JSON path")
    export.set_defaults(handler=cmd_export)

    demo = subparsers.add_parser("demo", help="Generate a synthetic radar dataset")
    demo.add_argument("--out", help="Output .npz path (default: demo_radar.npz)")
    demo.add_argument("--csv", action="store_true", help="Also write a .csv copy")
    demo.add_argument("--trials", type=int, default=120, help="Number of trials")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--noise", type=float, default=0.15, help="Noise level")
    demo.set_defaults(handler=cmd_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Sensor2SpikeError as exc:
        _error(f"{type(exc).__name__}: {exc}")
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        _error("Interrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _error(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
