#!/usr/bin/env python3
"""Run the full sensor2spike demo on a synthetic radar dataset.

End-to-end walkthrough of the developer workflow described in the project plan:

1. generate a reproducible radar presence-detection dataset (two classes);
2. compare every built-in encoder with a fixed evaluation budget;
3. run a deeper multi-objective search for the strongest encoder;
4. write reports (JSON/CSV/Markdown/HTML + Pareto plots) and an exportable config.

Every number printed comes from an actual measurement in this run.  Nothing is
hard-coded and no performance claim is made beyond what was observed.

Usage::

    python examples/run_demo.py                # ~20 s
    python examples/run_demo.py --trials 60    # deeper search
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sensor2spike import (
    AutoTuner,
    ExperimentRegistry,
    generate_report,
    load_dataset,
    save_dataset,
)
from sensor2spike.demo import demo_summary, generate_demo_dataset

ENCODERS = ("rate", "latency", "threshold", "delta", "population")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="demo_output", help="Output directory")
    parser.add_argument("--trials", type=int, default=20, help="Trials per encoder in the sweep")
    parser.add_argument("--final-trials", type=int, default=40, help="Trials for the final search")
    parser.add_argument("--dataset-trials", type=int, default=60, help="Synthetic trials in the dataset")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--window", type=float, default=0.25, help="Window length in seconds")
    parser.add_argument("--reuse", action="store_true", help="Reuse an existing dataset file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dataset_path = out / "demo_radar.npz"
    if args.reuse and dataset_path.exists():
        dataset = load_dataset(dataset_path)
        print(f"Reusing dataset: {dataset_path}")
    else:
        dataset = generate_demo_dataset(
            n_trials=args.dataset_trials, seed=args.seed, duration_seconds=1.0
        )
        save_dataset(dataset, dataset_path)
        print(f"Wrote dataset: {dataset_path}")

    summary = demo_summary(dataset)
    print()
    print("Dataset")
    print(f"  name        : {summary['name']}")
    print(f"  samples     : {summary['n_samples']:,} x {summary['n_channels']} channels")
    print(f"  trials      : {summary['n_trials']}")
    print(f"  classes     : {dict(zip(summary['class_names'], summary['class_counts'].values(), strict=False))}")
    print(f"  fingerprint : {summary['fingerprint']}")

    registry = ExperimentRegistry(out / "experiments.db")

    # ---------------------------------------------------------------- encoder sweep
    print()
    print(f"Encoder sweep ({args.trials} trials each, window={args.window}s)")
    header = f"{'encoder':<12}{'best acc':>9}{'spikes':>9}{'info loss':>11}{'cost':>11}{'secs':>7}"
    print(header)
    print("-" * len(header))
    sweep: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    for encoder in ENCODERS:
        t0 = time.perf_counter()
        result = AutoTuner(
            objective="multi_objective",
            encoder=encoder,
            n_trials=args.trials,
            seed=args.seed,
            window_seconds=args.window,
        ).optimize(dataset)
        elapsed = time.perf_counter() - t0
        best = result.best
        if best is None:  # pragma: no cover - defensive
            print(f"{encoder:<12}{'failed':>9}")
            continue
        sweep[encoder] = {
            "accuracy": best.accuracy,
            "spike_count": best.spike_count,
            "information_loss": best.information_loss,
            "compute_cost": best.compute_cost,
            "duration_s": elapsed,
            "experiment_id": result.experiment_id,
        }
        print(
            f"{encoder:<12}{best.accuracy:>9.4f}{best.spike_count:>9}"
            f"{best.information_loss:>11.4f}{best.compute_cost:>11.1f}{elapsed:>7.1f}"
        )
    print(f"sweep wall-clock: {time.perf_counter() - started:.1f}s")

    if not sweep:  # pragma: no cover - defensive
        print("No encoder produced a result; aborting.")
        return 1

    # ------------------------------------------------------------ final deep search
    leader = max(sweep, key=lambda name: (sweep[name]["accuracy"], -sweep[name]["spike_count"]))
    print()
    print(f"Final multi-objective search on the leading encoder: {leader!r}")
    final = AutoTuner(
        objective="multi_objective",
        encoder=leader,
        n_trials=args.final_trials,
        seed=args.seed,
        window_seconds=args.window,
    ).optimize(dataset)

    print(f"  experiment  : {final.experiment_id}")
    print(f"  trials ok   : {sum(1 for t in final.trials if t.status == 'ok')}/{len(final.trials)}")
    print(f"  pareto front: {len(final.pareto_front)}")
    print(f"  best params : {final.best.params}")
    print(f"  accuracy    : {final.best.accuracy:.4f}")
    print(f"  spikes      : {final.best.spike_count}")
    print(f"  info loss   : {final.best.information_loss:.4f}")

    # --------------------------------------------------------------------- reports
    reports_dir = out / "reports"
    written = final.export(reports_dir / f"{final.experiment_id}.json").parent
    generate_report(final, reports_dir)
    config_path = final.export_config(reports_dir / "best_config.json")
    print()
    print(f"Reports : {written}")
    for path in sorted(reports_dir.iterdir()):
        print(f"  {path.name}")
    print(f"Config  : {config_path}")

    # ------------------------------------------------------------------- summary
    payload = {
        "dataset": summary,
        "settings": {
            "trials_per_encoder": args.trials,
            "final_trials": args.final_trials,
            "window_seconds": args.window,
            "seed": args.seed,
        },
        "encoder_sweep": sweep,
        "final": {
            "encoder": leader,
            "experiment_id": final.experiment_id,
            "params": final.best.params,
            "accuracy": final.best.accuracy,
            "macro_f1": final.best.macro_f1,
            "spike_count": final.best.spike_count,
            "information_loss": final.best.information_loss,
            "compute_cost": final.best.compute_cost,
            "n_trials": final.n_trials,
            "pareto_front_size": len(final.pareto_front),
        },
        "registered_experiments": registry.experiment_count(),
    }
    summary_path = out / "benchmark_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Summary : {summary_path}")
    print()
    print("Note: accuracies come from a NumPy LIF + ridge readout on a synthetic")
    print("dataset. They illustrate the optimisation loop, not absolute performance")
    print("on real radar hardware.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
