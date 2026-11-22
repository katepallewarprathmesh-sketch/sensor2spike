#!/usr/bin/env python3
"""Benchmark every encoder on a controlled, band-limited synthetic dataset.

Why band-limited: the demonstration task is radar-based human presence
detection, where the informative content is a micro-Doppler signature in a
narrow band (roughly 0.25-8 Hz) on top of sensor noise.  Wideband sensor noise
is not recoverable by *any* spike encoder at a fixed spike budget, so it only
adds variance to the comparison; the generator therefore applies a low-pass
characteristic to the interference while leaving the gait signature intact.

The script measures, for each encoder, the accuracy / spike-count /
information-loss trade-off found by random search, then prints a Pareto table.
All figures are produced by this run.  Nothing is asserted about real radar
hardware, and the number of trials is small enough that differences must be
read as illustrative rather than statistically strong.

Usage::

    python examples/benchmark.py --trials 40 --out benchmark
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sensor2spike import AutoTuner, ExperimentRegistry
from sensor2spike.contracts import SensorDataset

ENCODERS = ("rate", "latency", "threshold", "delta", "population")


def band_limited_radar(
    *,
    n_trials: int = 60,
    seed: int = 7,
    fs: float = 1000.0,
    duration: float = 1.0,
    n_channels: int = 4,
    noise: float = 0.1,
    cutoff_hz: float = 25.0,
) -> SensorDataset:
    """Synthetic airborne-radar recording with a band-limited noise floor.

    Each trial is either an empty room (clutter only) or an occupied room (gait
    micro-Doppler with breathing modulation plus clutter).  Interference is
    low-pass filtered so that the recording respects the bandwidth a real
    presence sensor would digitise.
    """
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(seed)
    sos = butter(4, cutoff_hz / (fs / 2.0), btype="low", output="sos")
    n_samples = int(round(duration * fs))
    t = np.arange(n_samples) / fs

    blocks, labels, groups = [], [], []
    for trial in range(n_trials):
        label = trial % 2
        noise_floor = sosfiltfilt(sos, rng.standard_normal((n_samples, n_channels)), axis=0)
        noise_floor *= noise / max(1e-9, noise_floor.std())

        signal = noise_floor.copy()
        for channel in range(n_channels):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            # Simple breathing/clutter baseline present in both classes.
            signal[:, channel] += 0.10 * np.sin(2.0 * np.pi * 0.3 * t + phase)
            if label == 1:
                gait = rng.uniform(1.7, 2.5)
                sway = rng.uniform(0.45, 0.7)
                envelope = 1.0 + 0.35 * np.sin(2.0 * np.pi * sway * t + 0.4 * phase)
                signal[:, channel] += envelope * (
                    np.sin(2.0 * np.pi * gait * t + phase)
                    + 0.45 * np.sin(2.0 * np.pi * 2.0 * gait * t + 1.3 * phase)
                    + 0.2 * np.sin(2.0 * np.pi * 3.0 * gait * t + 0.7 * phase + channel)
                )

        peak = float(np.max(np.abs(signal)))
        if peak > 0:
            signal /= peak
        blocks.append(signal.astype(np.float32))
        labels.append(np.full(n_samples, label, dtype=np.int64))
        groups.append(np.full(n_samples, trial, dtype=np.int64))

    return SensorDataset(
        data=np.concatenate(blocks),
        sampling_frequency=fs,
        channel_names=tuple(f"range_bin_{i}" for i in range(n_channels)),
        labels=np.concatenate(labels),
        groups=np.concatenate(groups),
        name="bench_radar_presence",
        sensor_type="radar",
        metadata={
            "synthetic": True,
            "band_limited_hz": cutoff_hz,
            "noise": noise,
            "description": "Band-limited synthetic radar presence-detection benchmark.",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="benchmark", help="Output directory")
    parser.add_argument("--trials", type=int, default=30, help="Search trials per encoder")
    parser.add_argument("--dataset-trials", type=int, default=60)
    parser.add_argument("--window", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--optimizer", default="random", choices=("random", "grid", "bayes"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dataset = band_limited_radar(n_trials=args.dataset_trials, seed=args.seed)
    print("Benchmark dataset")
    print(f"  {dataset.n_samples:,} samples x {dataset.n_channels} channels, "
          f"{dataset.n_groups} trials, {dataset.sampling_frequency:.0f} Hz")
    print(f"  classes: {np.bincount(dataset.labels.astype(int)).tolist()} samples per class")
    print(f"  fingerprint: {dataset.fingerprint()}")
    print()

    registry = ExperimentRegistry(out / "benchmark.db")
    rows: list[dict[str, object]] = []

    header = (
        f"{'encoder':<12}{'best acc':>9}{'F1':>8}{'spikes':>9}"
        f"{'info loss':>11}{'feasible':>9}{'pareto':>7}{'trials':>7}{'secs':>7}"
    )
    print(header)
    print("-" * len(header))

    for encoder in ENCODERS:
        t0 = time.perf_counter()
        result = AutoTuner(
            objective="multi_objective",
            encoder=encoder,
            optimizer=args.optimizer,
            n_trials=args.trials,
            seed=args.seed,
            window_seconds=args.window,
        ).optimize(dataset)
        elapsed = time.perf_counter() - t0
        best = result.best
        if best is None:  # pragma: no cover - defensive
            continue
        n_ok = sum(1 for trial in result.trials if trial.status == "ok")
        row = {
            "encoder": encoder,
            "experiment_id": result.experiment_id,
            "accuracy": best.accuracy,
            "macro_f1": best.macro_f1,
            "spike_count": best.spike_count,
            "information_loss": best.information_loss,
            "compute_cost": best.compute_cost,
            "params": best.params,
            "n_trials": len(result.trials),
            "n_ok": n_ok,
            "n_feasible": len(result.feasible_trials),
            "pareto_front": len(result.pareto_front),
            "duration_s": elapsed,
        }
        rows.append(row)
        print(
            f"{encoder:<12}{best.accuracy:>9.4f}{best.macro_f1:>8.4f}"
            f"{best.spike_count:>9}{best.information_loss:>11.4f}"
            f"{len(result.feasible_trials):>9}{len(result.pareto_front):>7}"
            f"{len(result.trials):>7}{elapsed:>7.1f}"
        )

    # ---------------------------------------------------------------- best overall
    if rows:
        ordered = sorted(rows, key=lambda r: (-float(r["accuracy"]), int(r["spike_count"])))
        print()
        print("Ranking by measured accuracy (ties broken by fewer spikes)")
        for position, row in enumerate(ordered, start=1):
            print(
                f"  {position}. {row['encoder']:<12} acc={row['accuracy']:.4f} "
                f"spikes={row['spike_count']:>7} info_loss={row['information_loss']:.4f} "
                f"params={row['params']}"
            )

        cheapest = min(rows, key=lambda r: int(r["spike_count"]))
        most_accurate = max(rows, key=lambda r: float(r["accuracy"]))
        print()
        print("Summary")
        print(f"  most accurate : {most_accurate['encoder']} "
              f"({most_accurate['accuracy']:.4f} at {most_accurate['spike_count']} spikes)")
        print(f"  sparsest      : {cheapest['encoder']} "
              f"({cheapest['spike_count']} spikes at {cheapest['accuracy']:.4f})")

    payload = {
        "dataset": dataset.summary(),
        "settings": {
            "trials_per_encoder": args.trials,
            "window_seconds": args.window,
            "seed": args.seed,
            "optimizer": args.optimizer,
        },
        "results": rows,
        "registered_experiments": registry.experiment_count(),
        "caveat": (
            "Single synthetic band-limited dataset, small trial budget. Figures are "
            "indicative of the optimisation loop, not a claim about real sensor hardware."
        ),
    }
    path = out / "benchmark_results.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Results written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
