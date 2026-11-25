"""Compare encoders on temporal properties, not just accuracy.

Accuracy alone hides what an encoder does to time. Two encoders can both score
0.95 while one fires early and reliably and the other smears the same signal
across hundreds of redundant events — and on neuromorphic hardware those are
very different propositions.

This script measures, for every encoder on the same windows:

  latency_ms        mean time to the first spike, per window
  latency_spread    standard deviation of that first-spike time (precision)
  temporal_sparsity fraction of (time x neuron) slots that stay silent
  events_per_window mean synaptic events, the energy driver
  active_neurons    distinct neurons recruited at least once
  first_half_ratio  fraction of events in the first half of the window
  snn_accuracy      accuracy from the trained spiking backend
  info_preserved    1 - information_loss (linear reconstruction)

Run:  python examples/temporal_comparison.py
      python examples/temporal_comparison.py --out results/ --trials 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sensor2spike import build_backend, build_encoder, compute_spike_stats
from sensor2spike.demo import generate_demo_dataset

ENCODERS = ("rate", "latency", "threshold", "delta", "population")


def measure(trains, dt_seconds: float) -> dict[str, float]:
    """Temporal statistics for one encoder's output."""
    first_spike, spread, density, events, active, early = [], [], [], [], [], []

    for train in trains:
        matrix = np.asarray(train.spikes)
        steps, neurons = matrix.shape
        events.append(float(matrix.sum()))
        density.append(1.0 - float(matrix.mean()))
        active.append(float((matrix.sum(axis=0) > 0).sum()))

        # First spike time per window, over neurons that fired.
        fired = np.flatnonzero(matrix.any(axis=1))
        if fired.size:
            first_spike.append(float(fired[0] * dt_seconds * 1000.0))
            spread.append(float(np.std(fired) * dt_seconds * 1000.0))
        else:
            first_spike.append(float("nan"))
            spread.append(float("nan"))

        half = steps // 2
        total = matrix[:, :half].sum() + matrix[:, half:].sum()
        early.append(float(matrix[:, :half].sum()) / total if total else 0.5)

    return {
        "latency_ms": float(np.nanmean(first_spike)),
        "latency_spread": float(np.nanmean(spread)),
        "temporal_sparsity": float(np.mean(density)),
        "events_per_window": float(np.mean(events)),
        "active_neurons": float(np.mean(active)),
        "first_half_ratio": float(np.mean(early)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="write JSON results here")
    parser.add_argument("--dataset-trials", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--dt-ms", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    dataset = generate_demo_dataset(n_trials=args.dataset_trials, seed=args.seed)
    windows = dataset.to_windows(int(round(0.25 * dataset.sampling_frequency)))
    labels = windows.labels
    n = len(windows.samples)
    print(
        f"Dataset: {n} windows of {windows.samples[0].shape[0]} samples "
        f"x {windows.samples[0].shape[1]} channels, {args.dataset_trials} trials, seed {args.seed}"
    )
    print(f"Encoding at dt = {args.dt_ms} ms\n")

    rows = []
    for name in ENCODERS:
        encoder = build_encoder(name, dt_ms=args.dt_ms)
        encoder.fit(np.concatenate(windows.samples, axis=0), fs=dataset.sampling_frequency)
        trains = [
            encoder.transform(sample, fs=dataset.sampling_frequency)
            for sample in windows.samples
        ]

        temporal = measure(trains, trains[0].dt)

        backend = build_backend("numpy_sgd", hidden=32, epochs=args.epochs, lr=0.02, seed=0)
        evaluation = backend.train_evaluate(trains, labels, trains, labels, seed=0)

        losses = [encoder.information_loss(t, s) for t, s in zip(trains, windows.samples, strict=False)]
        losses = [x for x in losses if x is not None and np.isfinite(x)]
        stats = compute_spike_stats(trains)

        rows.append(
            {
                "encoder": name,
                **temporal,
                "snn_accuracy": float(evaluation.accuracy),
                "info_preserved": float(1.0 - (np.mean(losses) if losses else 0.0)),
                "spike_rate_hz": float(stats.spike_rate_hz),
            }
        )

    header = (
        f"{'encoder':10} {'lat_ms':>8} {'spread':>8} {'sparsity':>9} "
        f"{'events':>8} {'neurons':>8} {'early%':>7} {'SNN acc':>8} {'info':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['encoder']:10} {row['latency_ms']:8.2f} {row['latency_spread']:8.2f} "
            f"{100 * row['temporal_sparsity']:8.1f}% {row['events_per_window']:8.0f} "
            f"{row['active_neurons']:8.1f} {100 * row['first_half_ratio']:6.1f}% "
            f"{row['snn_accuracy']:8.3f} {row['info_preserved']:6.3f}"
        )

    print("\nHow to read this")
    print("  lat_ms      mean time to first spike -- lower means earlier evidence")
    print("  spread      std of spike times -- lower means more precise timing")
    print("  sparsity    fraction of silent slots -- higher means fewer events")
    print("  events      synaptic events per window -- the energy driver")
    print("  early%      share of events in the first half -- <50% means late coding")

    spread_lat = max(r["latency_ms"] for r in rows) - min(r["latency_ms"] for r in rows)
    spread_acc = max(r["snn_accuracy"] for r in rows) - min(r["snn_accuracy"] for r in rows)
    print(
        f"\nEncoding choice moved first-spike latency by {spread_lat:.1f} ms "
        f"and SNN accuracy by {100 * spread_acc:.1f} points."
    )

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        payload = {
            "settings": {
                "dt_ms": args.dt_ms,
                "epochs": args.epochs,
                "seed": args.seed,
                "dataset_trials": args.dataset_trials,
                "n_windows": n,
                "backend": "numpy_sgd",
            },
            "results": rows,
        }
        path = args.out / "temporal_comparison.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
