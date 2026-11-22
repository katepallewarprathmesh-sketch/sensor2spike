"""Experiment comparison helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..contracts import ExperimentResult

__all__ = ["compare_results", "render_comparison_markdown"]


def compare_results(results: Sequence[ExperimentResult]) -> dict[str, Any]:
    """Compare experiments on their recommended configuration.

    The first experiment is treated as the baseline; every later experiment is
    reported with deltas against it.
    """
    if not results:
        return {"experiments": [], "baseline_experiment_id": None}
    entries: list[dict[str, Any]] = []
    baseline = results[0].best
    for position, result in enumerate(results):
        best = result.best
        entry: dict[str, Any] = {
            "experiment_id": result.experiment_id,
            "encoder": result.encoder.get("encoder"),
            "optimizer": result.optimizer,
            "objective": result.objective,
            "n_trials": result.n_trials,
            "n_ok": sum(1 for trial in result.trials if trial.status == "ok"),
            "n_pareto": len(result.pareto_front),
            "duration_s": result.duration_s,
            "best": None,
            "delta_vs_baseline": None,
        }
        if best is not None:
            entry["best"] = {
                "index": best.index,
                "params": best.params,
                "accuracy": best.accuracy,
                "macro_f1": best.macro_f1,
                "spike_count": best.spike_count,
                "information_loss": best.information_loss,
                "compute_cost": best.compute_cost,
                "score": best.score,
            }
            if baseline is not None and position > 0:
                entry["delta_vs_baseline"] = {
                    "accuracy": best.accuracy - baseline.accuracy,
                    "spike_count": best.spike_count - baseline.spike_count,
                    "information_loss": best.information_loss - baseline.information_loss,
                    "compute_cost": best.compute_cost - baseline.compute_cost,
                }
        entries.append(entry)
    return {
        "baseline_experiment_id": results[0].experiment_id,
        "experiments": entries,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def render_comparison_markdown(results: Sequence[ExperimentResult]) -> str:
    """Render a Markdown comparison table."""
    comparison = compare_results(results)
    lines = ["# Experiment comparison", ""]
    lines.append("| Experiment | Encoder | Optimizer | Acc. | Spikes | Info loss | Cost | Δacc | Δspikes |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for entry in comparison["experiments"]:
        best = entry.get("best") or {}
        delta = entry.get("delta_vs_baseline") or {}
        lines.append(
            "| {experiment_id} | {encoder} | {optimizer} | {accuracy} | {spikes} | {loss} | {cost} | {dacc} | {dspikes} |".format(
                experiment_id=entry["experiment_id"],
                encoder=_fmt(entry["encoder"]),
                optimizer=_fmt(entry["optimizer"]),
                accuracy=_fmt(best.get("accuracy")),
                spikes=_fmt(best.get("spike_count")),
                loss=_fmt(best.get("information_loss")),
                cost=_fmt(best.get("compute_cost")),
                dacc=_fmt(delta.get("accuracy")),
                dspikes=_fmt(delta.get("spike_count")),
            )
        )
    return "\n".join(lines) + "\n"
