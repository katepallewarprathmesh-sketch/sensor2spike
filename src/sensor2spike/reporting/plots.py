"""Matplotlib figures (optional dependency, imported lazily)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..contracts import ExperimentResult, SpikeTrain
from ..exceptions import DependencyMissingError

__all__ = [
    "plot_pareto",
    "plot_spike_raster",
    "plot_objective_tradeoffs",
    "save_plot",
]


def _pyplot():
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DependencyMissingError(
            "matplotlib", extra="viz", purpose="chart generation"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_plot(figure: Any, path: str | Path, *, dpi: int = 150) -> Path:
    """Save and close a matplotlib figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path


def plot_pareto(result: ExperimentResult, path: str | Path) -> Path:
    """Scatter accuracy against spike activity with the Pareto front highlighted."""
    plt = _pyplot()
    trials = [trial for trial in result.trials if trial.status == "ok"]
    figure, axes = plt.subplots(figsize=(7.5, 5.0))
    axes.scatter(
        [trial.spike_count for trial in trials],
        [trial.accuracy for trial in trials],
        alpha=0.45,
        color="#4C78A8",
        label="Candidate configuration",
    )
    front = sorted(result.pareto_front, key=lambda trial: trial.spike_count)
    if front:
        axes.plot(
            [trial.spike_count for trial in front],
            [trial.accuracy for trial in front],
            "o-",
            color="#E45756",
            linewidth=1.6,
            markersize=5,
            label="Pareto front",
        )
    best = result.best
    if best is not None:
        axes.scatter(
            [best.spike_count],
            [best.accuracy],
            s=140,
            facecolors="none",
            edgecolors="#2CA02C",
            linewidths=2,
            label="Recommended",
        )
    if result.spike_budget:
        axes.axvline(result.spike_budget, linestyle="--", color="gray", linewidth=1, label="Spike budget")
    if result.accuracy_target is not None:
        axes.axhline(result.accuracy_target, linestyle=":", color="#4C78A8", linewidth=1, label="Accuracy target")
    axes.set_xlabel("Spike count (training windows)")
    axes.set_ylabel("Accuracy")
    axes.set_title("Accuracy vs. spike activity")
    axes.grid(alpha=0.25)
    axes.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return save_plot(figure, path)


def plot_objective_tradeoffs(result: ExperimentResult, path: str | Path) -> Path:
    """Three-panel view of the objective space."""
    plt = _pyplot()
    trials = [trial for trial in result.trials if trial.status == "ok" and trial.objectives]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    panels = [
        ("spikes", "information_loss", "Spike activity", "Information loss"),
        ("spikes", "compute_cost", "Spike activity", "Compute cost"),
        ("information_loss", "compute_cost", "Information loss", "Compute cost"),
    ]
    for axis, (x_key, y_key, x_label, y_label) in zip(axes, panels, strict=False):
        axis.scatter(
            [trial.objectives[x_key] for trial in trials],
            [trial.objectives[y_key] for trial in trials],
            c=[trial.accuracy for trial in trials],
            cmap="viridis",
            alpha=0.75,
        )
        front = [trial for trial in result.pareto_front if trial.objectives]
        axis.scatter(
            [trial.objectives[x_key] for trial in front],
            [trial.objectives[y_key] for trial in front],
            facecolors="none",
            edgecolors="#E45756",
            linewidths=1.4,
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.25)
    figure.suptitle("Objective trade-offs (colour = accuracy, red ring = Pareto front)")
    figure.tight_layout()
    return save_plot(figure, path)


def plot_spike_raster(
    spikes: SpikeTrain | Sequence[SpikeTrain],
    path: str | Path,
    *,
    max_neurons: int = 32,
    title: str | None = None,
) -> Path:
    """Raster plot of one or more spike trains."""
    plt = _pyplot()
    trains: list[SpikeTrain] = [spikes] if isinstance(spikes, SpikeTrain) else list(spikes)
    if not trains:
        raise ValueError("plot_spike_raster requires at least one spike train.")
    figure, axes = plt.subplots(
        len(trains), 1, figsize=(8.0, 2.2 * len(trains)), sharex=False, squeeze=False
    )
    for axis, train in zip(axes.ravel(), trains, strict=False):
        matrix = train.spikes[:, : max_neurons].astype(float)
        axis.imshow(
            matrix.T,
            aspect="auto",
            interpolation="nearest",
            cmap="Grays",
            extent=[0, train.n_steps * train.dt * 1000.0, matrix.shape[1], 0],
        )
        axis.set_ylabel("Neuron")
        axis.set_title(f"{train.encoder or 'encoder'} — {train.spike_count} spikes, {train.spike_rate_hz:.1f} spikes/s")
    axes.ravel()[-1].set_xlabel("Time (ms)")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return save_plot(figure, path)


def _unused(value: Iterable[Any]) -> None:  # pragma: no cover
    for _ in value:
        pass


