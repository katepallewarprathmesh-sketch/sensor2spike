"""Objective functions, Pareto ranking and best-configuration selection.

The optimisation problem (specification section 11.1) is::

    J = w_a (1 - accuracy) + w_s S + w_l L + w_c C

where ``S`` is normalised spike activity, ``L`` information loss and ``C``
computational cost.  ``score = -J`` so that higher is better.  When the caller
selects ``objective="multi"`` the same objectives are ranked with non-dominated
sorting instead of being collapsed into a single number, because a weighted sum
can hide trade-offs (section 11.2).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..contracts import TrialResult
from ..exceptions import ConfigurationError

__all__ = [
    "ObjectiveNormalizers",
    "ObjectiveWeights",
    "PARETO_KEYS",
    "compute_objectives",
    "crowding_distance",
    "dominates",
    "non_dominated_sort",
    "rank_trials",
    "select_best",
    "summarize_trials",
]

#: Objective names in the order used for Pareto dominance.
PARETO_KEYS = ("accuracy", "spikes", "information_loss", "compute_cost")


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for the weighted-sum objective.  All weights must be >= 0."""

    accuracy: float = 1.0
    spikes: float = 1.0
    information: float = 1.0
    compute: float = 1.0

    def __post_init__(self) -> None:
        for name in ("accuracy", "spikes", "information", "compute"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ConfigurationError(f"ObjectiveWeights.{name} must be a finite value >= 0.")
        if sum(float(getattr(self, name)) for name in ("accuracy", "spikes", "information", "compute")) <= 0:
            raise ConfigurationError("At least one objective weight must be positive.")

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": float(self.accuracy),
            "spikes": float(self.spikes),
            "information": float(self.information),
            "compute": float(self.compute),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ObjectiveWeights:
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ConfigurationError("ObjectiveWeights.from_dict expects a mapping.")
        allowed = {"accuracy", "spikes", "information", "compute"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ConfigurationError(
                f"Unknown objective weight(s): {', '.join(unknown)}. "
                f"Expected: {', '.join(sorted(allowed))}."
            )
        return cls(
            accuracy=float(payload.get("accuracy", 1.0)),
            spikes=float(payload.get("spikes", 1.0)),
            information=float(payload.get("information", 1.0)),
            compute=float(payload.get("compute", 1.0)),
        )


@dataclass(frozen=True)
class ObjectiveNormalizers:
    """Reference values that map raw metrics onto a comparable scale."""

    spike_count: float = 1.0
    compute_cost: float = 1.0
    information_loss: float = 1.0

    def __post_init__(self) -> None:
        for name in ("spike_count", "compute_cost", "information_loss"):
            if float(getattr(self, name)) <= 0:
                raise ConfigurationError(f"ObjectiveNormalizers.{name} must be positive.")

    @classmethod
    def fit(cls, trials: Sequence[TrialResult]) -> ObjectiveNormalizers:
        """Use the maximum of each raw metric across ``trials`` as reference."""
        usable = [trial for trial in trials if trial.status == "ok"]
        if not usable:
            return cls()
        return cls(
            spike_count=max(1.0, max(trial.spike_count for trial in usable)),
            compute_cost=max(1e-9, max(trial.compute_cost for trial in usable)),
            information_loss=max(1e-9, max(trial.information_loss for trial in usable)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "spike_count": float(self.spike_count),
            "compute_cost": float(self.compute_cost),
            "information_loss": float(self.information_loss),
        }


# --------------------------------------------------------------------------------------
# Objectives
# --------------------------------------------------------------------------------------
def compute_objectives(
    trial: TrialResult,
    normalizers: ObjectiveNormalizers,
    weights: ObjectiveWeights,
    *,
    accuracy_target: float | None = None,
    spike_budget: int | None = None,
) -> TrialResult:
    """Fill ``trial.objectives``, ``trial.score`` and ``trial.feasible`` in place."""
    spikes = min(trial.spike_count / normalizers.spike_count, 1.0)
    information = min(trial.information_loss / normalizers.information_loss, 1.0)
    compute = min(trial.compute_cost / normalizers.compute_cost, 1.0)
    accuracy = float(trial.accuracy)

    trial.objectives = {
        "accuracy": accuracy,
        "spikes": spikes,
        "information_loss": information,
        "compute_cost": compute,
        "1-accuracy": 1.0 - accuracy,
    }
    penalty = (
        weights.accuracy * (1.0 - accuracy)
        + weights.spikes * spikes
        + weights.information * information
        + weights.compute * compute
    )
    trial.score = float(-penalty)
    trial.feasible = bool(
        (accuracy_target is None or accuracy >= accuracy_target)
        and (spike_budget is None or trial.spike_count <= spike_budget)
    )
    return trial


def pareto_vector(trial: TrialResult) -> np.ndarray:
    """Objectives to *minimise*: 1-accuracy, spikes, information loss, cost."""
    return np.array(
        [
            1.0 - trial.accuracy,
            trial.objectives.get("spikes", trial.spike_count),
            trial.objectives.get("information_loss", trial.information_loss),
            trial.objectives.get("compute_cost", trial.compute_cost),
        ],
        dtype=np.float64,
    )


def dominates(a: TrialResult, b: TrialResult) -> bool:
    """Return True when ``a`` weakly dominates ``b`` in every objective."""
    va, vb = pareto_vector(a), pareto_vector(b)
    return bool(np.all(va <= vb) and np.any(va < vb))


def non_dominated_sort(trials: Sequence[TrialResult]) -> list[list[TrialResult]]:
    """Fast non-dominated sort (NSGA-II style) returning fronts best-first."""
    remaining = [trial for trial in trials if trial.status == "ok"]
    fronts: list[list[TrialResult]] = []
    while remaining:
        front = [
            trial
            for trial in remaining
            if not any(dominates(other, trial) for other in remaining if other is not trial)
        ]
        if not front:  # pragma: no cover - defensive
            front = [remaining[0]]
        fronts.append(front)
        remaining = [trial for trial in remaining if trial not in front]
    return fronts


def crowding_distance(front: Sequence[TrialResult]) -> dict[int, float]:
    """Crowding distance keyed by trial index (infinite for boundary solutions)."""
    if len(front) <= 2:
        return {trial.index: float("inf") for trial in front}
    vectors = np.vstack([pareto_vector(trial) for trial in front])
    distances = {trial.index: 0.0 for trial in front}
    n_objectives = vectors.shape[1]
    for objective in range(n_objectives):
        order = np.argsort(vectors[:, objective])
        values = vectors[order, objective]
        span = values[-1] - values[0]
        if span <= 1e-12:
            continue
        distances[front[order[0]].index] = float("inf")
        distances[front[order[-1]].index] = float("inf")
        for position in range(1, len(front) - 1):
            value = (values[position + 1] - values[position - 1]) / span
            current = distances[front[order[position]].index]
            if current != float("inf"):
                distances[front[order[position]].index] = current + float(value)
    return distances


def rank_trials(trials: Sequence[TrialResult]) -> None:
    """Assign Pareto rank and membership to every trial in place."""
    ok = [trial for trial in trials if trial.status == "ok"]
    feasible = [trial for trial in ok if trial.feasible]
    infeasible = [trial for trial in ok if not trial.feasible]
    failed = [trial for trial in trials if trial.status != "ok"]

    rank = 1
    for front in non_dominated_sort(feasible):
        distances = crowding_distance(front)
        ordered = sorted(front, key=lambda trial: (-distances[trial.index], trial.score), reverse=False)
        for trial in ordered:
            trial.rank = rank
            trial.is_pareto = rank == 1
        rank += 1

    for front in non_dominated_sort(infeasible):
        for trial in front:
            trial.rank = rank
            trial.is_pareto = False
        rank += 1

    for trial in failed:
        trial.rank = rank
        trial.is_pareto = False
        trial.feasible = False


def _knee(front: Sequence[TrialResult]) -> TrialResult | None:
    """Pick the Pareto solution closest to the ideal point (knee heuristic)."""
    if not front:
        return None
    vectors = np.vstack([pareto_vector(trial) for trial in front])
    ideal = vectors.min(axis=0)
    spans = vectors.max(axis=0) - ideal
    spans = np.where(spans < 1e-12, 1.0, spans)
    distances = np.linalg.norm((vectors - ideal) / spans, axis=1)
    return front[int(np.argmin(distances))]


def select_best(
    trials: Sequence[TrialResult],
    *,
    strategy: str = "balanced",
    accuracy_target: float | None = None,
    accuracy_tolerance: float = 0.02,
) -> TrialResult | None:
    """Select the recommended configuration.

    Strategies
    ----------
    ``"balanced"`` (default)
        Constrained sparsity: among trials whose accuracy is within
        ``accuracy_tolerance`` of the best observed accuracy (or above
        ``accuracy_target`` when one is given), pick the one with the fewest
        spikes, then the lowest compute cost.  This is the practical question a
        developer asks: *give me the cheapest encoding that does not cost me
        accuracy*.
    ``"knee"``
        Knee point of the Pareto front (closest to the ideal point).
    ``"accuracy"``
        Highest accuracy, fewest spikes as tie-break.
    ``"spikes"``
        Fewest spikes, accuracy as tie-break.
    ``"score"``
        Highest weighted score.
    """
    ok = [trial for trial in trials if trial.status == "ok"]
    if not ok:
        return None
    feasible = [trial for trial in ok if trial.feasible]
    pool = feasible or ok

    if strategy == "balanced":
        if feasible:
            best_accuracy = max(trial.accuracy for trial in feasible)
            threshold = (
                accuracy_target
                if accuracy_target is not None
                else max(0.0, best_accuracy - max(0.0, accuracy_tolerance))
            )
            candidates = [trial for trial in feasible if trial.accuracy >= threshold]
        else:
            # No configuration satisfies the constraints: report the most accurate one.
            candidates = ok
        return min(candidates, key=lambda trial: (trial.spike_count, -trial.accuracy, trial.compute_cost))

    if strategy == "knee":
        front = [trial for trial in pool if trial.is_pareto]
        return _knee(front) if front else max(pool, key=lambda trial: trial.score)
    if strategy == "accuracy":
        return max(pool, key=lambda trial: (trial.accuracy, -trial.spike_count, -trial.compute_cost))
    if strategy == "spikes":
        return min(pool, key=lambda trial: (trial.spike_count, -trial.accuracy))
    if strategy == "score":
        return max(pool, key=lambda trial: trial.score)
    raise ConfigurationError(
        f"Unknown selection strategy {strategy!r}; expected 'balanced', 'knee', 'accuracy', "
        "'spikes' or 'score'."
    )


def summarize_trials(trials: Iterable[TrialResult]) -> dict[str, Any]:
    """Summary statistics used by reports and the registry."""
    trials = list(trials)
    ok = [trial for trial in trials if trial.status == "ok"]
    if not ok:
        return {"n_trials": len(trials), "n_ok": 0}
    return {
        "n_trials": len(trials),
        "n_ok": len(ok),
        "n_feasible": sum(1 for trial in ok if trial.feasible),
        "n_pareto": sum(1 for trial in ok if trial.is_pareto),
        "accuracy_min": min(trial.accuracy for trial in ok),
        "accuracy_max": max(trial.accuracy for trial in ok),
        "accuracy_mean": float(np.mean([trial.accuracy for trial in ok])),
        "spike_count_min": min(trial.spike_count for trial in ok),
        "spike_count_max": max(trial.spike_count for trial in ok),
        "information_loss_min": min(trial.information_loss for trial in ok),
        "information_loss_max": max(trial.information_loss for trial in ok),
    }
