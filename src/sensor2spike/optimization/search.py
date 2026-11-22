"""Search strategies: grid, random and Bayesian optimisation.

Every strategy receives a ``space`` and an ``evaluate(params)`` callback and
returns the list of :class:`~sensor2spike.contracts.TrialResult` it produced.
This keeps the search algorithms independent of the evaluation pipeline, which
is what makes the optimisation engine pluggable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np

from ..contracts import TrialResult
from ..exceptions import ConfigurationError, DependencyMissingError
from ..spaces import SearchSpace

__all__ = [
    "SearchStrategy",
    "GridSearch",
    "RandomSearch",
    "BayesianSearch",
    "build_search",
    "SEARCH_STRATEGIES",
]

Evaluate = Callable[[dict[str, Any]], TrialResult]


try:  # pragma: no cover - environment dependent
    import optuna as _optuna  # noqa: F401

    HAS_OPTUNA = True
except ImportError:  # pragma: no cover
    HAS_OPTUNA = False


class SearchStrategy(ABC):
    """Base class for search strategies."""

    name: ClassVar[str] = "base"

    def __init__(self, *, n_trials: int = 20, seed: int = 0) -> None:
        if n_trials < 1:
            raise ConfigurationError("n_trials must be >= 1.")
        self.n_trials = int(n_trials)
        self.seed = int(seed)

    @abstractmethod
    def search(
        self,
        space: SearchSpace,
        evaluate: Evaluate,
        n_trials: int | None = None,
    ) -> list[TrialResult]:
        """Run the search and return one result per evaluated configuration."""

    def get_config(self) -> dict[str, Any]:
        return {"strategy": self.name, "n_trials": self.n_trials, "seed": self.seed}


class GridSearch(SearchStrategy):
    """Deterministic grid search over the encoder's search space."""

    name = "grid"

    def __init__(
        self,
        *,
        n_trials: int = 20,
        seed: int = 0,
        n_points: int = 3,
        max_combinations: int = 256,
    ) -> None:
        super().__init__(n_trials=n_trials, seed=seed)
        self.n_points = int(n_points)
        self.max_combinations = int(max_combinations)

    def search(self, space, evaluate, n_trials=None):
        total = self.n_trials if n_trials is None else int(n_trials)
        combinations = space.grid(self.n_points, max_combinations=self.max_combinations)
        if len(combinations) > total:
            order = np.random.default_rng(self.seed).permutation(len(combinations))[:total]
            combinations = [combinations[index] for index in order]
        return [evaluate(space.validate(params)) for params in combinations]

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "n_points": self.n_points, "max_combinations": self.max_combinations}


class RandomSearch(SearchStrategy):
    """Random sampling of the search space (strong baseline, embarrassingly parallel)."""

    name = "random"

    def search(self, space, evaluate, n_trials=None):
        total = self.n_trials if n_trials is None else int(n_trials)
        rng = np.random.default_rng(self.seed)
        results: list[TrialResult] = []
        for _ in range(total):
            results.append(evaluate(space.validate(space.sample(rng))))
        return results


class BayesianSearch(SearchStrategy):
    """Bayesian optimisation backed by Optuna's TPE sampler.

    Requires the optional ``bayes`` extra::

        pip install "sensor2spike[bayes]"
    """

    name = "bayes"

    def __init__(self, *, n_trials: int = 20, seed: int = 0, n_startup_trials: int = 5) -> None:
        super().__init__(n_trials=n_trials, seed=seed)
        self.n_startup_trials = int(n_startup_trials)

    def search(self, space, evaluate, n_trials=None):
        if not HAS_OPTUNA:  # pragma: no cover - environment dependent
            raise DependencyMissingError(
                "optuna", extra="bayes", purpose="Bayesian optimisation"
            )
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.seed, n_startup_trials=self.n_startup_trials)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        results: list[TrialResult] = []

        def objective(trial: optuna.trial.Trial) -> float:
            params: dict[str, Any] = {}
            for name, parameter in space.items():
                if parameter.kind == "float":
                    assert parameter.low is not None and parameter.high is not None
                    params[name] = trial.suggest_float(
                        name, float(parameter.low), float(parameter.high), log=parameter.log
                    )
                elif parameter.kind == "int":
                    assert parameter.low is not None and parameter.high is not None
                    params[name] = trial.suggest_int(
                        name, int(parameter.low), int(parameter.high), log=parameter.log
                    )
                else:
                    params[name] = trial.suggest_categorical(name, list(parameter.choices))
            result = evaluate(space.validate(params))
            results.append(result)
            if result.status != "ok":
                raise optuna.TrialPruned()
            # ``score`` is higher-is-better; Optuna minimises.
            return float(-result.score)

        study.optimize(objective, n_trials=self.n_trials if n_trials is None else int(n_trials))
        return results

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "n_startup_trials": self.n_startup_trials}


SEARCH_STRATEGIES: dict[str, type[SearchStrategy]] = {
    "grid": GridSearch,
    "random": RandomSearch,
    "bayes": BayesianSearch,
    "bayesian": BayesianSearch,
}


def build_search(name: str | SearchStrategy, **kwargs: Any) -> SearchStrategy:
    """Instantiate a search strategy by name."""
    if isinstance(name, SearchStrategy):
        return name
    key = str(name).strip().lower()
    if key not in SEARCH_STRATEGIES:
        available = ", ".join(sorted(set(SEARCH_STRATEGIES)))
        raise ConfigurationError(
            f"Unknown search strategy {name!r}. Available strategies: {available}."
        )
    return SEARCH_STRATEGIES[key](**kwargs)


def available_strategies() -> Sequence[str]:
    """Return the unique strategy names."""
    seen: list[str] = []
    for key in SEARCH_STRATEGIES:
        if key not in seen:
            seen.append(key)
    return tuple(seen)
