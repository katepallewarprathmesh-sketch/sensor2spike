"""Tests for objectives, Pareto ranking and search strategies."""

from __future__ import annotations

import numpy as np
import pytest

from sensor2spike.contracts import TrialResult
from sensor2spike.exceptions import ConfigurationError, DependencyMissingError
from sensor2spike.optimization import (
    BayesianSearch,
    GridSearch,
    ObjectiveNormalizers,
    ObjectiveWeights,
    OptimizeConfig,
    RandomSearch,
    available_strategies,
    build_search,
    compute_objectives,
    crowding_distance,
    dominates,
    non_dominated_sort,
    rank_trials,
    select_best,
    summarize_trials,
)
from sensor2spike.spaces import Parameter, SearchSpace


def _trial(
    index: int,
    accuracy: float,
    spikes: int,
    information: float = 0.5,
    cost: float = 1000.0,
) -> TrialResult:
    return TrialResult(
        index=index,
        params={"x": index},
        accuracy=accuracy,
        macro_f1=accuracy,
        spike_count=spikes,
        information_loss=information,
        compute_cost=cost,
    )


# ------------------------------------------------------------------------ weights
def test_objective_weights_validation_and_dict():
    weights = ObjectiveWeights(accuracy=2.0, spikes=0.5)
    assert weights.accuracy == pytest.approx(2.0)
    assert weights.to_dict()["spikes"] == pytest.approx(0.5)
    with pytest.raises(ConfigurationError):
        ObjectiveWeights(accuracy=-1.0)
    with pytest.raises(ConfigurationError):
        ObjectiveWeights.from_dict({"unknown": 1.0})
    with pytest.raises(ConfigurationError):
        ObjectiveWeights(accuracy=0.0, spikes=0.0, information=0.0, compute=0.0)


# -------------------------------------------------------------------- normalisers
def test_normalizers_fit_caps_at_one():
    trials = [_trial(0, 0.5, 100), _trial(1, 0.9, 1000)]
    normalizers = ObjectiveNormalizers.fit(trials)
    trial = _trial(2, 0.9, 5000, information=9.0, cost=1e9)
    compute_objectives(trial, normalizers, ObjectiveWeights())
    assert trial.objectives["spikes"] == pytest.approx(1.0)
    assert trial.objectives["information_loss"] == pytest.approx(1.0)
    assert trial.objectives["compute_cost"] == pytest.approx(1.0)


def test_compute_objectives_sets_score_and_feasibility():
    normalizers = ObjectiveNormalizers.fit([_trial(0, 0.5, 100), _trial(1, 1.0, 1000)])
    weights = ObjectiveWeights(accuracy=1.0, spikes=0.0, information=0.0, compute=0.0)
    trial = compute_objectives(_trial(0, 0.8, 500), normalizers, weights)
    assert trial.score == pytest.approx(-0.2)
    assert trial.feasible

    constrained = compute_objectives(
        _trial(1, 0.8, 5000), normalizers, weights, accuracy_target=0.95, spike_budget=1000
    )
    assert not constrained.feasible


# ------------------------------------------------------------------------- pareto
def test_dominance_definition():
    good = _trial(0, accuracy=0.9, spikes=100)
    bad = _trial(1, accuracy=0.8, spikes=200)
    assert dominates(good, bad)
    assert not dominates(bad, good)
    tie = _trial(2, accuracy=0.9, spikes=100)
    assert not dominates(good, tie)


def test_non_dominated_sort_orders_fronts():
    trials = [_trial(0, 0.9, 100), _trial(1, 0.8, 200), _trial(2, 0.7, 300)]
    fronts = non_dominated_sort(trials)
    assert [trial.index for trial in fronts[0]] == [0]
    assert [trial.index for trial in fronts[1]] == [1]
    assert [trial.index for trial in fronts[2]] == [2]


def test_rank_trials_marks_pareto_members():
    trials = [_trial(0, 0.9, 100), _trial(1, 0.95, 500), _trial(2, 0.5, 200)]
    rank_trials(trials)
    pareto = [trial.index for trial in trials if trial.is_pareto]
    assert 0 in pareto and 1 in pareto
    assert 2 not in pareto
    assert all(trial.rank >= 1 for trial in trials)


def test_crowding_distance_prefers_isolated_points():
    front = [_trial(0, 0.5, 100), _trial(1, 0.9, 200), _trial(2, 0.95, 900)]
    distances = crowding_distance(front)
    # The boundary points are infinitely separated.
    assert distances[0] == pytest.approx(np.inf)
    assert distances[2] == pytest.approx(np.inf)
    assert distances[1] < np.inf


# --------------------------------------------------------------------- selection
def test_select_best_strategies_disagree_sensibly():
    trials = [
        _trial(0, 0.99, 10_000),
        _trial(1, 0.95, 1_000),
        _trial(2, 0.60, 100),
    ]
    rank_trials(trials)
    assert select_best(trials, strategy="accuracy").index == 0
    assert select_best(trials, strategy="spikes").index == 2
    balanced = select_best(trials, strategy="balanced", accuracy_tolerance=0.05)
    # 0.95 is within 5 points of 0.99 and uses a tenth of the spikes.
    assert balanced.index == 1


def test_select_best_returns_none_without_ok_trials():
    failed = _trial(0, 0.0, 0)
    failed.status = "failed"
    assert select_best([failed]) is None


def test_select_best_rejects_unknown_strategy():
    with pytest.raises(ConfigurationError):
        select_best([_trial(0, 0.9, 10)], strategy="magic")


def test_summarize_trials_counts_statuses():
    trials = [_trial(0, 0.9, 100), _trial(1, 0.8, 200)]
    trials[1].status = "failed"
    summary = summarize_trials(trials)
    assert summary["n_trials"] == 2
    assert summary["n_ok"] == 1


# ----------------------------------------------------------------- search strategies
SPACE = SearchSpace(
    {
        "rate": Parameter("rate", "float", 10.0, 100.0, log=True),
        "count": Parameter("count", "int", 2, 6),
        "mode": Parameter("mode", "choice", choices=("fast", "slow")),
    }
)


def _evaluator(counter: list[int]):
    def evaluate(params: dict) -> TrialResult:
        counter.append(1)
        index = len(counter) - 1
        # Deterministic objective so tests can assert on ordering.
        score = params["rate"] / 100.0 + params["count"] / 10.0 + (0.1 if params["mode"] == "fast" else 0.0)
        return TrialResult(
            index=index,
            params=params,
            accuracy=min(1.0, score),
            macro_f1=min(1.0, score),
            spike_count=int(params["rate"] * params["count"]),
            information_loss=1.0 - min(1.0, score),
            compute_cost=float(params["rate"] * 10),
        )

    return evaluate


def test_registry_of_strategies():
    assert set(available_strategies()) >= {"grid", "random", "bayes"}
    assert isinstance(build_search("grid"), GridSearch)
    assert isinstance(build_search("random", n_trials=3), RandomSearch)
    with pytest.raises(ConfigurationError):
        build_search("magic")


def test_random_search_is_seeded_and_reproducible():
    calls: list[int] = []
    first = RandomSearch(n_trials=6, seed=5).search(SPACE, _evaluator(calls))
    assert len(first) == 6
    second = RandomSearch(n_trials=6, seed=5).search(SPACE, _evaluator(calls))
    assert [trial.params for trial in first] == [trial.params for trial in second]
    # Every produced configuration is valid for the space.
    for trial in first:
        SPACE.validate(trial.params)


def test_grid_search_is_deterministic_and_capped():
    calls: list[int] = []
    trials = GridSearch(n_trials=5, n_points=3, max_combinations=8).search(SPACE, _evaluator(calls))
    assert len(trials) <= 5
    assert len(calls) == len(trials)


def test_bayesian_search_requires_optuna(monkeypatch):
    import sensor2spike.optimization.search as search_module

    monkeypatch.setattr(search_module, "HAS_OPTUNA", False)
    with pytest.raises(DependencyMissingError) as excinfo:
        BayesianSearch(n_trials=2).search(SPACE, _evaluator([]))
    assert "optuna" in str(excinfo.value).lower()


def test_bayesian_search_runs_when_optuna_available():
    pytest.importorskip("optuna")
    calls: list[int] = []
    trials = BayesianSearch(n_trials=4, seed=1, n_startup_trials=2).search(
        SPACE, _evaluator(calls)
    )
    assert len(trials) == 4
    assert all(trial.status == "ok" for trial in trials)


# --------------------------------------------------------------------- config object
def test_optimize_config_validation_and_aliases():
    config = OptimizeConfig(objective="pareto", encoder="rate", n_trials=3)
    assert config.objective == "multi"
    config = OptimizeConfig(objective="weighted_sum", encoder="rate")
    assert config.objective == "weighted"
    with pytest.raises(ConfigurationError):
        OptimizeConfig(objective="magic", encoder="rate")
    with pytest.raises(ConfigurationError):
        OptimizeConfig(objective="multi", encoder="rate", n_trials=0)
    with pytest.raises(ConfigurationError):
        OptimizeConfig(objective="multi", encoder="rate", selection="magic")
    with pytest.raises(ConfigurationError):
        OptimizeConfig.from_dict({"objective": "multi", "encoder": "rate", "bogus": 1})


def test_optimize_config_resolves_preprocess_and_space():
    config = OptimizeConfig(
        objective="multi",
        encoder="rate",
        preprocess={"normalize": "zscore"},
        search_space={"rate_max_hz": {"kind": "choice", "choices": (50.0, 100.0)}},
    )
    resolved = config.resolved_preprocess()
    assert resolved.normalize == "zscore"
    space = config.resolved_search_space()
    assert space["rate_max_hz"].kind == "choice"
    assert space["rate_max_hz"].choices == (50.0, 100.0)


def test_optimize_config_encoder_params_narrow_the_space():
    config = OptimizeConfig(objective="multi", encoder="rate", encoder_params={"dt_ms": 2.0})
    space = config.resolved_search_space()
    assert space["dt_ms"].choices == (2.0,)
