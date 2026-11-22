"""Optimisation engine: objectives, search strategies and the ``AutoTuner`` facade."""

from __future__ import annotations

from .objectives import (
    ObjectiveNormalizers,
    ObjectiveWeights,
    compute_objectives,
    crowding_distance,
    dominates,
    non_dominated_sort,
    rank_trials,
    select_best,
    summarize_trials,
)
from .search import (
    SEARCH_STRATEGIES,
    BayesianSearch,
    GridSearch,
    RandomSearch,
    SearchStrategy,
    available_strategies,
    build_search,
)
from .tuner import AutoTuner, OptimizeConfig

__all__ = [
    "AutoTuner",
    "OptimizeConfig",
    "ObjectiveWeights",
    "ObjectiveNormalizers",
    "compute_objectives",
    "crowding_distance",
    "dominates",
    "non_dominated_sort",
    "rank_trials",
    "select_best",
    "summarize_trials",
    "SEARCH_STRATEGIES",
    "SearchStrategy",
    "GridSearch",
    "RandomSearch",
    "BayesianSearch",
    "build_search",
    "available_strategies",
]
