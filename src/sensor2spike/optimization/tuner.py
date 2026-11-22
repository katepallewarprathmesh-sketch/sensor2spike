"""High-level optimisation API.

``AutoTuner`` is the facade described in the specification (section 10.1)::

    from sensor2spike import SensorDataset, AutoTuner

    dataset = SensorDataset.load("radar_recordings.csv")
    tuner = AutoTuner(objective="multi_objective", encoder="latency")
    results = tuner.optimize(dataset=dataset, accuracy_target=0.95, spike_budget=10000)
    results.export("optimized_config.json")
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any

from ..contracts import ExperimentResult, SensorDataset
from ..data.preprocessing import PreprocessConfig
from ..exceptions import ConfigurationError
from ..snn import SNNBackend, build_backend
from ..spaces import SearchSpace
from ..utils.validation import reject_unknown_keys, require_choice, require_positive, require_range
from .objectives import ObjectiveWeights

__all__ = ["OptimizeConfig", "AutoTuner", "OBJECTIVE_ALIASES"]

OBJECTIVE_ALIASES = {
    "multi": "multi",
    "multi_objective": "multi",
    "pareto": "multi",
    "weighted": "weighted",
    "weighted_sum": "weighted",
}


@dataclass
class OptimizeConfig:
    """Complete description of one optimisation experiment."""

    encoder: str = "rate"
    encoder_params: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, Any] | None = None
    optimizer: str = "random"
    n_trials: int = 20
    objective: str = "multi"
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    accuracy_target: float | None = None
    spike_budget: int | None = None
    seed: int = 0
    backend: Any = "lif_ridge"
    preprocess: Any = None
    window_seconds: float | None = None
    window_stride_seconds: float | None = None
    test_fraction: float = 0.3
    n_grid_points: int = 3
    max_grid_combinations: int = 256
    selection: str | None = None
    accuracy_tolerance: float = 0.02
    fail_fast: bool = False
    persist: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        self.encoder = str(self.encoder)
        if not self.encoder:
            raise ConfigurationError("OptimizeConfig.encoder must not be empty.")
        if not isinstance(self.encoder_params, Mapping):
            raise ConfigurationError("OptimizeConfig.encoder_params must be a mapping.")
        self.encoder_params = dict(self.encoder_params)
        if self.search_space is not None and not isinstance(self.search_space, Mapping):
            raise ConfigurationError("OptimizeConfig.search_space must be a mapping.")
        self.optimizer = str(self.optimizer)
        if int(self.n_trials) < 1:
            raise ConfigurationError("OptimizeConfig.n_trials must be >= 1.")
        self.n_trials = int(self.n_trials)
        self.objective = OBJECTIVE_ALIASES.get(str(self.objective).strip().lower(), "")
        if self.objective not in {"multi", "weighted"}:
            raise ConfigurationError(
                "OptimizeConfig.objective must be 'multi' (Pareto) or 'weighted'; "
                f"got {self.objective!r}."
            )
        if not isinstance(self.weights, ObjectiveWeights):
            self.weights = ObjectiveWeights.from_dict(self.weights)
        if self.accuracy_target is not None:
            self.accuracy_target = require_range(self.accuracy_target, "accuracy_target", low=0.0, high=1.0)
        if self.spike_budget is not None:
            self.spike_budget = int(require_positive(self.spike_budget, "spike_budget"))
        if self.window_seconds is not None:
            self.window_seconds = require_positive(self.window_seconds, "window_seconds")
        if self.window_stride_seconds is not None:
            self.window_stride_seconds = require_positive(self.window_stride_seconds, "window_stride_seconds")
        self.test_fraction = require_range(self.test_fraction, "test_fraction", low=0.0, high=1.0)
        if not 0.0 < self.test_fraction < 1.0:
            raise ConfigurationError("OptimizeConfig.test_fraction must be strictly between 0 and 1.")
        if int(self.n_grid_points) < 1:
            raise ConfigurationError("OptimizeConfig.n_grid_points must be >= 1.")
        self.n_grid_points = int(self.n_grid_points)
        if int(self.max_grid_combinations) < 1:
            raise ConfigurationError("OptimizeConfig.max_grid_combinations must be >= 1.")
        self.max_grid_combinations = int(self.max_grid_combinations)
        if self.selection is not None:
            self.selection = str(
                require_choice(self.selection, "selection", ("balanced", "knee", "accuracy", "spikes", "score"))
            )
        self.accuracy_tolerance = require_range(
            self.accuracy_tolerance, "accuracy_tolerance", low=0.0, high=1.0
        )
        if self.notes is None:
            self.notes = ""
        self.notes = str(self.notes)

    # ------------------------------------------------------------------ helpers
    def resolved_preprocess(self) -> PreprocessConfig:
        if isinstance(self.preprocess, PreprocessConfig):
            return self.preprocess
        return PreprocessConfig.from_dict(self.preprocess)

    def resolved_backend(self) -> SNNBackend:
        return build_backend(self.backend)

    def resolved_search_space(self) -> SearchSpace:
        from ..encoders import build_encoder

        encoder = build_encoder(self.encoder, **self.encoder_params)
        return encoder.to_space(self.search_space)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for spec in fields(self):
            value = getattr(self, spec.name)
            if spec.name == "backend" and isinstance(value, SNNBackend):
                value = value.get_config()
            elif spec.name == "weights":
                value = self.weights.to_dict()
            elif spec.name == "preprocess" and isinstance(value, PreprocessConfig):
                value = value.to_dict()
            else:
                value = value
            payload[spec.name] = value
        return payload

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any] | None, *, strict: bool = True) -> OptimizeConfig:
        if mapping is None:
            return cls()
        if not isinstance(mapping, Mapping):
            raise ConfigurationError("OptimizeConfig.from_dict expects a mapping.")
        if strict:
            reject_unknown_keys(mapping, [spec.name for spec in fields(cls)], name="optimize")
        data = dict(mapping)
        weights = data.get("weights")
        if isinstance(weights, Mapping):
            data["weights"] = ObjectiveWeights.from_dict(dict(weights))
        return cls(**data)


class AutoTuner:
    """Facade that turns a dataset into a ranked set of encoder configurations."""

    def __init__(
        self,
        objective: str = "multi",
        *,
        encoder: str = "rate",
        backend: Any = "lif_ridge",
        registry: Any = None,
        **kwargs: Any,
    ) -> None:
        self._registry = registry
        self.config = OptimizeConfig(
            objective=objective, encoder=encoder, backend=backend, **kwargs
        )

    @property
    def registry(self) -> Any:
        return self._registry

    def optimize(
        self,
        dataset: SensorDataset,
        *,
        notes: str | None = None,
        **overrides: Any,
    ) -> ExperimentResult:
        """Run the optimisation and return a complete :class:`ExperimentResult`."""
        from ..experiment.runner import ExperimentRunner

        config = self.config
        if overrides:
            config = replace(config, **overrides)
        if notes:
            config = replace(config, notes=notes)
        runner = ExperimentRunner(self._registry)
        return runner.run(dataset, config)

    def __repr__(self) -> str:
        return (
            f"AutoTuner(encoder={self.config.encoder!r}, optimizer={self.config.optimizer!r}, "
            f"objective={self.config.objective!r}, n_trials={self.config.n_trials})"
        )


