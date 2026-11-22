"""Experiment orchestration.

``ExperimentRunner`` wires the pipeline together::

    dataset -> preprocess -> window -> split -> [encode -> evaluate]* -> rank -> report

Every candidate configuration is evaluated in isolation, so a failure in one
trial never corrupts the results of another (reliability requirement 13).
"""

from __future__ import annotations

import importlib.metadata
import itertools
import platform
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from ..analysis.spikes import compute_spike_stats
from ..contracts import ExperimentResult, SensorDataset, TrialResult
from ..data.preprocessing import preprocess
from ..data.splits import split_windows
from ..encoders import build_encoder
from ..exceptions import DataValidationError
from ..optimization.objectives import (
    ObjectiveNormalizers,
    compute_objectives,
    rank_trials,
    select_best,
    summarize_trials,
)
from ..optimization.search import SearchStrategy, build_search
from ..optimization.tuner import OptimizeConfig
from ..snn import SNNBackend
from ..utils.hashing import stable_hash
from ..utils.security import redact_paths
from .registry import ExperimentRegistry

__all__ = ["ExperimentRunner", "DEFAULT_WINDOW_SECONDS", "software_versions"]

DEFAULT_WINDOW_SECONDS = 1.0


def software_versions() -> dict[str, Any]:
    """Record the dependency versions needed to reproduce an experiment."""
    try:
        package_version = importlib.metadata.version("sensor2spike")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - local checkout
        package_version = "0.0.0+local"
    return {
        "sensor2spike": package_version,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ExperimentRunner:
    """Run a complete optimisation experiment."""

    def __init__(
        self,
        registry: ExperimentRegistry | str | Path | None = None,
        *,
        persist: bool = True,
    ) -> None:
        if registry is None:
            self.registry: ExperimentRegistry | None = None
        elif isinstance(registry, ExperimentRegistry):
            self.registry = registry
        else:
            self.registry = ExperimentRegistry(registry)
        self.persist = bool(persist)

    # ------------------------------------------------------------------ public API
    def run(
        self,
        dataset: SensorDataset,
        config: OptimizeConfig | Mapping[str, Any] | None = None,
        *,
        notes: str | None = None,
    ) -> ExperimentResult:
        """Execute the experiment described by ``config``."""
        config = config if isinstance(config, OptimizeConfig) else OptimizeConfig.from_dict(config)
        if notes:
            config = replace(config, notes=notes)

        started = time.perf_counter()
        created_at = _utc_now()

        processed = preprocess(dataset, config.resolved_preprocess())
        window_samples, stride_samples = self._windowing(processed, config)
        windows = processed.to_windows(window_samples, stride_samples)
        train, test = split_windows(
            windows, test_fraction=config.test_fraction, seed=config.seed
        )

        backend = config.resolved_backend()
        space = config.resolved_search_space()
        strategy = self._strategy(config)

        counter = itertools.count()
        partial: dict[str, list[TrialResult]] = {"trials": []}

        def _failed(index: int, params: dict[str, Any], exc: BaseException) -> TrialResult:
            return TrialResult(
                index=index,
                params=dict(params),
                status="failed",
                error=redact_paths(f"{type(exc).__name__}: {exc}"),
                feasible=False,
                score=-1e9,
            )

        def evaluate(params: dict[str, Any]) -> TrialResult:
            index = next(counter)
            try:
                trial = self._evaluate(index, params, train, test, backend, config)
            except Exception as exc:  # noqa: BLE001 - a single trial must not kill the run
                trial = _failed(index, params, exc)
                partial["trials"].append(trial)
                if config.fail_fast:
                    raise
            else:
                partial["trials"].append(trial)
            return trial

        search_error: str | None = None
        try:
            trials = strategy.search(space, evaluate, n_trials=config.n_trials)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            search_error = redact_paths(f"{type(exc).__name__}: {exc}")
            trials = list(partial["trials"])

        # Recompute objectives with normalisers fitted on the *whole* batch so
        # that scores are comparable across all trials.
        ok_trials = [trial for trial in trials if trial.status == "ok"]
        normalizers = ObjectiveNormalizers.fit(ok_trials)
        for trial in ok_trials:
            compute_objectives(
                trial,
                normalizers,
                config.weights,
                accuracy_target=config.accuracy_target,
                spike_budget=config.spike_budget,
            )
        for trial in trials:
            if trial.status != "ok":
                trial.objectives = {}
                trial.score = -1e9
                trial.feasible = False
        rank_trials(trials)

        selection = config.selection or (
            "balanced" if config.objective == "multi" else "score"
        )
        best = select_best(
            trials,
            strategy=selection,
            accuracy_target=config.accuracy_target,
            accuracy_tolerance=config.accuracy_tolerance,
        )
        summary = summarize_trials(trials)

        result = ExperimentResult(
            experiment_id="",
            dataset={
                **processed.summary(),
                "source_name": dataset.name,
                "source_fingerprint": dataset.fingerprint(),
                "preprocessing_applied": bool(config.resolved_preprocess().enabled),
            },
            encoder={
                "encoder": config.encoder,
                "dt_ms": None,
                "seed": config.seed,
                **config.encoder_params,
            },
            search_space=space.to_dict(),
            optimizer=strategy.name,
            n_trials=len(trials),
            seed=config.seed,
            objective=config.objective,
            weights=config.weights.to_dict(),
            accuracy_target=config.accuracy_target,
            spike_budget=config.spike_budget,
            backend=backend.get_config(),
            preprocess=config.resolved_preprocess().to_dict(),
            window={
                "window_seconds": window_samples / processed.sampling_frequency,
                "window_samples": int(window_samples),
                "stride_seconds": stride_samples / processed.sampling_frequency,
                "stride_samples": int(stride_samples),
                "n_windows": int(windows.n_windows),
            },
            split={
                "test_fraction": float(config.test_fraction),
                "n_train_windows": int(train.n_windows),
                "n_test_windows": int(test.n_windows),
                "n_train_groups": int(np.unique(train.groups).size) if train.groups is not None else int(train.n_windows),
                "n_test_groups": int(np.unique(test.groups).size) if test.groups is not None else int(test.n_windows),
                "train_labels": None if train.labels is None else np.unique(train.labels).tolist(),
                "test_labels": None if test.labels is None else np.unique(test.labels).tolist(),
            },
            trials=trials,
            status="completed" if ok_trials else "failed",
            created_at=created_at,
            duration_s=float(time.perf_counter() - started),
            software=software_versions(),
            notes=config.notes,
            normalization=normalizers.to_dict(),
            best_index=None if best is None else int(best.index),
            extras={
                "summary": summary,
                "selection": selection,
                **({} if search_error is None else {"error": search_error}),
            },
        )
        result.experiment_id = self._experiment_id(result)
        if search_error is not None:
            result.status = "failed"

        if self.registry is not None and self.persist:
            self.registry.record_experiment(result, processed)
        return result

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _experiment_id(result: ExperimentResult) -> str:
        """Deterministic identifier for a result (same inputs -> same id)."""
        payload = {
            "dataset": result.dataset,
            "encoder": result.encoder,
            "search_space": result.search_space,
            "optimizer": result.optimizer,
            "objective": result.objective,
            "weights": result.weights,
            "accuracy_target": result.accuracy_target,
            "spike_budget": result.spike_budget,
            "backend": result.backend,
            "preprocess": result.preprocess,
            "window": result.window,
            "split": result.split,
            "seed": result.seed,
            "n_trials": result.n_trials,
            "notes": result.notes,
            "software": result.software,
        }
        return "exp_" + stable_hash(payload, length=16)

    @staticmethod
    def _windowing(
        dataset: SensorDataset, config: OptimizeConfig
    ) -> tuple[int, int]:
        fs = dataset.sampling_frequency
        requested = config.window_seconds or min(
            DEFAULT_WINDOW_SECONDS, max(dataset.duration_seconds, 1e-6)
        )
        window_samples = max(1, int(round(requested * fs)))
        segment_lengths = [stop - start for start, stop in dataset.segments()]
        if segment_lengths:
            window_samples = min(window_samples, min(segment_lengths))
        if window_samples > dataset.n_samples:
            window_samples = dataset.n_samples
        if window_samples < 1:
            raise DataValidationError("Window size resolved to zero samples.")

        stride_seconds = config.window_stride_seconds or (window_samples / fs)
        stride_samples = max(1, int(round(stride_seconds * fs)))
        stride_samples = min(stride_samples, window_samples)
        return window_samples, stride_samples

    @staticmethod
    def _strategy(config: OptimizeConfig) -> SearchStrategy:
        kwargs: dict[str, Any] = {}
        if config.optimizer.lower() == "grid":
            kwargs = {
                "n_points": config.n_grid_points,
                "max_combinations": config.max_grid_combinations,
            }
        return build_search(
            config.optimizer, n_trials=config.n_trials, seed=config.seed, **kwargs
        )

    @staticmethod
    def _evaluate(
        index: int,
        params: dict[str, Any],
        train,
        test,
        backend: SNNBackend,
        config: OptimizeConfig,
    ) -> TrialResult:
        merged = {**config.encoder_params, **params}
        # Deterministic reruns (spec 14): the encoder's RNG is seeded from the
        # experiment seed and the trial index, never from process state.
        encoder = build_encoder(config.encoder, seed=config.seed + index, **merged)
        encoder.fit(np.concatenate(train.samples, axis=0), fs=train.sampling_frequency)

        train_spikes = encoder.transform_windows(train)
        test_spikes = encoder.transform_windows(test)

        losses: list[float] = []
        for spikes, window in zip(train_spikes, train.samples, strict=False):
            loss = encoder.information_loss(spikes, window)
            if loss is not None and np.isfinite(loss):
                losses.append(float(loss))
        information_loss = float(np.mean(losses)) if losses else 0.0

        evaluation = backend.train_evaluate(
            train_spikes,
            train.labels,
            test_spikes,
            test.labels,
            seed=config.seed,
        )
        stats = compute_spike_stats(
            train_spikes,
            information_loss=information_loss,
            reconstruction_r2=1.0 - information_loss,
        )
        return TrialResult(
            index=index,
            params=dict(params),
            accuracy=float(evaluation.accuracy),
            macro_f1=float(evaluation.macro_f1),
            train_accuracy=evaluation.train_accuracy,
            spike_count=int(stats.spike_count),
            spike_rate_hz=float(stats.spike_rate_hz),
            spikes_per_second_per_neuron=float(stats.spikes_per_second_per_neuron),
            temporal_sparsity=float(stats.temporal_sparsity),
            information_loss=information_loss,
            reconstruction_r2=1.0 - information_loss,
            compute_cost=float(stats.compute_cost),
            inference_time_s=float(evaluation.inference_time_s),
            train_time_s=float(evaluation.train_time_s),
            evaluation={
                "backend": evaluation.backend,
                "n_parameters": int(evaluation.n_parameters),
                "n_train": int(evaluation.n_train),
                "n_test": int(evaluation.n_test),
                "train_accuracy": evaluation.train_accuracy,
                "per_class_accuracy": evaluation.extra.get("summary", {}).get("per_class_accuracy"),
                "confusion_matrix": evaluation.extra.get("summary", {}).get("confusion_matrix"),
            },
        )
