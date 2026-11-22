"""SQLite experiment registry.

Stores a relational index (datasets, encoder configurations, experiments,
results) *and* the full JSON payload of every experiment.  The relational rows
make experiments queryable with SQL; the payload guarantees exact reproduction
of a reported result.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..contracts import ExperimentResult, SensorDataset, TrialResult
from ..exceptions import RegistryError
from ..utils.hashing import stable_hash

__all__ = ["ExperimentRegistry", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sensor_type TEXT,
    sampling_frequency REAL,
    n_samples INTEGER,
    n_channels INTEGER,
    duration_seconds REAL,
    fingerprint TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS encoder_configs (
    config_id TEXT PRIMARY KEY,
    encoder_type TEXT NOT NULL,
    dt_ms REAL,
    params_json TEXT NOT NULL,
    config_hash TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    config_json TEXT NOT NULL,
    n_parameters INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    dataset_id TEXT,
    config_id TEXT,
    model_id TEXT,
    optimizer TEXT,
    objective TEXT,
    seed INTEGER,
    status TEXT,
    n_trials INTEGER,
    accuracy_target REAL,
    spike_budget INTEGER,
    duration_s REAL,
    best_accuracy REAL,
    best_spike_count INTEGER,
    best_index INTEGER,
    notes TEXT,
    started_at TEXT,
    finished_at TEXT,
    software_json TEXT,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    trial_index INTEGER NOT NULL,
    params_json TEXT NOT NULL,
    accuracy REAL,
    macro_f1 REAL,
    train_accuracy REAL,
    spike_count INTEGER,
    spike_rate_hz REAL,
    information_loss REAL,
    compute_cost REAL,
    score REAL,
    rank INTEGER,
    is_pareto INTEGER,
    feasible INTEGER,
    status TEXT,
    error TEXT,
    objectives_json TEXT,
    evaluation_json TEXT,
    inference_time_s REAL,
    train_time_s REAL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_results_experiment ON results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ExperimentRegistry:
    """A local SQLite-backed experiment registry."""

    def __init__(self, path: str | Path = "sensor2spike.db") -> None:
        self.path = Path(path)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    # ------------------------------------------------------------------ connection
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        except sqlite3.Error as exc:
            raise RegistryError(f"Registry operation failed: {exc}") from exc
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._transaction() as connection:
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    # ------------------------------------------------------------------ datasets / configs
    def record_dataset(self, dataset: SensorDataset) -> str:
        """Insert (or reuse) a dataset row and return its identifier."""
        summary = dataset.summary()
        dataset_id = "ds_" + stable_hash(
            {
                "fingerprint": summary["fingerprint"],
                "name": dataset.name,
                "fs": dataset.sampling_frequency,
            },
            length=16,
        )
        with self._transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO datasets
                   (dataset_id, name, sensor_type, sampling_frequency, n_samples, n_channels,
                    duration_seconds, fingerprint, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    dataset_id,
                    dataset.name,
                    dataset.sensor_type,
                    float(dataset.sampling_frequency),
                    int(dataset.n_samples),
                    int(dataset.n_channels),
                    float(dataset.duration_seconds),
                    summary["fingerprint"],
                    json.dumps(dataset.metadata, sort_keys=True, default=str),
                    _now(),
                ),
            )
        return dataset_id

    def record_encoder_config(self, config: Mapping[str, Any]) -> str:
        """Insert (or reuse) an encoder configuration and return its identifier."""
        payload = dict(config)
        encoder_type = str(payload.get("encoder", "unknown"))
        params = {key: value for key, value in payload.items() if key not in {"encoder", "dt_ms"}}
        config_id = "cfg_" + stable_hash(payload, length=16)
        with self._transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO encoder_configs
                   (config_id, encoder_type, dt_ms, params_json, config_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    config_id,
                    encoder_type,
                    payload.get("dt_ms"),
                    json.dumps(params, sort_keys=True, default=str),
                    stable_hash(payload, length=16),
                    _now(),
                ),
            )
        return config_id

    def record_model(self, config: Mapping[str, Any]) -> str:
        """Insert (or reuse) a backend description and return its identifier."""
        payload = dict(config)
        model_id = "mdl_" + stable_hash(payload, length=16)
        with self._transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO models
                   (model_id, backend, config_json, n_parameters, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    model_id,
                    str(payload.get("backend", "unknown")),
                    json.dumps(payload, sort_keys=True, default=str),
                    payload.get("n_parameters"),
                    _now(),
                ),
            )
        return model_id

    # ------------------------------------------------------------------ experiments
    def experiment_id_for(self, result: ExperimentResult) -> str:
        """Deterministic identifier derived from the experiment configuration."""
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

    def record_experiment(self, result: ExperimentResult, dataset: SensorDataset | None = None) -> str:
        """Persist a completed experiment and return its identifier."""
        experiment_id = self.experiment_id_for(result)
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT experiment_id FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            suffix = 2
            while existing is not None:
                candidate = f"{experiment_id}-{suffix}"
                existing = connection.execute(
                    "SELECT experiment_id FROM experiments WHERE experiment_id = ?",
                    (candidate,),
                ).fetchone()
                if existing is None:
                    experiment_id = candidate
                    break
                suffix += 1

            dataset_id = self.record_dataset(dataset) if dataset is not None else None
            config_id = self.record_encoder_config(result.encoder)
            model_id = self.record_model(result.backend)

            result.experiment_id = experiment_id
            best = result.best
            connection.execute(
                """INSERT INTO experiments
                   (experiment_id, dataset_id, config_id, model_id, optimizer, objective, seed,
                    status, n_trials, accuracy_target, spike_budget, duration_s, best_accuracy,
                    best_spike_count, best_index, notes, started_at, finished_at, software_json,
                    payload_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id,
                    dataset_id,
                    config_id,
                    model_id,
                    result.optimizer,
                    result.objective,
                    int(result.seed),
                    result.status,
                    int(result.n_trials),
                    result.accuracy_target,
                    result.spike_budget,
                    float(result.duration_s),
                    None if best is None else float(best.accuracy),
                    None if best is None else int(best.spike_count),
                    result.best_index,
                    result.notes,
                    result.created_at,
                    _now(),
                    json.dumps(result.software, sort_keys=True, default=str),
                    json.dumps(result.to_dict(), sort_keys=True, default=str),
                ),
            )
            for trial in result.trials:
                self._insert_trial(connection, experiment_id, trial)
        return experiment_id

    @staticmethod
    def _insert_trial(connection: sqlite3.Connection, experiment_id: str, trial: TrialResult) -> None:
        connection.execute(
            """INSERT INTO results
               (experiment_id, trial_index, params_json, accuracy, macro_f1, train_accuracy,
                spike_count, spike_rate_hz, information_loss, compute_cost, score, rank,
                is_pareto, feasible, status, error, objectives_json, evaluation_json,
                inference_time_s, train_time_s)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment_id,
                int(trial.index),
                json.dumps(trial.params, sort_keys=True, default=str),
                float(trial.accuracy),
                float(trial.macro_f1),
                trial.train_accuracy,
                int(trial.spike_count),
                float(trial.spike_rate_hz),
                float(trial.information_loss),
                float(trial.compute_cost),
                float(trial.score),
                int(trial.rank),
                int(trial.is_pareto),
                int(trial.feasible),
                trial.status,
                trial.error,
                json.dumps(trial.objectives, sort_keys=True, default=str),
                json.dumps(trial.evaluation, sort_keys=True, default=str),
                float(trial.inference_time_s),
                float(trial.train_time_s),
            ),
        )

    # ------------------------------------------------------------------ queries
    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Return the full experiment payload, or ``None`` when unknown."""
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT payload_json FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["payload_json"])
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RegistryError(f"Experiment {experiment_id} has a corrupt payload.") from exc

    def load_experiment(self, experiment_id: str) -> ExperimentResult:
        """Reconstruct an :class:`ExperimentResult` from the registry."""
        payload = self.get_experiment(experiment_id)
        if payload is None:
            raise RegistryError(f"Experiment {experiment_id!r} was not found in the registry.")
        return ExperimentResult.from_dict(payload)

    def list_experiments(
        self, *, limit: int = 50, status: str | None = None
    ) -> list[dict[str, Any]]:
        """Return experiment summaries, newest first."""
        query = """SELECT experiment_id, dataset_id, optimizer, objective, seed, status, n_trials,
                          accuracy_target, spike_budget, duration_s, best_accuracy,
                          best_spike_count, best_index, notes, started_at, finished_at
                   FROM experiments"""
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY finished_at DESC LIMIT ?"
        params = (*params, int(limit))
        with self._transaction() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_results(self, experiment_id: str) -> list[dict[str, Any]]:
        """Return the trial rows recorded for an experiment."""
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM results WHERE experiment_id = ? ORDER BY trial_index",
                (experiment_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def experiment_count(self, *, status: str | None = None) -> int:
        query = "SELECT COUNT(*) AS n FROM experiments"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        with self._transaction() as connection:
            row = connection.execute(query, params).fetchone()
        return int(row["n"]) if row is not None else 0

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and its trials.  Returns True when something was removed."""
        with self._transaction() as connection:
            exists = connection.execute(
                "SELECT 1 FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            if exists is None:
                return False
            # Children first: ``results`` references ``experiments`` with a FK.
            connection.execute("DELETE FROM results WHERE experiment_id = ?", (experiment_id,))
            connection.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
        return True

    def trial_table(self, experiment_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Concatenate trial rows from several experiments for comparison."""
        rows: list[dict[str, Any]] = []
        for experiment_id in experiment_ids:
            rows.extend(self.get_results(experiment_id))
        return rows

    # ------------------------------------------------------------------ misc
    def __repr__(self) -> str:
        return f"ExperimentRegistry(path={str(self.path)!r})"


