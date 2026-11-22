"""SNN evaluation backend interface.

The optimisation engine only ever talks to :class:`SNNBackend`, never to a
specific framework.  Backends convert spike trains into features, train a
readout and return an :class:`~sensor2spike.contracts.EvaluationResult`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from ..analysis.metrics import accuracy, classification_summary, macro_f1
from ..contracts import EvaluationResult, SpikeTrain
from ..exceptions import EvaluationError
from ..utils.validation import as_1d_int_array

__all__ = ["SNNBackend"]


class SNNBackend(ABC):
    """Abstract SNN evaluation backend."""

    name: ClassVar[str] = "base"
    description: ClassVar[str] = ""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self.n_classes_: int | None = None
        self.classes_: np.ndarray | None = None

    # ------------------------------------------------------------------ abstract API
    @abstractmethod
    def _features(self, spikes: SpikeTrain) -> np.ndarray:
        """Map one spike train to a fixed-length feature vector."""

    @abstractmethod
    def _fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the readout on feature vectors."""

    @abstractmethod
    def _predict_scores(self, features: np.ndarray) -> np.ndarray:
        """Return class scores of shape ``(n_samples, n_classes)``."""

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Number of trainable parameters in the readout."""

    # ------------------------------------------------------------------ shared API
    def features(self, spikes: Sequence[SpikeTrain]) -> np.ndarray:
        """Extract features for a sequence of spike trains."""
        if not spikes:
            raise EvaluationError("Cannot extract features from an empty sequence.")
        vectors = [np.asarray(self._features(train), dtype=np.float64).ravel() for train in spikes]
        width = vectors[0].shape[0]
        if any(vector.shape[0] != width for vector in vectors):
            raise EvaluationError(
                "All spike trains must produce feature vectors of the same length."
            )
        return np.vstack(vectors)

    def fit(self, spikes: Sequence[SpikeTrain], labels: Any, *, seed: int | None = None) -> SNNBackend:
        """Fit the readout to ``spikes``/``labels``."""
        if seed is not None:
            self.seed = int(seed)
        targets = as_1d_int_array(labels, name="labels", minimum=0)
        if targets.shape[0] != len(spikes):
            raise EvaluationError(
                f"Got {len(spikes)} spike trains but {targets.shape[0]} labels."
            )
        if targets.size == 0:
            raise EvaluationError("Cannot train on an empty dataset.")
        self.classes_ = np.unique(targets)
        if self.classes_.size < 2:
            raise EvaluationError(
                "Training labels must contain at least two classes; "
                f"got {self.classes_.tolist()}."
            )
        self.n_classes_ = int(self.classes_.size)
        features = self.features(spikes)
        if features.shape[0] != targets.shape[0]:
            raise EvaluationError("Feature/label count mismatch.")
        self._fit(features, targets)
        return self

    def predict(self, spikes: Sequence[SpikeTrain]) -> np.ndarray:
        """Return predicted class indices."""
        self._require_fitted()
        scores = self._predict_scores(self.features(spikes))
        return np.argmax(scores, axis=1).astype(np.int64)

    def evaluate(self, spikes: Sequence[SpikeTrain], labels: Any) -> tuple[float, float, np.ndarray, dict[str, Any]]:
        """Return ``(accuracy, macro_f1, predictions, summary)``."""
        targets = as_1d_int_array(labels, name="labels", minimum=0)
        if targets.shape[0] != len(spikes):
            raise EvaluationError("Label count does not match the number of spike trains.")
        predictions = self.predict(spikes)
        # Class indices must stay in the training numbering: an unseen class in
        # the labels would otherwise produce a confusing indexing error.
        unseen = np.setdiff1d(targets, self.classes_)
        if unseen.size:
            raise EvaluationError(
                f"Test labels contain class(es) {unseen.tolist()} that were not present "
                "during training; cannot evaluate."
            )
        summary = classification_summary(targets, predictions, n_classes=self.n_classes_)
        return accuracy(targets, predictions), macro_f1(targets, predictions, n_classes=self.n_classes_), predictions, summary

    def train_evaluate(
        self,
        train_spikes: Sequence[SpikeTrain],
        train_labels: Any,
        test_spikes: Sequence[SpikeTrain],
        test_labels: Any,
        *,
        seed: int | None = None,
    ) -> EvaluationResult:
        """Train on the training split and evaluate on the test split.

        Timing covers feature extraction and readout inference, i.e. the full
        path a deployed model would execute per inference.
        """
        if seed is None:
            seed = self.seed
        self.seed = int(seed)

        start = time.perf_counter()
        self.fit(train_spikes, train_labels, seed=seed)
        train_time = time.perf_counter() - start

        train_accuracy = None
        if train_spikes:
            train_accuracy = accuracy(
                as_1d_int_array(train_labels, name="train_labels", minimum=0),
                self.predict(train_spikes),
            )

        start = time.perf_counter()
        acc, f1, predictions, summary = self.evaluate(test_spikes, test_labels)
        inference_time = time.perf_counter() - start

        return EvaluationResult(
            backend=self.name,
            accuracy=float(acc),
            macro_f1=float(f1),
            n_train=len(train_spikes),
            n_test=len(test_spikes),
            train_time_s=float(train_time),
            inference_time_s=float(inference_time),
            n_parameters=self.n_parameters,
            train_accuracy=None if train_accuracy is None else float(train_accuracy),
            predictions=predictions,
            seed=self.seed,
            extra={"summary": summary, "n_classes": self.n_classes_},
        )

    # ------------------------------------------------------------------ config
    def get_config(self) -> dict[str, Any]:
        """Return a JSON-safe description of the backend."""
        return {"backend": self.name, "seed": self.seed, "n_classes": self.n_classes_}

    def _require_fitted(self) -> None:
        if self.n_classes_ is None:
            raise EvaluationError(f"{self.name}: call fit() before predict().")
