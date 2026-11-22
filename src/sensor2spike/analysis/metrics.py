"""Classification metrics implemented with NumPy only.

Keeping these dependency-free means the optimiser can score thousands of
candidate configurations without importing scikit-learn.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import DataValidationError
from ..utils.validation import as_1d_int_array

__all__ = [
    "accuracy",
    "confusion_matrix",
    "macro_f1",
    "per_class_accuracy",
    "classification_summary",
]


def _check_pair(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    truth = as_1d_int_array(y_true, name="y_true", minimum=0)
    prediction = as_1d_int_array(y_pred, name="y_pred", minimum=0)
    if truth.shape != prediction.shape:
        raise DataValidationError(
            f"y_true and y_pred must have the same length; got {truth.shape[0]} and {prediction.shape[0]}."
        )
    if truth.size == 0:
        raise DataValidationError("Cannot compute metrics on an empty set.")
    return truth, prediction


def confusion_matrix(y_true: Any, y_pred: Any, n_classes: int | None = None) -> np.ndarray:
    """Return a ``C x C`` matrix where rows are true classes and columns predicted."""
    truth, prediction = _check_pair(y_true, y_pred)
    if n_classes is None:
        n_classes = int(max(truth.max(), prediction.max())) + 1
    highest = int(max(truth.max(), prediction.max()))
    if highest >= n_classes:
        raise DataValidationError(
            f"Observed class index {highest} but n_classes={n_classes}; "
            "class indices must be in [0, n_classes)."
        )
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    return matrix


def accuracy(y_true: Any, y_pred: Any) -> float:
    truth, prediction = _check_pair(y_true, y_pred)
    return float(np.mean(truth == prediction))


def per_class_accuracy(y_true: Any, y_pred: Any, n_classes: int | None = None) -> np.ndarray:
    matrix = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    support = matrix.sum(axis=1)
    return np.divide(
        np.diag(matrix),
        support,
        out=np.zeros(matrix.shape[0], dtype=np.float64),
        where=support > 0,
    )


def macro_f1(y_true: Any, y_pred: Any, n_classes: int | None = None) -> float:
    """Macro-averaged F1 score over the classes present in ``y_true``."""
    truth, prediction = _check_pair(y_true, y_pred)
    if n_classes is None:
        n_classes = int(max(truth.max(), prediction.max())) + 1
    matrix = confusion_matrix(truth, prediction, n_classes=n_classes)
    tp = np.diag(matrix).astype(np.float64)
    fp = matrix.sum(axis=0).astype(np.float64) - tp
    fn = matrix.sum(axis=1).astype(np.float64) - tp
    denominator = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    present = matrix.sum(axis=1) > 0
    if not np.any(present):
        return 0.0
    return float(f1[present].mean())


def classification_summary(y_true: Any, y_pred: Any, n_classes: int | None = None) -> dict[str, Any]:
    truth, prediction = _check_pair(y_true, y_pred)
    if n_classes is None:
        n_classes = int(max(truth.max(), prediction.max())) + 1
    return {
        "n_samples": int(truth.size),
        "n_classes": int(n_classes),
        "accuracy": accuracy(truth, prediction),
        "macro_f1": macro_f1(truth, prediction, n_classes=n_classes),
        "per_class_accuracy": per_class_accuracy(truth, prediction, n_classes=n_classes).tolist(),
        "confusion_matrix": confusion_matrix(truth, prediction, n_classes=n_classes).tolist(),
    }
