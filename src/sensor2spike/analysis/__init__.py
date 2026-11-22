"""Analysis layer: signal metrics, spike statistics and classification scores."""

from __future__ import annotations

from .dsp import exp_smooth, reconstruction_r2, resample_to_steps
from .metrics import (
    accuracy,
    classification_summary,
    confusion_matrix,
    macro_f1,
    per_class_accuracy,
)
from .spikes import SPIKE_COST, STEP_COST, compute_cost, compute_spike_stats

__all__ = [
    "exp_smooth",
    "reconstruction_r2",
    "resample_to_steps",
    "accuracy",
    "classification_summary",
    "confusion_matrix",
    "macro_f1",
    "per_class_accuracy",
    "SPIKE_COST",
    "STEP_COST",
    "compute_cost",
    "compute_spike_stats",
]
