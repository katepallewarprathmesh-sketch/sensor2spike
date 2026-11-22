"""Internal: result and statistic containers.

Not part of the public API. Import from ``sensor2spike`` instead.
"""

from __future__ import annotations

from ._core._contracts import (
    EvaluationResult,
    ExperimentResult,
    SpikeStats,
    TrialResult,
)

__all__ = ["EvaluationResult", "ExperimentResult", "TrialResult", "SpikeStats"]
