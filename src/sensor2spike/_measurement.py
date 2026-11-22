"""Internal: evaluation backends, search, registry and reporting.

Not part of the public API. Import from ``sensor2spike`` instead.
"""

from __future__ import annotations

from ._core._analysis import compute_spike_stats
from ._core._experiment import (
    ExperimentRegistry,
    ExperimentRunner,
    compare_results,
    render_comparison_markdown,
)
from ._core._optimization import AutoTuner, ObjectiveWeights, OptimizeConfig
from ._core._reporting import export_config, export_result, generate_report
from ._core._snn import (
    LIFRidgeBackend,
    SNNBackend,
    build_backend,
    list_backends,
    register_backend,
)

__all__ = [
    "SNNBackend",
    "LIFRidgeBackend",
    "build_backend",
    "list_backends",
    "register_backend",
    "AutoTuner",
    "OptimizeConfig",
    "ObjectiveWeights",
    "ExperimentRegistry",
    "ExperimentRunner",
    "compare_results",
    "render_comparison_markdown",
    "generate_report",
    "export_result",
    "export_config",
    "compute_spike_stats",
]
