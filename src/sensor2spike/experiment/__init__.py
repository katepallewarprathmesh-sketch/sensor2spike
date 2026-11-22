"""Experiment management: registry, orchestration and comparison."""

from __future__ import annotations

from .compare import compare_results, render_comparison_markdown
from .registry import ExperimentRegistry
from .runner import DEFAULT_WINDOW_SECONDS, ExperimentRunner, software_versions

__all__ = [
    "ExperimentRegistry",
    "ExperimentRunner",
    "DEFAULT_WINDOW_SECONDS",
    "software_versions",
    "compare_results",
    "render_comparison_markdown",
]
