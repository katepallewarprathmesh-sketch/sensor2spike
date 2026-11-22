"""Implementation: experiment runner, SQLite registry and comparison."""

from .experiment import (
    ExperimentRegistry,
    ExperimentRunner,
    compare_results,
    render_comparison_markdown,
)

__all__ = [
    "ExperimentRegistry",
    "ExperimentRunner",
    "compare_results",
    "render_comparison_markdown",
]
