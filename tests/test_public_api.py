"""Contract tests for the public API boundary.

The package promises a small supported surface and hides everything else. These
tests fail if that promise erodes — either by leaking implementation names into
the top level, or by letting one of the documented names drift.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import sensor2spike

DOCUMENTED = {
    # Data
    "SensorDataset",
    "WindowSet",
    "SpikeTrain",
    "load_dataset",
    "save_dataset",
    "preprocess",
    "split_windows",
    "PreprocessConfig",
    # Encoding
    "SpikeEncoder",
    "RateEncoder",
    "LatencyEncoder",
    "ThresholdEncoder",
    "DeltaEncoder",
    "PopulationEncoder",
    "build_encoder",
    "encoder_space",
    "list_encoders",
    "register_encoder",
    # Measurement
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
    # Results
    "EvaluationResult",
    "ExperimentResult",
    "TrialResult",
    "SpikeStats",
    # Errors
    "Sensor2SpikeError",
    "DataValidationError",
    "ConfigurationError",
    "EncodingError",
    "EvaluationError",
    "OptimizationError",
    "RegistryError",
    "ReportError",
    "DependencyMissingError",
}


# ------------------------------------------------------------------- the surface
def test_every_documented_name_exists():
    missing = sorted(name for name in DOCUMENTED if not hasattr(sensor2spike, name))
    assert not missing, f"documented but missing from the package: {missing}"


def test_all_matches_the_documented_surface():
    exported = set(sensor2spike.__all__) - {"__version__"}
    assert exported == DOCUMENTED, (
        f"undocumented exports: {sorted(exported - DOCUMENTED)}; "
        f"documented but unexported: {sorted(DOCUMENTED - exported)}"
    )


def test_no_dunder_leaks_in_all():
    assert all(not name.startswith("_") for name in sensor2spike.__all__ if name != "__version__")


def test_documented_names_are_importable_by_name():
    namespace: dict[str, object] = {}
    statement = "from sensor2spike import " + ", ".join(sorted(DOCUMENTED))
    exec(statement, namespace)  # noqa: S102 - the point is to exercise the import line
    assert DOCUMENTED <= set(namespace)


# --------------------------------------------------------------- hidden internals
@pytest.mark.parametrize(
    "module",
    [
        "sensor2spike.contracts",
        "sensor2spike.encoders",
        "sensor2spike.snn",
        "sensor2spike.optimization",
        "sensor2spike.experiment",
        "sensor2spike.reporting",
        "sensor2spike.data",
        "sensor2spike.analysis",
        "sensor2spike.spaces",
        "sensor2spike.exceptions",
        "sensor2spike.utils",
    ],
)
def test_implementation_packages_are_not_importable(module: str):
    with pytest.raises(ModuleNotFoundError):
        __import__(module)


def test_attribute_access_explains_where_to_look_instead():
    with pytest.raises(AttributeError) as excinfo:
        _ = sensor2spike.encoders  # noqa: B009 - intentional attribute access
    message = str(excinfo.value)
    assert "_encoding" in message
    assert "implementation detail" in message


def test_unknown_attribute_raises_plain_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = sensor2spike.definitely_not_a_thing  # noqa: B009


# ----------------------------------------------------- the package works standalone
def test_public_import_does_not_pull_in_optional_dependencies():
    """The core install is numpy+scipy only; importing must not need torch etc."""
    code = (
        "import sys; import sensor2spike; "
        "heavy = [m for m in ('torch','optuna','streamlit','pandas','matplotlib','sklearn') "
        "if m in sys.modules]; "
        "assert not heavy, f'optional dependency imported eagerly: {heavy}'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_version_is_semver_ish():
    parts = sensor2spike.__version__.split(".")
    assert len(parts) >= 2 and all(part.isdigit() for part in parts[:2])
