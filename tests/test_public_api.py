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
    "TorchLIFBackend",
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
    """The message must name the public thing to use, not the private module."""
    with pytest.raises(AttributeError) as excinfo:
        _ = sensor2spike.encoders  # noqa: B009 - intentional attribute access
    message = str(excinfo.value)
    assert "implementation detail" in message
    assert "SpikeEncoder" in message
    assert "_encoding" not in message


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


# --------------------------------------------------- migration hints stay honest
def test_migration_hints_name_real_public_entities():
    """A hint that points at a rename, a typo or a private module is worse than
    no hint: it sends the reader in a circle. Every identifier a hint mentions
    must be in ``__all__``, and no hint may recommend a private module."""
    import inspect
    import re

    source = inspect.getsource(sensor2spike.__getattr__)
    body = source[source.index("replacements = {") : source.index("hint = replacements.get")]

    recommended = set()
    for literal in re.findall(r'"([^"]*)"', body):
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", literal):
            if token in {"the", "a", "an", "and", "its", "of", "not", "part", "public"}:
                continue
            recommended.add(token)

    unknown = {
        name
        for name in recommended
        if name not in sensor2spike.__all__ and not name.startswith("__")
    }
    # Prose words that happen to look like identifiers are allowed, but a name
    # that looks like a class or a snake_case API (e.g. ``parse_search_space``)
    # is not.
    api_shaped = {
        name
        for name in unknown
        if not name.isupper()  # allow acronyms such as API in prose
        and (name[0].isupper() or (name.islower() and "_" in name))
    }
    assert not api_shaped, f"hints recommend names that are not public: {sorted(api_shaped)}"


def test_migration_hints_never_point_at_private_modules():
    import inspect

    source = inspect.getsource(sensor2spike.__getattr__)
    assert "_core" not in source, "a hint must not send users to the private core"


# ------------------------------------- not even a path to the internals escapes
def test_listed_class_paths_are_public_import_paths():
    """``list_encoders`` / ``list_backends`` feed the CLI table and report JSON.

    Printing ``sensor2spike._core.encoders.delta.DeltaEncoder`` there would leak
    the private layout to anyone who reads the output, so the path must be the
    public one — and it must actually resolve.
    """
    from sensor2spike import list_backends, list_encoders

    for entry in [*list_encoders(), *list_backends()]:
        path = entry["class"]
        assert "_core" not in path, f"{entry['name']} leaks a private path: {path}"
        module_name, _, attr = path.rpartition(".")
        module = __import__(module_name, fromlist=[attr])
        assert hasattr(module, attr), f"{path} does not resolve"


def test_encoder_errors_do_not_leak_private_modules():
    """Validation messages are user-facing: they must not name private modules."""
    from sensor2spike import ConfigurationError, build_encoder

    with pytest.raises(ConfigurationError) as excinfo:
        build_encoder("nope")
    message = str(excinfo.value)
    assert "_core" not in message
    assert "rate" in message  # the available encoders are listed
