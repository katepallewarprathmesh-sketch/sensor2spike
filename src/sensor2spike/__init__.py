"""The public API of Sensor2Spike.

This module is the contract. Everything named here is supported, documented and
safe to depend on; it changes only with a deprecation cycle. Everything not
named here lives under an underscore-prefixed implementation package and may be
reorganised at any time without notice.

The surface is organised in four layers, from the data you already have to the
artefacts you keep:

==================  ==========================================================
Layer               What you touch
==================  ==========================================================
**Data**            ``SensorDataset`` / ``WindowSet`` / ``SpikeTrain`` and the
                    load-save-preprocess-split functions.
**Encoding**        ``SpikeEncoder`` (the extension contract) plus
                    ``build_encoder`` and the encoder registry.
**Measurement**     ``SNNBackend`` (the other extension contract),
                    ``AutoTuner`` for search, ``ExperimentRegistry`` for
                    provenance, and the report writers.
**Results**         ``TrialResult`` / ``ExperimentResult`` / ``SpikeStats``.
==================  ==========================================================

Typical use::

    from sensor2spike import SensorDataset, AutoTuner

    dataset = SensorDataset.load("recording.npz")
    result = AutoTuner(encoder="rate", n_trials=40).optimize(dataset)
    result.export("run.json")

Extending: subclass ``SpikeEncoder`` or ``SNNBackend``, decorate with
``register_encoder`` / ``register_backend``, and the optimiser, CLI and reports
pick the new strategy up automatically. Nothing else needs to change.

See ``sensor2spike.__all__`` for the authoritative list.
"""

from __future__ import annotations

# -- Data ----------------------------------------------------------------------
from ._data import (
    PreprocessConfig,
    SensorDataset,
    SpikeTrain,
    WindowSet,
    load_dataset,
    preprocess,
    save_dataset,
    split_windows,
)

# -- Encoding ------------------------------------------------------------------
from ._encoding import (
    DeltaEncoder,
    LatencyEncoder,
    PopulationEncoder,
    RateEncoder,
    SpikeEncoder,
    ThresholdEncoder,
    build_encoder,
    encoder_space,
    list_encoders,
    register_encoder,
)

# -- Errors --------------------------------------------------------------------
from ._errors import (
    ConfigurationError,
    DataValidationError,
    DependencyMissingError,
    EncodingError,
    EvaluationError,
    OptimizationError,
    RegistryError,
    ReportError,
    Sensor2SpikeError,
)

# -- Measurement ---------------------------------------------------------------
from ._measurement import (
    AutoTuner,
    ExperimentRegistry,
    ExperimentRunner,
    LIFRidgeBackend,
    ObjectiveWeights,
    OptimizeConfig,
    SNNBackend,
    build_backend,
    compare_results,
    compute_spike_stats,
    export_config,
    export_result,
    generate_report,
    list_backends,
    register_backend,
    render_comparison_markdown,
)

# -- Results -------------------------------------------------------------------
from ._results import (
    EvaluationResult,
    ExperimentResult,
    SpikeStats,
    TrialResult,
)
from ._version import __version__

__all__ = [
    "__version__",
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
]


def __getattr__(name: str) -> object:
    """Fail loudly (and helpfully) on the implementation-package names.

    Before the modules were made private, ``import sensor2spike.encoders`` and
    friends worked. Reaching for them now has to say what to use instead rather
    than raising a bare ``ModuleNotFoundError``.
    """
    replacements = {
        "contracts": "_results / _data (types are re-exported from sensor2spike)",
        "encoders": "_encoding",
        "snn": "_measurement",
        "optimization": "_measurement",
        "experiment": "_measurement",
        "reporting": "_measurement",
        "data": "_data",
        "analysis": "compute_spike_stats is re-exported from sensor2spike",
        "spaces": "_encoding (encoder_space is the supported entry point)",
        "exceptions": "_errors",
        "utils": "no public surface",
    }
    hint = replacements.get(name)
    if hint is not None:
        raise AttributeError(
            f"{name!r} is an implementation detail of Sensor2Spike and is not part "
            f"of the public API. Use {hint}."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
