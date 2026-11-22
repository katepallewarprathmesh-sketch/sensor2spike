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
    NormalizationStats,
    PreprocessConfig,
    SensorDataset,
    SpikeTrain,
    WindowSet,
    export_event_frames,
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
    TorchLIFBackend,
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
    "export_event_frames",
    "NormalizationStats",
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
]


def __getattr__(name: str) -> object:
    """Fail loudly (and helpfully) on the implementation-package names.

    Before the modules were made private, ``import sensor2spike.encoders`` and
    friends worked. Reaching for them now has to say what to use instead rather
    than raising a bare ``ModuleNotFoundError`` — and "what to use" always means
    a name in ``__all__``, never another private module.
    """
    replacements = {
        "contracts": "SensorDataset, WindowSet, SpikeTrain, TrialResult and "
        "ExperimentResult",
        "encoders": "SpikeEncoder, register_encoder, build_encoder, "
        "list_encoders and the five built-in encoder classes",
        "snn": "SNNBackend, register_backend, build_backend, list_backends and "
        "LIFRidgeBackend",
        "optimization": "AutoTuner, OptimizeConfig, ObjectiveWeights and "
        "TrialResult",
        "experiment": "ExperimentRunner, ExperimentRegistry, ExperimentResult "
        "and compare_results",
        "reporting": "generate_report, render_comparison_markdown, export_result "
        "and export_config",
        "data": "SensorDataset, load_dataset, save_dataset, preprocess, "
        "PreprocessConfig and split_windows",
        "analysis": "compute_spike_stats, SpikeStats and EvaluationResult",
        "spaces": "encoder_space (the supported way to describe a search space)",
        "exceptions": "Sensor2SpikeError and its subclasses: DataValidationError, "
        "ConfigurationError, EncodingError, EvaluationError, OptimizationError, "
        "RegistryError, ReportError and DependencyMissingError",
        "utils": "nothing — that module was internal only, and none of its "
        "helpers are part of the public API",
    }
    hint = replacements.get(name)
    if hint is not None:
        raise AttributeError(
            f"{name!r} is an implementation detail of Sensor2Spike and is not "
            f"part of the public API. Use {hint}."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
