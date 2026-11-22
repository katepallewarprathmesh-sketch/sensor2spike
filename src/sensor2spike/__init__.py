"""Sensor2Spike — sensor-to-spike encoding optimisation for spiking neural networks.

Quick start
-----------
>>> from sensor2spike import SensorDataset, AutoTuner
>>> dataset = SensorDataset.load("radar_recordings.csv", sampling_frequency=1000)
>>> tuner = AutoTuner(objective="multi_objective", encoder="rate")
>>> results = tuner.optimize(dataset=dataset, n_trials=12)
>>> results.export("experiment.json")
"""

from __future__ import annotations

from ._version import __version__
from .analysis import compute_spike_stats
from .contracts import (
    EvaluationResult,
    ExperimentResult,
    SensorDataset,
    SpikeStats,
    SpikeTrain,
    TrialResult,
    WindowSet,
)
from .data import (
    PreprocessConfig,
    load_dataset,
    preprocess,
    save_dataset,
    split_windows,
)
from .encoders import (
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
from .exceptions import (
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
from .experiment import (
    ExperimentRegistry,
    ExperimentRunner,
    compare_results,
    render_comparison_markdown,
)
from .optimization import (
    AutoTuner,
    ObjectiveWeights,
    OptimizeConfig,
)
from .reporting import export_config, export_result, generate_report
from .snn import (
    LIFRidgeBackend,
    SNNBackend,
    build_backend,
    list_backends,
)

__all__ = [
    "__version__",
    # contracts
    "SensorDataset",
    "WindowSet",
    "SpikeTrain",
    "SpikeStats",
    "EvaluationResult",
    "TrialResult",
    "ExperimentResult",
    # data
    "PreprocessConfig",
    "load_dataset",
    "preprocess",
    "save_dataset",
    "split_windows",
    # encoders
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
    # analysis
    "compute_spike_stats",
    # snn
    "SNNBackend",
    "LIFRidgeBackend",
    "build_backend",
    "list_backends",
    # optimisation
    "AutoTuner",
    "OptimizeConfig",
    "ObjectiveWeights",
    # experiment
    "ExperimentRegistry",
    "ExperimentRunner",
    "compare_results",
    "render_comparison_markdown",
    # reporting
    "export_config",
    "export_result",
    "generate_report",
    # errors
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
