"""Internal: data containers, I/O, preprocessing and windowing.

Not part of the public API. Import from ``sensor2spike`` instead.
"""

from __future__ import annotations

from ._core._contracts import SensorDataset, SpikeTrain, WindowSet
from ._core._spaces import Parameter, SearchSpace
from ._core.data import (
    PreprocessConfig,
    load_dataset,
    preprocess,
    save_dataset,
    split_windows,
)
from ._core.io import load_spike_train, save_spike_train

__all__ = [
    "SensorDataset",
    "SpikeTrain",
    "WindowSet",
    "Parameter",
    "SearchSpace",
    "PreprocessConfig",
    "load_dataset",
    "save_dataset",
    "preprocess",
    "split_windows",
    "save_spike_train",
    "load_spike_train",
]
