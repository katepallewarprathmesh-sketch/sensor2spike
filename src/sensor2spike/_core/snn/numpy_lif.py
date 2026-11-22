"""Reference SNN backend implemented with NumPy only.

The backend is a *leaky integrate-and-fire readout*: spikes are low-pass
filtered with a membrane time constant, and a ridge-regularised linear readout
is trained on the resulting membrane traces.  It has no external dependencies,
is deterministic for a fixed seed, and is fast enough to sit inside an
optimisation loop with hundreds of trials.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..analysis.dsp import exp_smooth
from ..contracts import SpikeTrain
from ..exceptions import ConfigurationError
from ..utils.validation import require_choice, require_range
from .base import SNNBackend
from .registry import register_backend

__all__ = ["LIFRidgeBackend"]

_FEATURE_SETS = ("membrane", "mean_max", "mean_max_rate")


@register_backend
class LIFRidgeBackend(SNNBackend):
    """Leaky-integrate-and-fire feature extractor with a ridge readout.

    Parameters
    ----------
    tau_ms:
        Membrane time constant used to smooth the spike train.
    alpha:
        Ridge regularisation strength.
    feature_set:
        ``"membrane"`` (mean membrane potential per neuron), ``"mean_max"``
        (adds the peak membrane potential) or ``"mean_max_rate"`` (adds the
        per-neuron spike rate as well).
    """

    name = "lif_ridge"
    description = "NumPy LIF membrane features + ridge-regularised linear readout (reference backend)"

    def __init__(
        self,
        *,
        tau_ms: float = 20.0,
        alpha: float = 1.0,
        feature_set: str = "mean_max",
        temporal_bins: int = 10,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.tau_ms = float(require_range(tau_ms, "tau_ms", low=1e-3))
        self.alpha = float(require_range(alpha, "alpha", low=0.0))
        self.feature_set = str(require_choice(feature_set, "feature_set", _FEATURE_SETS))
        self.temporal_bins = int(require_range(temporal_bins, "temporal_bins", low=1, high=64))
        self.weights_: np.ndarray | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None

    # ------------------------------------------------------------------ features
    def _pool(self, trace: np.ndarray) -> np.ndarray:
        """Average the membrane trace into ``temporal_bins`` equal segments.

        The pooled trace gives the linear readout access to the coarse shape of
        the spike pattern (for example a 2 Hz gait modulation), which a plain
        mean/max summary cannot represent.
        """
        n_steps = trace.shape[0]
        bins = min(self.temporal_bins, n_steps)
        edges = np.linspace(0, n_steps, bins + 1).astype(int)
        pooled = np.zeros((bins, trace.shape[1]), dtype=np.float64)
        for index in range(bins):
            start, stop = edges[index], edges[index + 1]
            if stop > start:
                pooled[index] = trace[start:stop].mean(axis=0)
        return pooled

    def _features(self, spikes: SpikeTrain) -> np.ndarray:
        membrane = exp_smooth(spikes.spikes.astype(np.float64), spikes.dt, self.tau_ms / 1000.0)
        if self.feature_set == "membrane":
            return np.concatenate([self._pool(membrane).ravel(), membrane.mean(axis=0)])
        parts = [self._pool(membrane).ravel(), membrane.mean(axis=0), membrane.max(axis=0)]
        if self.feature_set == "mean_max_rate":
            parts.append(spikes.spikes.mean(axis=0, dtype=np.float64))
        return np.concatenate(parts)

    # ------------------------------------------------------------------ training
    def _fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        assert self.n_classes_ is not None
        n_samples, n_features = features.shape
        # Standardise so that ``alpha`` has the same meaning regardless of the
        # encoder's output scale (rate coding and delta modulation differ by
        # orders of magnitude).
        self.feature_mean_ = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        self.feature_scale_ = scale
        features = (features - self.feature_mean_) / self.feature_scale_
        targets = np.zeros((n_samples, self.n_classes_), dtype=np.float64)
        targets[np.arange(n_samples), labels] = 1.0
        design = np.hstack([features, np.ones((n_samples, 1), dtype=np.float64)])
        regulariser = self.alpha * np.eye(design.shape[1], dtype=np.float64)
        regulariser[-1, -1] = 0.0  # do not regularise the bias
        try:
            self.weights_ = np.linalg.solve(design.T @ design + regulariser, design.T @ targets)
        except np.linalg.LinAlgError:
            self.weights_ = np.linalg.pinv(design.T @ design + regulariser) @ design.T @ targets
        if not np.all(np.isfinite(self.weights_)):
            raise ConfigurationError("Ridge readout produced non-finite weights.")

    def _predict_scores(self, features: np.ndarray) -> np.ndarray:
        assert self.weights_ is not None and self.feature_mean_ is not None
        features = (features - self.feature_mean_) / self.feature_scale_
        design = np.hstack([features, np.ones((features.shape[0], 1), dtype=np.float64)])
        return design @ self.weights_

    @property
    def n_parameters(self) -> int:
        return 0 if self.weights_ is None else int(self.weights_.size)

    # ------------------------------------------------------------------ config
    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "tau_ms": self.tau_ms,
            "alpha": self.alpha,
            "feature_set": self.feature_set,
            "temporal_bins": self.temporal_bins,
            "framework": "numpy",
        }

    _FEATURE_SETS: ClassVar[tuple[str, ...]] = _FEATURE_SETS
