"""Digital signal processing helpers shared by encoders and metrics."""

from __future__ import annotations

import math

import numpy as np
from scipy import signal as scipy_signal

from ..exceptions import ConfigurationError, EncodingError

__all__ = ["exp_smooth", "resample_to_steps", "reconstruction_r2"]


def resample_to_steps(data: np.ndarray, fs: float, dt: float, n_steps: int) -> np.ndarray:
    """Linearly resample ``(n_samples, n_channels)`` onto the spike time base."""
    array = np.asarray(data, dtype=np.float64)
    n_samples = array.shape[0]
    if n_samples == n_steps and abs(fs * dt - 1.0) < 1e-9:
        return array
    t_in = np.arange(n_samples, dtype=np.float64) / fs
    t_out = np.arange(n_steps, dtype=np.float64) * dt
    out = np.empty((n_steps, array.shape[1]), dtype=np.float64)
    for channel in range(array.shape[1]):
        out[:, channel] = np.interp(t_out, t_in, array[:, channel])
    return out


def exp_smooth(values: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Causal exponential smoothing with time constant ``tau`` (seconds)."""
    if tau <= 0:
        raise ConfigurationError("tau must be positive.")
    alpha = 1.0 - math.exp(-dt / tau)
    return scipy_signal.lfilter([alpha], [1.0, -(1.0 - alpha)], values, axis=0)


def reconstruction_r2(decoded: np.ndarray, target: np.ndarray) -> float:
    """Squared Pearson correlation, averaged over channels.

    An affine (gain + offset) correction is implicit, which keeps the metric
    fair for encoders whose decoder has an arbitrary output scale (threshold,
    delta modulation).  Returns 0.0 when either signal is constant.
    """
    decoded = np.asarray(decoded, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if decoded.shape != target.shape:
        raise EncodingError(
            f"Decoded signal shape {decoded.shape} does not match reference shape {target.shape}."
        )
    scores: list[float] = []
    for channel in range(target.shape[1]):
        x = decoded[:, channel]
        y = target[:, channel]
        if x.std() < 1e-12 or y.std() < 1e-12:
            scores.append(0.0)
            continue
        correlation = float(np.corrcoef(x, y)[0, 1])
        scores.append(correlation * correlation)
    return float(np.mean(scores)) if scores else 0.0
