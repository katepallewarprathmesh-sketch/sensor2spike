"""Optional PyTorch LIF backend.

This backend trains a small surrogate-gradient spiking network and is intended
for *final* validation of a configuration, not for the inner optimisation loop
(it is orders of magnitude slower than :class:`LIFRidgeBackend`).

Requires the optional ``ml`` extra::

    pip install "sensor2spike[ml]"
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..contracts import EvaluationResult, SpikeTrain
from ..exceptions import ConfigurationError, DependencyMissingError, EvaluationError
from .base import SNNBackend
from .registry import register_backend

__all__ = ["TorchLIFBackend"]


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DependencyMissingError("torch", extra="ml", purpose="the PyTorch LIF backend") from exc
    return torch


@register_backend
class TorchLIFBackend(SNNBackend):
    """A single-hidden-layer LIF network trained with a surrogate gradient."""

    name = "torch_lif"
    description = "PyTorch LIF network with surrogate-gradient training (optional)"

    def __init__(
        self,
        *,
        hidden: int = 64,
        tau_ms: float = 20.0,
        epochs: int = 12,
        lr: float = 1e-3,
        batch_size: int = 32,
        temporal_pool: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.hidden = int(hidden)
        self.tau_ms = float(tau_ms)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.temporal_pool = max(1, int(temporal_pool))
        if self.hidden <= 0 or self.epochs <= 0 or self.batch_size <= 0:
            raise ConfigurationError("hidden, epochs and batch_size must be positive.")
        self._model: Any = None
        self._dt_seconds: float = 0.001
        self._torch: Any = None

    # ------------------------------------------------------------------ features
    def _features(self, spikes: SpikeTrain) -> np.ndarray:
        """Return a pooled spike matrix (kept 2-D for the base class contract)."""
        matrix = spikes.spikes.astype(np.float32)
        if self.temporal_pool > 1:
            trim = (matrix.shape[0] // self.temporal_pool) * self.temporal_pool
            matrix = matrix[:trim].reshape(-1, self.temporal_pool, matrix.shape[1]).max(axis=1)
        return matrix

    def features(self, spikes: Sequence[SpikeTrain]) -> np.ndarray:
        """Return a stacked ``(n_samples, n_steps, n_features)`` tensor.

        The membrane time constant depends on how long each step lasts, so the
        timestep is recorded here and consumed when the model is built.  Spike
        trains with different timesteps cannot share one recurrence.
        """
        if not spikes:
            raise EvaluationError("Cannot extract features from an empty sequence.")
        matrices = [np.asarray(self._features(train), dtype=np.float32) for train in spikes]
        shapes = {matrix.shape for matrix in matrices}
        if len(shapes) != 1:
            raise EvaluationError("All spike trains must share the same pooled shape.")
        steps = {float(train.dt) for train in spikes}
        if len(steps) != 1:
            raise EvaluationError(
                f"All spike trains must share one timestep; got {sorted(steps)} seconds."
            )
        self._dt_seconds = steps.pop()
        return np.stack(matrices)

    # ------------------------------------------------------------------ torch plumbing
    def _build_model(self, n_inputs: int, n_classes: int) -> None:
        torch = _import_torch()
        self._torch = torch

        class _SurrogateSpike(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, membrane: Any) -> Any:
                ctx.save_for_backward(membrane)
                return (membrane > 0.0).to(membrane.dtype)

            @staticmethod
            def backward(ctx: Any, grad_output: Any) -> Any:
                (membrane,) = ctx.saved_tensors
                surrogate = torch.sigmoid(4.0 * membrane)
                return grad_output * surrogate * (1.0 - surrogate) * 4.0

        class _LIFNet(torch.nn.Module):
            def __init__(self, n_in: int, n_hidden: int, n_out: int, decay: float) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(n_in, n_hidden)
                self.readout = torch.nn.Linear(n_hidden, n_out)
                self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))

            def forward(self, inputs: Any) -> Any:
                # inputs: (batch, time, features)
                membrane = torch.zeros(
                    inputs.shape[0], self.fc.out_features, device=inputs.device
                )
                output = torch.zeros(
                    inputs.shape[0], self.readout.out_features, device=inputs.device
                )
                for step in range(inputs.shape[1]):
                    drive = self.fc(inputs[:, step])
                    membrane = membrane * self.decay + (1.0 - self.decay) * drive
                    spikes = _SurrogateSpike.apply(membrane - 1.0)
                    membrane = membrane * (1.0 - spikes)
                    output = output + self.readout(spikes)
                return output / float(inputs.shape[1])

        # Decay follows the actual timestep of the spike trains, not a fixed
        # 1 ms assumption: a 5 ms frame must leak five times faster than a
        # 1 ms frame for the membrane time constant to stay the same.
        dt_seconds = getattr(self, "_dt_seconds", 0.001)
        decay = math.exp(-dt_seconds / (self.tau_ms / 1000.0))
        self._model = _LIFNet(n_inputs, self.hidden, n_classes, decay=decay)

    def _to_tensor(self, features: np.ndarray) -> Any:
        torch = self._torch
        return torch.tensor(features, dtype=torch.float32)

    # ------------------------------------------------------------------ training
    def _fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        assert self.n_classes_ is not None
        torch = _import_torch()
        torch.manual_seed(self.seed)
        if features.ndim != 3:
            raise EvaluationError("TorchLIFBackend expects features of shape (n_samples, steps, features).")
        self._build_model(features.shape[2], self.n_classes_)
        model = self._model
        inputs = self._to_tensor(features)
        targets = torch.tensor(labels, dtype=torch.long)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        n_samples = inputs.shape[0]
        for _ in range(self.epochs):
            permutation = torch.randperm(n_samples)
            for start in range(0, n_samples, self.batch_size):
                index = permutation[start : start + self.batch_size]
                optimiser.zero_grad()
                scores = model(inputs[index])
                loss = criterion(scores, targets[index])
                loss.backward()
                optimiser.step()
        model.eval()

    def _predict_scores(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise EvaluationError("TorchLIFBackend has not been fitted.")
        torch = self._torch
        with torch.no_grad():
            return self._model(self._to_tensor(features)).cpu().numpy()

    @property
    def n_parameters(self) -> int:
        if self._model is None:
            return 0
        return int(sum(parameter.numel() for parameter in self._model.parameters()))

    # ------------------------------------------------------------------ evaluation
    def train_evaluate(
        self,
        train_spikes: Sequence[SpikeTrain],
        train_labels: Any,
        test_spikes: Sequence[SpikeTrain],
        test_labels: Any,
        *,
        seed: int | None = None,
    ) -> EvaluationResult:
        """Train and evaluate with a fixed temporal pooling factor."""
        from ..analysis.metrics import accuracy, macro_f1

        if seed is None:
            seed = self.seed
        self.seed = int(seed)
        train_targets = np.asarray(train_labels, dtype=np.int64)
        test_targets = np.asarray(test_labels, dtype=np.int64)
        self.classes_ = np.unique(train_targets)
        self.n_classes_ = int(self.classes_.size)
        self._fit(self.features(train_spikes), train_targets)
        predictions = np.argmax(self._predict_scores(self.features(test_spikes)), axis=1)
        return EvaluationResult(
            backend=self.name,
            accuracy=float(accuracy(test_targets, predictions)),
            macro_f1=float(macro_f1(test_targets, predictions, n_classes=self.n_classes_)),
            n_train=len(train_spikes),
            n_test=len(test_spikes),
            train_time_s=0.0,
            inference_time_s=0.0,
            n_parameters=self.n_parameters,
            predictions=predictions,
            seed=self.seed,
            extra={"hidden": self.hidden, "epochs": self.epochs, "framework": "torch"},
        )

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "hidden": self.hidden,
            "tau_ms": self.tau_ms,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "temporal_pool": self.temporal_pool,
            "framework": "torch",
        }
