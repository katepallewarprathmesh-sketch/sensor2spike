"""Surrogate-gradient SNN backend implemented with NumPy only.

The reference backend smooths spikes and solves a ridge regression in closed
form: fast, but the network never adapts to the data.  This backend trains an
actual recurrent spiking network instead — a leaky integrate-and-fire layer whose
spike nonlinearity is differentiated with a surrogate, so gradients flow back
through time and the weights are updated by error backpropagation.

It exists for two reasons:

* **It runs anywhere.** NumPy and nothing else, so continuous integration
  exercises real surrogate-gradient training on every commit.  The PyTorch
  backend implements the same idea but cannot be tested without a torch install.
* **It is honest.** Gradients are computed by hand and verified against finite
  differences in ``tests/test_snn_training.py``; if the derivation is wrong the
  test fails rather than the accuracy quietly degrading.

The network is deliberately small — one recurrent hidden layer — because the
point is to prove the training loop, not to win a benchmark on a toy dataset.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import SpikeTrain
from ..exceptions import EvaluationError
from ..utils.validation import require_range
from .base import SNNBackend
from .registry import register_backend

__all__ = ["NumpySGDBackend"]


def _surrogate(membrane: np.ndarray, sharpness: float) -> np.ndarray:
    """Derivative of the fast-sigmoid surrogate at ``membrane``.

    The forward pass emits ``membrane > 0``, whose true derivative is zero
    almost everywhere and undefined at the threshold.  The surrogate replaces it
    with the derivative of ``sigmoid(sharpness * x)``, which is smooth, peaks at
    the threshold and decays away from it.
    """
    s = 1.0 / (1.0 + np.exp(-sharpness * membrane))
    return sharpness * s * (1.0 - s)


@register_backend
class NumpySGDBackend(SNNBackend):
    """Single-hidden-layer LIF network trained by backpropagation through time.

    Parameters
    ----------
    hidden:
        Number of hidden LIF neurons.
    tau_ms:
        Membrane time constant.  The decay factor is derived from the actual
        timestep of the spike trains, so a 5 ms frame leaks five times faster
        than a 1 ms frame and the time constant stays comparable across
        encoders.
    lr:
        Learning rate for Adam.
    epochs:
        Passes over the training set.
    batch_size:
        Mini-batch size.
    sharpness:
        Slope of the surrogate derivative.
    weight_decay:
        L2 penalty on both weight matrices.
    """

    name = "numpy_sgd"
    description = "NumPy LIF network trained by surrogate-gradient backpropagation (no framework)"

    def __init__(
        self,
        *,
        hidden: int = 24,
        tau_ms: float = 20.0,
        lr: float = 0.05,
        epochs: int = 25,
        batch_size: int = 32,
        sharpness: float = 4.0,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.hidden = int(require_range(hidden, "hidden", low=1, high=512))
        self.tau_ms = float(require_range(tau_ms, "tau_ms", low=1e-3))
        self.lr = float(require_range(lr, "lr", low=1e-6, high=1.0))
        self.epochs = int(require_range(epochs, "epochs", low=1, high=500))
        self.batch_size = int(require_range(batch_size, "batch_size", low=1, high=4096))
        self.sharpness = float(require_range(sharpness, "sharpness", low=0.1, high=50.0))
        self.weight_decay = float(require_range(weight_decay, "weight_decay", low=0.0))
        self._params: dict[str, np.ndarray] = {}
        self._dt_seconds: float = 0.001

    # ------------------------------------------------------------------ features
    def _features(self, spikes: SpikeTrain) -> np.ndarray:
        """Return the raw spike matrix; the recurrence consumes the time axis.

        The base class pools this into a vector, which is not what a recurrent
        network needs, so :meth:`features` is overridden below to keep the
        ``(steps, inputs)`` structure.  This method exists because the contract
        requires it and to keep the two paths consistent.
        """
        return np.asarray(spikes.spikes, dtype=np.float64)

    def features(self, spikes) -> np.ndarray:
        """Stack spike trains into ``(n_samples, n_steps, n_inputs)``.

        Unlike the ridge backend this keeps the full time axis: the recurrence
        needs the order of events, not a pooled summary.
        """
        if not spikes:
            raise EvaluationError("Cannot extract features from an empty sequence.")
        matrices = [np.asarray(train.spikes, dtype=np.float64) for train in spikes]
        shapes = {matrix.shape for matrix in matrices}
        if len(shapes) != 1:
            raise EvaluationError("All spike trains must share the same shape.")
        steps = {float(train.dt) for train in spikes}
        if len(steps) != 1:
            raise EvaluationError(
                f"All spike trains must share one timestep; got {sorted(steps)} seconds."
            )
        self._dt_seconds = steps.pop()
        return np.stack(matrices)

    # ------------------------------------------------------------------ model
    def _build(self, n_inputs: int, n_classes: int, rng: np.random.Generator) -> None:
        # Encoded spikes are sparse (a few percent of slots), so a direct
        # (1 - decay) * spike * W_in drive is far too small to reach threshold
        # and the network never fires. The bias gives every neuron a resting
        # drive just below threshold, so arriving spikes tip it over. Without
        # this the recurrence is silent and the surrogate gradient has nothing
        # to differentiate.
        scale = 1.0 / np.sqrt(max(n_inputs, 1))
        self._params = {
            "W_in": rng.normal(0.0, scale, (n_inputs, self.hidden)),
            "b_h": np.full(self.hidden, 0.5),
            "W_out": rng.normal(0.0, 1.0 / np.sqrt(self.hidden), (self.hidden, n_classes)),
            "b_o": np.zeros(n_classes),
        }

    def _decay(self) -> float:
        return float(np.exp(-self._dt_seconds / (self.tau_ms / 1000.0)))

    def _forward(self, inputs: np.ndarray, *, cache: bool = False):
        """Unroll the LIF recurrence over time.

        ``inputs`` is ``(batch, steps, n_inputs)``.  Returns logits of shape
        ``(batch, n_classes)``; when ``cache`` is set also returns the tensors
        needed by the backward pass.
        """
        W_in, b_h = self._params["W_in"], self._params["b_h"]
        W_out, b_o = self._params["W_out"], self._params["b_o"]
        decay = self._decay()
        batch, steps, _ = inputs.shape

        membrane = np.zeros((batch, self.hidden))
        logits = np.zeros((batch, W_out.shape[1]))
        spikes_all = np.empty((steps, batch, self.hidden))
        membrane_all = np.empty((steps, batch, self.hidden))

        for t in range(steps):
            # Dividing by (1 - decay) normalises the synaptic contribution so a
            # burst of spikes delivers roughly the same charge regardless of the
            # timestep: 250 steps at 1 ms and 50 steps at 5 ms then respond to
            # comparable input evidence.
            drive = inputs[:, t, :] @ W_in / max(1.0 - decay, 1e-6) + b_h
            membrane = decay * membrane + (1.0 - decay) * drive
            spikes = (membrane > 0.0).astype(np.float64)
            # Hard reset: a fired neuron is returned to rest, which keeps the
            # membrane bounded over long sequences.
            pre_reset = membrane
            membrane = membrane * (1.0 - spikes)
            spikes_all[t] = spikes
            membrane_all[t] = pre_reset
            logits += spikes @ W_out
        logits = logits / max(steps, 1) + b_o

        if not cache:
            return logits
        return logits, (inputs, spikes_all, membrane_all)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def _backward(self, cache, logits: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        """Backpropagation through the unrolled recurrence.

        Per timestep, with elementwise products:

            pre_t   = decay * post_{t-1} + (1 - decay) * drive_t
            spike_t = H(pre_t - 1)                     forward: hard threshold
            post_t  = pre_t * (1 - spike_t)            hard reset

        Because ``spike_t`` is a step function, the chain rule is taken with
        ``d spike_t / d pre_t = surrogate(pre_t - 1)`` -- a *straight-through
        substitution*: the reset multiply keeps its factor ``(1 - spike_t)``,
        but the extra term the true reset derivative would add
        (``-pre_t * d spike_t``) is dropped, because the surrogate is standing in
        for a derivative that is zero almost everywhere.  Keeping that term
        double-counts the threshold.

        Gradient arrives at ``post_t`` from two places: the readout at time t,
        and the recurrence at t+1.  So::

            grad_post = grad_spikes[t] + grad_post_next
            grad_pre  = grad_post * (1 - spike_t) + grad_spikes[t] * surrogate
            grad_prev = grad_pre * decay
            grad_drive= grad_pre * (1 - decay)
        """
        inputs, spikes_all, pre_all = cache
        W_out = self._params["W_out"]
        batch, steps, _ = inputs.shape
        decay = self._decay()

        grad_logits = self._softmax(logits)
        grad_logits[np.arange(batch), labels] -= 1.0
        grad_logits /= batch

        grad_W_out = np.zeros_like(W_out)
        grad_spikes = np.empty((steps, batch, self.hidden))
        for t in range(steps):
            grad_W_out += spikes_all[t].T @ grad_logits / max(steps, 1)
            grad_spikes[t] = grad_logits @ W_out.T / max(steps, 1)
        grad_b_o = grad_logits.sum(axis=0)

        grad_W_in = np.zeros_like(self._params["W_in"])
        grad_b_h = np.zeros_like(self._params["b_h"])
        grad_post_next = np.zeros((batch, self.hidden))

        for t in range(steps - 1, -1, -1):
            spike = spikes_all[t]
            pre = pre_all[t]
            grad_post = grad_spikes[t] + grad_post_next
            grad_pre = grad_post * (1.0 - spike) + grad_spikes[t] * _surrogate(
                pre, self.sharpness
            )
            grad_drive = grad_pre * (1.0 - decay)
            grad_post_next = grad_pre * decay
            grad_W_in += inputs[:, t, :].T @ grad_drive
            grad_b_h += grad_drive.sum(axis=0)

        return {"W_in": grad_W_in, "b_h": grad_b_h, "W_out": grad_W_out, "b_o": grad_b_o}

    # ------------------------------------------------------------------ training
    def _fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        assert self.n_classes_ is not None
        if features.ndim != 3:
            raise EvaluationError(
                "NumpySGDBackend expects features of shape (n_samples, steps, inputs)."
            )
        rng = np.random.default_rng(self.seed)
        self._build(features.shape[2], self.n_classes_, rng)

        n_samples = features.shape[0]
        # Adam state.
        moment1 = {k: np.zeros_like(v) for k, v in self._params.items()}
        moment2 = {k: np.zeros_like(v) for k, v in self._params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        for _ in range(self.epochs):
            order = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                index = order[start : start + self.batch_size]
                if index.size < 2:
                    continue
                batch = features[index]
                targets = labels[index]
                logits, cache = self._forward(batch, cache=True)
                grads = self._backward(cache, logits, targets)
                step += 1
                for key, grad in grads.items():
                    if self.weight_decay and key in ("W_in", "W_out"):
                        grad = grad + self.weight_decay * self._params[key]
                    moment1[key] = beta1 * moment1[key] + (1 - beta1) * grad
                    moment2[key] = beta2 * moment2[key] + (1 - beta2) * grad * grad
                    m_hat = moment1[key] / (1 - beta1**step)
                    v_hat = moment2[key] / (1 - beta2**step)
                    self._params[key] = self._params[key] - self.lr * m_hat / (np.sqrt(v_hat) + eps)
        self.training_curve_ = [self._loss(features, labels)]

    def _loss(self, features: np.ndarray, labels: np.ndarray) -> float:
        logits = self._forward(features)
        probs = self._softmax(logits)
        picked = np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0)
        return float(-np.mean(np.log(picked)))

    def _predict_scores(self, features: np.ndarray) -> np.ndarray:
        if not self._params:
            raise EvaluationError("NumpySGDBackend has not been fitted.")
        return self._forward(features)

    @property
    def n_parameters(self) -> int:
        return int(sum(parameter.size for parameter in self._params.values()))

    # ------------------------------------------------------------------ config
    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "hidden": self.hidden,
            "tau_ms": self.tau_ms,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "sharpness": self.sharpness,
            "weight_decay": self.weight_decay,
            "framework": "numpy",
        }

