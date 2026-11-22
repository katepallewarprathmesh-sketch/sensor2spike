# ADR-004: Pluggable SNN backend abstraction

- **Status:** Accepted

## Context

The SNN ecosystem moves fast: SpikingJelly, Norse, snnTorch, Lava and bare
PyTorch all exist, and none is obviously permanent. Hard-coding one would tie the
product's useful life to that library's. They also disagree on data layout, time
stepping and training loop conventions.

## Decision

Define `SNNBackend` as a narrow ABC in `snn/base.py`:

```python
class SNNBackend(ABC):
    def _features(self, spikes: SpikeTrain) -> np.ndarray: ...   # one train -> vector
    def _fit(self, features, labels) -> None: ...                # train the readout
    def _predict_scores(self, features) -> np.ndarray: ...        # class scores
    @property
    def n_parameters(self) -> int: ...                            # complexity report
```

The base class owns everything else — validation, feature stacking, metrics,
timing, `EvaluationResult`, `train_evaluate`. A backend implements four small
things and inherits consistent behaviour, including the class-numbering checks
that keep metrics honest.

Backends register through `@register_backend` and are built by name
(`build_backend("lif_ridge", tau_ms=20)`), exactly like encoders.

## Consequences

**Good**

- Adding SpikingJelly is one class plus one decorator; nothing else moves.
- Every backend produces the same `EvaluationResult`, so reports and the registry
  need no special cases.
- The abstract-method check in `register_backend` rejects an incomplete class at
  import time instead of failing mid-experiment.

**Bad**

- The abstraction forces feature-vector output, which fits a readout-style
  classifier but not every architecture (e.g. a fully recurrent end-to-end
  network). Those need either a richer contract or an adapter that still exposes
  a feature vector.
- Two levels of API (`_features`/`_fit`/`_predict_scores` versus `fit`/
  `predict`) take a moment to learn.

## Revisit when

A backend cannot express its architecture through the feature-vector seam — for
example a stateful streaming model that must consume windows in order.
