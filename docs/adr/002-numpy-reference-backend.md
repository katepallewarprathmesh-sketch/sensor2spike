# ADR-002: A NumPy LIF reference backend as the default

- **Status:** Accepted

## Context

The optimiser needs an accuracy signal. Training a real SNN with surrogate
gradients (PyTorch/SpikingJelly/Norse) is minutes-to-hours per configuration, so
using one as the inner loop of a search is impractical: a 40-trial search would
take days. But reporting accuracy from a non-spiking model would make the tool's
central number meaningless — the whole point is to evaluate *spike* encodings.

## Decision

Ship `lif_ridge` as the default backend: the spike train drives leaky
integrate-and-fire membrane dynamics (`exp_smooth` with a configurable `tau_ms`),
the membrane trace is pooled into temporal bins and summarised as mean/max, and a
ridge-regularised linear readout classifies it. Features are standardised before
the readout so `alpha` means the same thing for every encoder's output scale.

This is genuinely LIF-based — the leaky integration is the LIF membrane equation
solved analytically — while staying closed-form and deterministic.

## Consequences

**Good**

- Seconds per trial, so search is interactive.
- Fully deterministic; no torch seed behaviour to chase.
- No 2 GB dependency for the default install.
- Works on CPU-only machines, which is what a data scientist's laptop is.

**Bad**

- It underestimates what a trained SNN could do, and cannot discover temporal
  codes a learned network would exploit. The pooled features are a deliberate
  summary, not a learned representation.
- Its accuracy depends on feature engineering choices (`feature_set`,
  `temporal_bins`) that a trained network would learn instead.

**Mitigation**

- Every report names the backend and states that it is a reference readout.
- `TorchLIFBackend` exists for the final validation pass; `DependencyMissingError`
  gives the install command rather than an `ImportError` traceback.
- The backend is a registry entry, so a SpikingJelly/Norse backend is an
  additive change.

## Revisit when

A trained backend becomes cheap enough to run inside the search loop (e.g. a
pre-trained feature extractor shared across trials).
