# ADR-007: Sigma-delta error feedback for the delta encoder

- **Status:** Accepted (supersedes an earlier open-loop implementation)

## Context

The delta encoder is the one whose documentation makes a strong claim:
"decoding is a cumulative sum, so the signal can be reconstructed". The first
implementation did not honour it.

It thresholded the sample-to-sample difference:

```python
delta = np.diff(values, axis=0, prepend=values[:1, :])
up   = delta >=  threshold
down = delta <= -threshold
```

and decoded with `np.cumsum((up - down) * threshold)`.

Two things go wrong. First, every change smaller than `threshold` is discarded
outright. Second, and worse, the *direction* is lost: a slow rise of
`0.9·threshold` per sample never fires, while the decoder keeps integrating from a
level the encoder never corrected. The error accumulates without bound.

Measured on a two-tone band-limited signal, this version scored R² ≈ 0.30 — the
decoder's output wandered away from the signal — while reporting a healthy spike
count. The information-loss term in the objective was, correctly, punishing it,
but the encoder's stated contract was simply false.

## Decision

Implement the encoder as a sigma-delta (error-feedback) modulator:

```python
tracked = self.baseline_.copy()
for step in range(n_steps):
    for _ in range(max_steps_per_sample):
        error = values[step] - tracked
        up, down = error >= threshold, error <= -threshold
        if not (up.any() or down.any()):
            break
        spikes[step, 0::2] |= up
        spikes[step, 1::2] |= down
        tracked += (up - down) * threshold
```

The encoder maintains the same reconstruction the decoder will build, and only
emits a spike when the residual reaches one quantisation step, then moves the
tracked value by exactly that step. The decoder is unchanged — cumulative sum —
because now it is the exact inverse.

The baseline is *learned* during `fit` (`baseline_ = normalized_signal[0]`) so
both sides start integration at the same level; that removes the offset error and
makes a flat signal cost zero spikes. It is persisted in `get_state` and restored
via the `_restore_state` hook.

`max_steps_per_sample` (1–8) bounds the spike budget: if the signal slews faster
than one step per sample can follow, the modulator saturates and the
reconstruction lags. That limitation is real and the information-loss term
surfaces it rather than hiding it.

## Consequences

**Good**

- R² = 0.9984 on a band-limited two-tone signal at ~500 spikes per 2 s.
- A constant signal produces zero spikes, which is what "spikes encode change"
  should mean.
- The accuracy/sparsity trade-off is now genuinely controlled by `threshold`, so
  the optimiser's search finds a meaningful curve.
- The documented claim and the implementation agree, and a regression test
  (`test_delta_encoder_reconstructs_band_limited_signal`) holds them together.

**Bad**

- Broadband noise is not recoverable at a bounded step budget — measured
  information loss rises to 0.48–0.83 depending on the step size. This is inherent
  to a first-order modulator at a fixed spike budget, not a bug; a higher-order
  modulator or a larger budget would be needed.
- Slightly more code than `np.diff` thresholding, and the loop is per-sample in
  Python (vectorised across channels, but not across time).
- The learned baseline is extra state to persist, which is why the `_restore_state`
  hook exists in the base class.

## Alternatives considered

- **Open-loop thresholding with a correction term at decode time.** Pushes the
  complexity onto every decoder and still cannot recover the discarded
  sub-threshold increments.
- **Adaptive threshold.** Helps on some signals, but adds a parameter that is
  hard to explain and interacts with the search space; rejected for now.

## Revisit when

A second-order modulator (or vectorised time loop) is wanted. The contract — 
partial sums reconstruct the signal — should be the invariant either way.
