# ADR-003: Multi-objective optimisation with Pareto ranking by default

- **Status:** Accepted

## Context

The naive formulation is a single weighted score:

```
J = w_a·(1 − accuracy) + w_s·S + w_l·L + w_c·C
```

It is easy to rank and easy to explain. It is also easy to mislead: two
configurations with the same `J` can be wildly different designs — one accurate
and expensive, one cheap and sloppy — and the weights silently encode a
preference the user never examined. Worse, `J` changes meaning when an encoder
changes the magnitude of `S` or `L`, so comparing across encoders is not
apples-to-apples.

## Decision

Make non-dominated sorting the default (`objective="multi_objective"`), with the
weighted sum retained as an option (`objective="weighted"`) and as the internal
tie-break signal `score = -J`.

- Each trial carries four objectives, all normalised against the batch's own
  extremes: `1−accuracy`, spike activity, information loss, compute cost.
- `rank_trials` applies non-dominated sorting (rank 1 = Pareto front) and
  crowding distance within each front, so results are ordered *and* diverse.
- The recommended configuration is then selected with an explicit rule
  (`balanced` by default: within `accuracy_tolerance` of the best accuracy, take
  the fewest spikes), never by a hidden weight.
- Hard constraints (`accuracy_target`, `spike_budget`) mark trials `feasible`,
  and infeasible trials are excluded from selection when feasible ones exist.

## Consequences

**Good**

- The report shows the trade-off curve; the user picks with their own priorities.
- Ties are broken by an explicit, documented rule instead of by weight drift.
- Comparisons across encoders use the same normalised axes.

**Bad**

- More to explain than "lower is better": the user must read the Pareto front.
- Normalisation is batch-relative, so an `S` of 1.0 means "sparsest in this run",
  not an absolute rate. Reports state this.

## Revisit when

Users routinely want a single headline number *and* fast ranking. The `weighted`
mode and `score` field already cover that; promotion to the default would need
evidence that Pareto output is being ignored in practice.
