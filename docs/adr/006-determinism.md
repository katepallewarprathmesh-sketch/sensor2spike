# ADR-006: Determinism as a first-class requirement

- **Status:** Accepted

## Context

A tool that "optimises" is a tool whose output you have to trust. If re-running
the same configuration yields a different accuracy, then a difference between two
encoders might be noise from the tool itself rather than a property of the data.
Users compare runs; an unstable harness makes every comparison suspect. The
specification calls this out as a reliability requirement.

This turned out to matter in practice: during development, Poisson-mode rate
coding gave different accuracies on identical reruns because the encoder's RNG was
constructed without a seed derived from the experiment.

## Decision

Make reproducibility a property the test suite enforces, not a hopeful goal.

1. **Seed everything from the experiment.** Each trial builds its encoder with
   `seed = config.seed + trial_index`. Nothing reads global RNG state.
2. **Seed the backend too.** The ridge readout receives the experiment seed, so
   any sampling inside it is stable.
3. **Never rely on dict or set iteration order** for anything that affects a
   number.
4. **Record what you ran.** Software versions, dataset fingerprint, windowing,
   split sizes and preprocessing are stored with the result; the deterministic
   experiment id is derived from them.
5. **Test it.** `test_runner_is_deterministic_for_fixed_seed`,
   `test_experiment_can_be_reproduced_bit_for_bit` and
   `test_reproducibility_does_not_depend_on_the_process_rng` assert equality of
   the actual measurements.

```python
a = AutoTuner(encoder="rate", n_trials=3, seed=11, window_seconds=0.2).optimize(ds)
b = AutoTuner(encoder="rate", n_trials=3, seed=11, window_seconds=0.2).optimize(ds)
assert [t.accuracy for t in a.trials] == [t.accuracy for t in b.trials]
```

## Consequences

**Good**

- Reruns reproduce exactly, so A/B comparisons mean something.
- Failures are reproducible too: a `failed` row carries a redacted error string
  and the configuration that produced it.
- Different seeds still explore differently, which is what you want from a search.

**Bad**

- Poisson encoders are deterministic *given a seed*; the same configuration with a
  different seed legitimately differs. Reports print the seed for this reason.
- Threading or multiprocessing across trials must derive per-trial seeds
  explicitly rather than relying on a global counter — a constraint any future
  parallel search has to respect.
- Timing fields (`duration_s`, `inference_time_s`) are the one thing that cannot
  reproduce. They are excluded from the experiment id.

## Revisit when

Parallel search is added: it will need deterministic seed derivation per trial
(`seed + index`) rather than a shared RNG. The design already points that way.
