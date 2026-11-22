# Contributing to Sensor2Spike

Thanks for looking. This is a small, focused tool; the bar for a change is that
it makes the measurement more trustworthy or the code easier to reason about.

## Development setup

```bash
git clone https://github.com/<your-fork>/sensor2spike.git
cd sensor2spike
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,bayes,viz]"
```

## Before opening a pull request

```bash
pytest                 # must stay green
ruff check src tests   # must stay clean
```

CI runs both on Python 3.10–3.13.

## What makes a good change here

**New encoders** are the most welcome contribution, and they should be a single
file plus a test. Subclass `SpikeEncoder`, set `name`, `description` and
`PARAM_NAMES`, implement `_fit` (set `self.n_neurons_`) and `_transform`
(return a binary `(n_steps, n_neurons)` array), decorate with
`@register_encoder`, and define `param_space()`. `examples/custom_encoder.py`
is a complete working template. Nothing in the engine needs to change.

**New SNN backends** follow the same pattern against `SNNBackend`: implement
`_features`, `_fit`, `_predict_scores` and `n_parameters`, then decorate with
`@register_backend`.

**Bug fixes** should come with the test that would have caught the bug. The
existing suite is organised by layer (`tests/test_encoders.py`,
`tests/test_experiment.py`, …); put the test next to its neighbours.

## Invariants to preserve

These are load-bearing; a change that breaks one needs a very good reason and an
ADR.

1. **Determinism.** Re-running a configuration must reproduce the same numbers.
   Never read global RNG state; derive seeds from the experiment seed and the
   trial index. `tests/test_experiment.py` enforces this.
2. **Failures are recorded, not raised away.** A crashing trial becomes a
   `failed` row; a crashing search becomes a `failed` experiment.
3. **No absolute paths in stored records or error strings.** Route anything
   user-facing through `redact_paths`.
4. **Reports contain only measured values.** No performance claim that the run
   did not produce.
5. **Outputs never overwrite inputs.** Report exporters refuse to write over a
   file that looks like a dataset.

## Adding a dependency

Core stays at `numpy` + `scipy`. Anything heavier goes in an extra
(`ml`, `bayes`, `viz`, `ui`) and must be imported lazily so the core install and
its tests never need it. `DependencyMissingError("pkg", extra="…")` produces the
install hint.

## Commit messages

Conventional-ish and in the imperative, one sentence of *why*:

```
encoders: seed trial RNG from the experiment seed

Poisson rate coding reused a process-wide RNG, so identical reruns
disagreed. Deriving the seed from (seed, trial index) makes experiments
reproducible, which the registry and the comparison view depend on.
```

## Architecture decisions

Non-obvious choices are recorded in `docs/adr/`. If your change alters one — the
default objective, the backend contract, the registry schema — add a new ADR
rather than quietly rewriting the old one.
