# ADR-001: Ship a modular monolith, not services

- **Status:** Accepted
- **Context:** Local-first developer tool, single user, one workstation.

## Context

The specification lists a dataset loader, a preprocessing stage, five encoders, an
optimiser, an SNN evaluation layer, a registry and a reporter. A service-oriented
decomposition would spread these across processes and require an orchestration
layer, a message format and deployment story, all to serve one user on one
machine. Network boundaries also mean data leaves the process — which conflicts
with the local-first constraint for users whose sensor recordings cannot be
shipped around.

## Decision

One installable Python package with clear internal seams:

- `contracts.py` defines the data types that cross every boundary
  (`SensorDataset`, `WindowSet`, `SpikeTrain`, `TrialResult`, `ExperimentResult`).
- `encoders/` and `snn/` are registries: strategies register themselves and are
  resolved by name, so neither the optimiser nor the CLI needs to know the list.
- `optimization/` depends on the encoder *contract*, never on a concrete encoder.
- `reporting/` and `experiment/` depend on `ExperimentResult`, never on the
  optimiser's internals.

## Consequences

**Good**

- One `pip install`; no ports, no config files, no daemon. `sensor2spike optimize`
  just runs.
- Extending an encoder is a single decorated class with no wiring changes.
- The whole pipeline is importable, so it is usable from a notebook and testable
  without a running system.
- No data leaves the process.

**Bad**

- No horizontal scaling. Fine: a 40-trial search on 80 000 samples takes seconds,
  and a user who needs more can parallelise at the process level by sharding
  configurations.
- Import-time coupling is possible if discipline slips. Mitigated by keeping the
  registries as the only cross-package entry points and by the dependency
  direction above.

## Revisit when

Multiple users need to share a registry, or a single optimisation exceeds one
machine. At that point the `ExperimentRegistry` interface is the natural seam to
put behind a server — the contracts do not change.
