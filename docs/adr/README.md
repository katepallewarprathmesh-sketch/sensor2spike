# Architecture decision records

Short records of the decisions that shape Sensor2Spike, written in the form
"context → decision → consequences". The intent is that a future maintainer can
see *why* the code looks like it does, and can tell a deliberate trade-off from
an accident.

| # | Decision |
|---|---|
| [001](001-modular-monolith.md) | Ship a modular monolith, not services |
| [002](002-numpy-reference-backend.md) | NumPy LIF reference backend as the default |
| [003](003-multi-objective-default.md) | Multi-objective optimisation with Pareto ranking by default |
| [004](004-pluggable-snn-backend.md) | Pluggable SNN backend abstraction |
| [005](005-sqlite-registry.md) | SQLite for the experiment registry |
| [006](006-determinism.md) | Determinism as a first-class requirement |
| [007](007-error-feedback-delta.md) | Sigma-delta error feedback for the delta encoder |

## Shared context

The product is a local-first developer tool. A single person runs it on a laptop
or a lab workstation, against data they are not allowed to upload anywhere, to
answer one question: *which spike encoding should I use, and is it worth the
spikes?* Everything below follows from that.

## What we deliberately did not do

- **No cloud service.** Uploading sensor data is often forbidden by the data
  owner; a local tool is the only one that can be used at all.
- **No custom neural framework.** Reinventing LIF simulation, surrogate
  gradients and training loops would produce a worse version of work that already
  exists; the backend registry exists so SpikingJelly/Norse/torch can be plugged
  in.
- **No claim of neuromorphic energy savings.** We report a compute *proxy* and
  say so. An unmeasured efficiency claim in a report is worse than no claim.
- **No leaderboard of encoders.** Results are dataset-specific. The tool measures;
  it does not rank encoders in the abstract.
