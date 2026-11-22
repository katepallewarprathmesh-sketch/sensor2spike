# ADR-005: SQLite for the experiment registry

- **Status:** Accepted

## Context

Every experiment must be reproducible: dataset identity, encoder configuration,
seed, software versions, status and results. Users need to list what they ran,
reload an old result, compare two runs and regenerate a report — on a laptop,
with no server and no account.

Options considered: JSON files on disk, an embedded document store, or SQLite.

## Decision

SQLite, through the standard library's `sqlite3`, with an explicit schema:

| Table | Holds |
|---|---|
| `meta` | schema version, tool version |
| `datasets` | name, sensor type, shape, fingerprint, version |
| `encoder_configs` | normalised encoder configuration + search space |
| `models` | backend configuration |
| `experiments` | one row per run: ids, objective, seed, status, timings, best |
| `results` | one row per trial, foreign-keyed to an experiment |

Three properties come from this choice:

1. **Deterministic ids.** `experiment_id_for(result)` hashes the meaningful
   configuration — dataset fingerprint, encoder config, space, optimizer,
   objective, backend, preprocessing, windowing, split, seed, trial count,
   software versions. The same experiment gets the same id; a genuine repeat into
   the same registry gets a `-2` suffix rather than overwriting evidence.
2. **Transactional writes.** An experiment and its trials land in one
   transaction, so a crash cannot leave a half-recorded run.
3. **Queryability.** `list_experiments`, status filters, `trial_table` and the
   comparison view are plain SQL, and any DB browser works for inspection.

## Consequences

**Good**

- Zero operational overhead: one file, no daemon, easy to copy or archive.
- Foreign keys (enabled per connection) keep trial rows attached to experiments;
  deletes cascade explicitly (children first).
- `load_experiment` reconstructs a full `ExperimentResult`, so reports can be
  regenerated from the registry alone.

**Bad**

- Single-writer. Concurrent processes writing the same DB file will serialise or
  fail; parallel searches should use separate registry files and be merged later.
- Schema migrations are manual. The `meta` table carries a schema version so a
  future migration has something to branch on.
- Old rows accumulate. `delete_experiment` exists; automated pruning does not.

## Revisit when

Teams need a shared registry. The `ExperimentRegistry` method surface
(`record_experiment`, `load_experiment`, `list_experiments`, `get_results`) is the
seam — a Postgres implementation would not change the contracts.
