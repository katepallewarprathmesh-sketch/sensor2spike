# Quick start

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[viz]"
```

`viz` brings in matplotlib for plots. `.[all]` adds `optuna` (Bayesian search),
`torch` (the PyTorch LIF backend) and `streamlit` (the demo UI).

Check that it worked:

```bash
sensor2spike --version
sensor2spike encoders
```

## 2. Get a dataset

Use your own, or start with the synthetic radar presence dataset:

```bash
sensor2spike demo --out demo.npz --csv
```

That writes `demo.npz` (native, keeps metadata) and `demo.csv` (for inspection).
Two classes: `empty` and `presence`.

To validate your own file:

```bash
sensor2spike validate mydata.csv \
    --label-column label \
    --group-column trial \
    --sampling-frequency 1000
```

The output tells you the shape, sampling rate, channel names, class count, group
count and a content fingerprint. If a group column is present, windows will never
cross a trial boundary and splits will be made per trial — this is what stops
overlapping windows from inflating accuracy.

## 3. Encode once, by hand

```bash
sensor2spike encode demo.npz --method rate --dt-ms 1 --out spikes.npz --plot
```

Prints spike count, firing rate and temporal sparsity, and writes `spikes.npz`
plus a raster plot.

## 4. Optimise

```bash
sensor2spike optimize demo.npz \
    --encoder rate \
    --optimizer random \
    --trials 30 \
    --window 0.25 \
    --registry runs.db \
    --out results
```

You get the experiment id, the recommended configuration with its measured
metrics, a table of the top trials, and reports in `results/`:

- `<id>.json` — full result, reloadable with `ExperimentResult.from_dict`
- `<id>.csv` — one row per trial, for a spreadsheet
- `<id>.md` — human-readable report
- `<id>.html` — self-contained, images inlined
- `<id>_pareto.png` — accuracy vs spike count with the front highlighted
- `<id>_tradeoffs.png` — the other objective pairs

## 5. Compare encoders

```bash
for enc in rate latency threshold delta population; do
  sensor2spike optimize demo.npz --encoder $enc --trials 20 \
      --window 0.25 --registry runs.db --out results/$enc
done

sensor2spike list --registry runs.db
sensor2spike compare --registry runs.db <id_rate> <id_delta> --out compare.md
```

The first experiment is the baseline; the rest are reported with deltas.

## 6. Take the configuration away

```bash
sensor2spike export --registry runs.db --experiment <id> --out best_config.json
```

```json
{
  "encoder": "rate",
  "params": {"dt_ms": 2.0, "gain": 1.18, "mode": "deterministic", "rate_max_hz": 59.0},
  "encoder_config": {"encoder": "rate", "dt_ms": 2.0, "...": "..."},
  "backend": {"backend": "lif_ridge", "tau_ms": 20.0},
  "reproduce": "AutoTuner(encoder='rate', seed=0, encoder_params={...}).optimize(dataset)"
}
```

`encoder_config` goes straight back into `build_encoder(...)`.

## From Python instead

```python
from sensor2spike import SensorDataset, AutoTuner

dataset = SensorDataset.load("demo.npz")
result = AutoTuner(
    objective="multi_objective",
    encoder="rate",
    n_trials=30,
    seed=0,
    window_seconds=0.25,
).optimize(dataset)

for trial in result.top(5):
    print(trial.rank, trial.params, trial.accuracy, trial.spike_count)
```

## Common knobs

| Setting | Meaning |
|---|---|
| `window_seconds` | Analysis window length. Should cover at least one period of the slowest signal you care about. |
| `window_stride_seconds` | Hop between windows; defaults to non-overlapping. |
| `test_fraction` | Fraction of **groups** held out for testing. |
| `accuracy_target` | Hard floor; trials below it are marked infeasible. |
| `spike_budget` | Hard ceiling on spikes per window. |
| `selection` | `balanced` (default), `knee`, `accuracy`, `spikes`, `score`. |
| `accuracy_tolerance` | How much accuracy `balanced` will trade for fewer spikes. |

## Troubleshooting

**"All windows belong to a single group"** — every sample carries the same group
id, so a leakage-free split is impossible. Either supply a `group_column`, or let
it default to one window per group.

**Accuracy stuck near chance** — check with
`sensor2spike encode ... --plot` that the raster actually differs between
classes. If the signal is dominated by noise, no encoding will fix it.

**"DependencyMissingError: optuna"** — `pip install "sensor2spike[bayes]"`.

**Plots missing from the report** — matplotlib is absent or has no display
backend; install the `viz` extra. Reports are still generated and record the
plot failure in `extras["plot_error"]`.
