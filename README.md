# Sensor2Spike

**Find the spike encoding that gives you the accuracy you need for the fewest spikes.**

Sensor2Spike is a local-first Python toolkit for turning multichannel sensor
recordings into spike trains and *measuring* which encoder configuration is
worth running on a spiking neural network. It is built for the awkward middle of
an SNN project: you have sensor data, you have a neuromorphic target, and you
need evidence for choosing an encoding instead of a hunch.

- Five encoders behind one interface: rate, latency, threshold, delta
  (sigma-delta) modulation and population coding.
- Multi-objective search (accuracy, spike count, information loss, compute cost)
  with Pareto ranking — not a single opaque score.
- Pluggable SNN backends; a NumPy LIF reference ships in the box.
- Every experiment is recorded in SQLite with enough metadata to reproduce it.
- Reports as JSON/CSV/Markdown/HTML with accuracy-vs-spike-count and Pareto plots.
- Python SDK, CLI and an optional Streamlit UI.

```text
pip install -e ".[viz]"          # core + plotting
pip install -e ".[all]"          # everything (adds optuna, torch, streamlit)
```

---

## Quick start

```python
from sensor2spike import SensorDataset, AutoTuner, build_encoder

# 1. Load a recording (CSV, .npz or .npy).
dataset = SensorDataset.load("data/radar.npz")

# 2. Look at one encoding by hand.
encoder = build_encoder("rate", dt_ms=1.0, rate_max_hz=100.0).fit(
    dataset.data, fs=dataset.sampling_frequency
)
spikes = encoder.transform(dataset.data, fs=dataset.sampling_frequency)
print(spikes.spike_count, spikes.spike_rate_hz, spikes.temporal_sparsity)

# 3. Let the optimiser search the encoder's space.
result = AutoTuner(
    objective="multi_objective",
    encoder="rate",
    n_trials=40,
    seed=0,
    window_seconds=0.25,
).optimize(dataset)

best = result.best
print(best.params, best.accuracy, best.spike_count, best.information_loss)

# 4. Persist the measurement and the configuration.
result.export("out/result.json")
result.export_config("out/best_config.json")
```

`result` is an `ExperimentResult`: the trials, the Pareto front, the dataset
fingerprint, preprocessing settings, software versions and the recommended
configuration. Because all of that is captured, re-running the same configuration
reproduces the same measurements.

Round-trip check from the test suite:

```python
again = AutoTuner(objective="multi_objective", encoder="rate", n_trials=40,
                  seed=0, window_seconds=0.25).optimize(dataset)
assert [t.accuracy for t in again.trials] == [t.accuracy for t in result.trials]
```

---

## Try it in one command

```bash
sensor2spike demo --out demo.npz                 # synthetic radar presence dataset
sensor2spike optimize demo.npz \
    --encoder rate --trials 30 --window 0.25 \
    --registry runs.db --out results
sensor2spike list --registry runs.db
sensor2spike report --registry runs.db --experiment <id> --out reports
```

The `demo` dataset is a two-class synthetic radar recording: an empty room
(clutter plus noise) versus an occupied room (gait micro-Doppler modulated by
breathing). It exists so the whole pipeline can be exercised without hardware.

---

## Command line

| Command | What it does |
|---|---|
| `sensor2spike encoders` | List encoders, their parameters and the SNN backends |
| `sensor2spike demo` | Generate the synthetic radar dataset (`.npz`, optional `.csv`) |
| `sensor2spike validate DATA` | Parse a dataset, report shape/rate/groups/fingerprint |
| `sensor2spike encode DATA --method rate --out spikes.npz` | Encode one dataset, print spike statistics |
| `sensor2spike optimize DATA --encoder rate --trials 40` | Search, record, and write reports |
| `sensor2spike list` | List recorded experiments |
| `sensor2spike report --experiment ID` | Regenerate reports from the registry |
| `sensor2spike export --experiment ID` | Export the recommended configuration |
| `sensor2spike compare ID1 ID2` | Compare experiments with deltas against the baseline |

Everything accepts `--registry PATH` (default `sensor2spike.db`). Paths in error
messages are reduced to basenames, so a shared error log does not leak someone's
home directory.

---

## Encoders

All encoders implement the same contract — `fit`, `transform`, `get_config`,
`get_state`/`from_state`, `param_space` — so the optimiser treats them
uniformly and a new one can be added without touching the engine.

| Encoder | Idea | Key parameters |
|---|---|---|
| `rate` | Amplitude → firing rate | `rate_max_hz`, `mode` (deterministic/poisson), `gain` |
| `latency` | First-spike timing on Gaussian tuning curves, per frame | `neurons_per_channel`, `sigma`, `t_max_ms`, `frame_ms` |
| `threshold` | Upward crossings (`crossing`), up+down banks (`signed`), or level coding | `threshold`, `mode`, `refractory_ms` |
| `delta` | Sigma-delta error feedback: spikes track the signal's change | `threshold`, `max_steps_per_sample` |
| `population` | Gaussian receptive fields across neurons per channel | `neurons_per_channel`, `sigma`, `rate_max_hz`, `mode` |

`dt_ms` is part of every search space with the discrete values `0.5, 1, 2, 5 ms`
that hardware can actually generate. `mode` covers both stochastic and
deterministic spike generation so you can compare them honestly.

`delta` deserves a note. It keeps a running reconstruction and only fires when
the error reaches `threshold`, then moves the reconstruction by exactly one step.
Decoding is the cumulative sum of the spikes. On a band-limited signal this
reaches R² ≈ 0.998 at ~500 spikes per 2 s of two-tone input — verified, not
asserted, by `tests/test_encoders.py::test_delta_encoder_reconstructs_band_limited_signal`.

---

## What the optimiser actually optimises

```
J = w_a·(1 − accuracy) + w_s·S + w_l·L + w_c·C        subject to  accuracy ≥ A_min
```

where `S` is normalised spike activity, `L` information loss and `C` a compute
proxy (`spikes·10 + steps·neurons`). Everything is measured from the run; the
compute cost is clearly a proxy, not a silicon energy number.

`objective="multi_objective"` keeps the objectives separate and ranks trials by
non-dominated sorting plus crowding distance, so you see the whole trade-off
curve instead of one weighted sum hiding it. `objective="weighted"` collapses to
`J` when you want a single number.

The recommended configuration is chosen by `selection`:

- `balanced` (default) — among trials within `accuracy_tolerance` of the best
  observed accuracy, take the one with the fewest spikes;
- `knee` — knee point of the Pareto front;
- `accuracy`, `spikes`, `score` — single-objective choices.

---

## Measured results (reproducible)

`examples/benchmark.py` generates a band-limited synthetic radar dataset
(80 trials, 4 range bins, 1 kHz, 0.25 s windows, seed 7) and runs 40 random
search trials per encoder. Numbers below are from
`benchmark/benchmark_results.json`, produced by this repository:

| Encoder | Best accuracy | Macro F1 | Spikes | Info loss |
|---|---|---|---|---|
| `threshold` | **1.0000** | 1.0000 | 2 833 | 0.909 |
| `delta` | 0.9688 | 0.9687 | 35 848 | **0.488** |
| `latency` | 0.9271 | 0.9269 | 51 072 | 0.578 |
| `population` | 0.8958 | 0.8957 | 27 117 | 0.782 |
| `rate` | 0.8646 | 0.8646 | **2 668** | 0.898 |

Read this the way it should be read: on **one synthetic band-limited dataset with
40 trials per encoder**, `threshold` found a sparse, accurate configuration, while
`delta` preserved by far the most recoverable signal (information loss 0.488 versus
0.909). That is a statement about this dataset and this evaluation backend, not a
ranking of encoders in general. Real radar data, different windows and different
spike budgets will move these numbers. The point of the tool is that you can
re-run exactly this measurement on your own data.

The accuracy figures come from the NumPy LIF + ridge readout, which is a
*reference* backend for fast iteration, not a trained SNN. Validate final choices
with a framework backend before deployment.

---

## Architecture

The package is split into a **published surface** and a **private core**. You
only ever import from `sensor2spike`.

```
sensor2spike/
├── __init__.py        THE PUBLIC API — everything supported lives here
├── cli.py             command line, built entirely on the public API
├── demo.py            synthetic radar dataset generator
├── ui/                optional Streamlit app
├── _data.py ─┐
├── _encoding.py ─┤
├── _measurement.py ─┼─→  _core/     implementation (private)
├── _results.py ─┤          ├── contracts.py      data + result containers
├── _errors.py ─┘           ├── data/            I/O, preprocessing, splits
                            ├── encoders/        the five encoders + registry
                            ├── analysis/        spike stats, metrics, DSP
                            ├── snn/             backend ABC + reference LIF
                            ├── optimization/    objectives, Pareto, search
                            ├── experiment/      runner, SQLite registry
                            ├── reporting/       JSON/CSV/MD/HTML + plots
                            ├── spaces.py        search-space primitives
                            ├── exceptions.py    the exception hierarchy
                            └── utils/           hashing, validation, redaction
```

The leading underscore is a promise: `_core` is not an API. Importing
`sensor2spike.encoders` or `sensor2spike.contracts` fails with a message telling
you what to use instead, and `tests/test_public_api.py` keeps it that way.

### The two extension points

Everything else is internal, but these two contracts are meant to be implemented
by you, and they are versioned as such:

| Contract | Implement | Register |
|---|---|---|
| `SpikeEncoder` | `_fit`, `_transform`, `_neuron_count`, `param_space` | `@register_encoder` |
| `SNNBackend` | `_features`, `_fit`, `_predict_scores`, `n_parameters` | `@register_backend` |

Both inherit the rest — validation, metrics, timing, serialisation, determinism —
from the base class, so a new strategy is a small, self-contained file. The CLI,
the optimiser and the reports pick it up with no further wiring.

### Adding an encoder

```python
import numpy as np
from sensor2spike import SpikeEncoder, register_encoder

@register_encoder
class MyEncoder(SpikeEncoder):
    name = "my_encoder"
    description = "One sentence for the CLI listing"
    PARAM_NAMES = ("window_ms",)

    def __init__(self, *, dt_ms=1.0, seed=None, window_ms=10.0):
        super().__init__(dt_ms=dt_ms, seed=seed)
        self.window_ms = window_ms

    def _fit(self, data):                       # data: (n_samples, n_channels)
        self.n_neurons_ = data.shape[1]

    def _transform(self, data, fs, n_steps):     # -> (n_steps, n_neurons), 0/1
        values = self._normalize(self._steps(data, fs))
        return (values > 0.5).astype(np.uint8)

    def param_space(self):
        return super().param_space().merge(
            {"window_ms": {"kind": "float", "low": 1.0, "high": 50.0}}
        )
```

`_steps` resamples onto the encoder's time base and `_normalize` maps values onto
`[0, 1]` using the range learned during `fit`; `super().param_space()` already
includes `dt_ms`. `examples/custom_encoder.py` is a complete working version of
this, including a custom `decode`.

`sensor2spike encoders` now lists it and `AutoTuner(encoder="my_encoder")` will
search its space.

---

## Data requirements

- Array of shape `(n_samples, n_channels)`, finite values, float-friendly.
- `sampling_frequency` in Hz.
- Optional `labels` (integer class per sample) and `groups` (trial id per sample).
  **Groups matter**: windows never cross a group boundary and train/test splits
  are made at group level, which is what keeps a window-overlap leak out of your
  accuracy numbers.
- Guardrails: files up to 4 GiB and 50 M samples; `.npz` is read with
  `allow_pickle=False`, so a crafted archive cannot execute code.

CSV columns are selected with `--time-column`, `--label-column`,
`--group-column`, `--channel-columns`, and missing values with `--nan-policy
{raise,interpolate,zero,drop}`. A time column gives you the sampling rate for
free; without one, pass `--sampling-frequency`.

---

## Reliability and security

Built into the design rather than bolted on:

- **Deterministic runs.** Encoder RNGs are seeded from the experiment seed and
  the trial index, never from process state. Same inputs, same numbers.
- **Failures are recorded.** A crashing trial becomes a `failed` row with a
  redacted error; a crashing search is stored as a `failed` experiment, not lost.
- **No partial writes.** Registry writes happen in a single transaction; report
  files are written as complete documents.
- **Path redaction.** Error messages and stored records carry basenames, not
  absolute paths (`utils/security.py`).
- **Outputs stay separate from inputs.** Report exporters refuse to overwrite a
  file that looks like a dataset.
- **Input validation at the boundary.** Unknown configuration keys, non-finite
  data, unknown encoder names and malformed `key=value` parameters produce typed
  errors with actionable messages — no `KeyError` from three layers down.

---

## Testing

```bash
pytest                       # 235 tests
ruff check src tests         # clean
```

The suite covers the encoder contract for all five encoders, dataset round trips
(including CSV at full float precision), leakage-free splitting, Pareto ranking,
search strategies, registry persistence and determinism, report formats, the CLI
end to end, and the security properties above. `tests/test_workflow.py` walks the
full developer workflow, including bit-for-bit reproduction of a stored
experiment, and `tests/test_public_api.py` enforces the boundary described above:
the documented names must exist, nothing else may leak, and importing the package
must not pull in an optional dependency.

---

## Limitations

Stated plainly, because the reports are supposed to be trustworthy:

- The bundled SNN backend is a NumPy LIF membrane-feature + ridge readout. It is
  fast and deterministic, and it is **not** a trained spiking network. The
  PyTorch backend is optional and not exercised in CI here because `torch` is not
  installed in this environment.
- Compute cost is a *proxy*, not measured energy on neuromorphic hardware.
- Information loss is "1 − R² of a linear reconstruction" — a linear-information
  proxy, not a full mutual-information estimate.
- The benchmark numbers come from a synthetic dataset. Treat them as a
  demonstration of the loop, not as evidence about real sensors.
- Windows spanning class transitions are labelled by the centre sample by
  default; use `label_mode="majority"` if your data has many transitions.

---

## License

MIT. See `LICENSE`.
