"""Optional Streamlit UI for exploring Sensor2Spike experiments.

Run with::

    pip install "sensor2spike[ui]"
    streamlit run src/sensor2spike/ui/app.py

The UI is deliberately thin: it loads the same public API the SDK and CLI use,
so it cannot drift from them. It is a convenience for browsing runs, not a
second implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensor2spike import (
    AutoTuner,
    ExperimentRegistry,
    SensorDataset,
    build_encoder,
    list_encoders,
)
from sensor2spike.demo import demo_summary, generate_demo_dataset

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - optional extra
    raise SystemExit(
        'The UI needs Streamlit. Install it with:  pip install "sensor2spike[ui]"'
    ) from exc


st.set_page_config(page_title="Sensor2Spike", page_icon="⚡", layout="wide")


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _demo_dataset(n_trials: int, seed: int, noise: float) -> SensorDataset:
    return generate_demo_dataset(n_trials=n_trials, seed=seed, noise=noise)


@st.cache_data(show_spinner=False)
def _demo_arrays(n_trials: int, seed: int, noise: float) -> tuple[np.ndarray, np.ndarray, int]:
    dataset = _demo_dataset(n_trials, seed, noise)
    return dataset.data, dataset.labels, dataset.n_channels


def _raster_figure(spikes, title: str):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 3.2))
    rows, columns = np.nonzero(spikes.spikes)
    axis.scatter(columns * spikes.dt * 1000.0, rows, s=1.2, marker="|", color="#2251a4")
    axis.set_xlabel("time (ms)")
    axis.set_ylabel("neuron")
    axis.set_title(title)
    axis.set_ylim(spikes.n_neurons, -1)
    figure.tight_layout()
    return figure


def _pareto_figure(result):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    ok = [trial for trial in result.trials if trial.status == "ok"]
    front = sorted(result.pareto_front, key=lambda trial: trial.spike_count)
    others = [trial for trial in ok if not trial.is_pareto]
    if others:
        axis.scatter(
            [trial.spike_count for trial in others],
            [trial.accuracy for trial in others],
            s=22,
            alpha=0.45,
            label="dominated",
            color="#9aa5b1",
        )
    if front:
        axis.plot(
            [trial.spike_count for trial in front],
            [trial.accuracy for trial in front],
            "o-",
            label="Pareto front",
            color="#c2410c",
            markersize=5,
        )
    best = result.best
    if best is not None:
        axis.scatter(
            [best.spike_count], [best.accuracy], s=140, marker="*",
            color="#1d4ed8", zorder=5, label="recommended",
        )
    axis.set_xlabel("spikes per window")
    axis.set_ylabel("accuracy")
    axis.set_title(f"{result.experiment_id} — accuracy vs spikes")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    return figure


# ----------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------
st.sidebar.title("⚡ Sensor2Spike")
page = st.sidebar.radio("View", ["Dataset", "Encode", "Optimise", "Browse runs"])

registry_path = st.sidebar.text_input("Registry", value="sensor2spike.db")

st.sidebar.markdown("---")
st.sidebar.caption(
    "All computation happens locally. The synthetic dataset is generated in-process "
    "and nothing is uploaded."
)

st.sidebar.subheader("Dataset")
source = st.sidebar.radio("Source", ["Synthetic radar", "Upload .npz"])

dataset: SensorDataset | None = None
if source == "Synthetic radar":
    n_trials = st.sidebar.slider("Trials", 10, 200, 40, step=10)
    seed = st.sidebar.number_input("Seed", 0, 9_999, 7, step=1)
    noise = st.sidebar.slider("Noise", 0.0, 0.6, 0.15, step=0.05)
    dataset = _demo_dataset(int(n_trials), int(seed), float(noise))
else:
    upload = st.sidebar.file_uploader("Dataset (.npz from save_dataset)", type=["npz"])
    if upload is not None:
        import io

        buffer = io.BytesIO(upload.getvalue())
        data = np.load(buffer, allow_pickle=False)
        dataset = SensorDataset(
            data=np.asarray(data["data"], dtype=np.float32),
            sampling_frequency=float(np.asarray(data["sampling_frequency"]).reshape(-1)[0]),
            channel_names=tuple(str(n) for n in data["channel_names"].tolist())
            if "channel_names" in data else (),
            labels=np.asarray(data["labels"]) if "labels" in data else None,
            groups=np.asarray(data["groups"]) if "groups" in data else None,
            name="uploaded",
        )


# ----------------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------------
def page_dataset(dataset: SensorDataset) -> None:
    st.title("Dataset")
    summary = demo_summary(dataset) if dataset.name.startswith("demo") else dataset.summary()

    columns = st.columns(4)
    columns[0].metric("Samples", f"{dataset.n_samples:,}")
    columns[1].metric("Channels", dataset.n_channels)
    columns[2].metric("Trials", dataset.n_groups if dataset.groups is not None else 1)
    rate = dataset.sampling_frequency
    columns[3].metric("Sample rate", f"{rate:,.0f} Hz")

    if dataset.labels is not None:
        counts = np.bincount(dataset.labels.astype(int))
        st.bar_chart(
            {"samples": counts.tolist()},
        )
        st.caption("Class balance (samples per class).")

    st.subheader("Signal preview")
    preview = dataset.data[: int(min(dataset.n_samples, rate * 2))]
    st.line_chart(preview[:, : min(4, dataset.n_channels)])

    st.subheader("Metadata")
    st.json({"fingerprint": dataset.fingerprint(), **summary})


def page_encode(dataset: SensorDataset) -> None:
    st.title("Encode")
    st.caption("Apply one encoder with explicit parameters and inspect the spikes.")

    encoders = {entry["name"]: entry for entry in list_encoders()}
    name = st.selectbox("Encoder", sorted(encoders))
    entry = encoders[name]

    params: dict[str, object] = {}
    columns = st.columns(len(entry["parameters"]) or 1)
    for position, parameter in enumerate(entry["parameters"]):
        space = build_encoder(name).param_space()
        spec = space[parameter]
        with columns[position % len(columns)]:
            if spec.kind == "choice":
                params[parameter] = st.selectbox(parameter, list(spec.choices))
            elif spec.kind == "int":
                params[parameter] = st.number_input(
                    parameter, int(spec.low), int(spec.high), int(spec.low), step=1
                )
            else:
                params[parameter] = st.slider(
                    parameter, float(spec.low), float(spec.high),
                    float((spec.low + spec.high) / 2.0),
                )

    dt_ms = st.select_slider("dt_ms (ms)", options=[0.5, 1.0, 2.0, 5.0], value=1.0)

    if st.button("Encode", type="primary"):
        encoder = build_encoder(name, dt_ms=dt_ms, seed=0, **params)
        encoder.fit(dataset.data, fs=dataset.sampling_frequency)
        spikes = encoder.transform(dataset.data, fs=dataset.sampling_frequency)

        metrics = st.columns(4)
        metrics[0].metric("Spikes", f"{spikes.spike_count:,}")
        metrics[1].metric("Rate", f"{spikes.spike_rate_hz:,.1f} /s")
        metrics[2].metric("Sparsity", f"{spikes.temporal_sparsity:.3f}")
        loss = encoder.information_loss(spikes, dataset.data)
        metrics[3].metric("Info loss", "n/a" if loss is None else f"{loss:.3f}")

        st.pyplot(_raster_figure(spikes, f"{name} encoding — {dataset.name}"))


def page_optimise(dataset: SensorDataset) -> None:
    st.title("Optimise")
    columns = st.columns(3)
    encoder = columns[0].selectbox("Encoder", [e["name"] for e in list_encoders()])
    optimizer = columns[1].selectbox("Search", ["random", "grid", "bayes"])
    n_trials = columns[2].number_input("Trials", 2, 200, 12, step=2)

    columns = st.columns(3)
    window = columns[0].select_slider("Window (s)", options=[0.1, 0.25, 0.5, 1.0], value=0.25)
    objective = columns[1].selectbox("Objective", ["multi_objective", "weighted"])
    selection = columns[2].selectbox("Selection", ["balanced", "knee", "accuracy", "spikes", "score"])

    columns = st.columns(3)
    target = columns[0].number_input("Accuracy target", 0.0, 1.0, 0.0, step=0.05)
    budget = columns[1].number_input("Spike budget", 0, 1_000_000, 0, step=100)
    seed = columns[2].number_input("Seed", 0, 9_999, 0, step=1)

    if st.button("Run optimisation", type="primary"):
        registry = ExperimentRegistry(registry_path)
        with st.spinner(f"Searching {n_trials} configurations…"):
            result = AutoTuner(
                objective=objective,
                encoder=encoder,
                optimizer=optimizer,
                n_trials=int(n_trials),
                seed=int(seed),
                window_seconds=float(window),
                accuracy_target=None if target == 0 else float(target),
                spike_budget=None if budget == 0 else int(budget),
                selection=selection,
                registry=registry,
            ).optimize(dataset)

        if result.best is None:
            st.error("No trial completed successfully.")
            st.json(result.extras)
            return

        best = result.best
        st.success(f"Experiment `{result.experiment_id}` — {result.status}")
        columns = st.columns(4)
        columns[0].metric("Accuracy", f"{best.accuracy:.4f}")
        columns[1].metric("Spikes", f"{best.spike_count:,}")
        columns[2].metric("Info loss", f"{best.information_loss:.3f}")
        columns[3].metric("Pareto front", len(result.pareto_front))

        st.pyplot(_pareto_figure(result))

        st.subheader("Recommended configuration")
        st.json({"params": best.params, "metrics": best.to_dict()})

        st.subheader("All trials")
        st.dataframe(
            [
                {
                    "rank": trial.rank,
                    "status": trial.status,
                    "accuracy": round(trial.accuracy, 4),
                    "spikes": trial.spike_count,
                    "info loss": round(trial.information_loss, 4),
                    "pareto": trial.is_pareto,
                    "params": json.dumps(trial.params, default=str),
                }
                for trial in result.trials
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download result (JSON)",
            data=json.dumps(result.to_dict(), indent=2, default=str),
            file_name=f"{result.experiment_id}.json",
            mime="application/json",
        )
        config = json.dumps(
            {"encoder": encoder, "params": best.params, "seed": result.seed},
            indent=2, default=str,
        )
        st.download_button(
            "Download configuration (JSON)", data=config,
            file_name="best_config.json", mime="application/json",
        )


def page_browse(registry_path: str) -> None:
    st.title("Browse runs")
    path = Path(registry_path)
    if not path.exists():
        st.info(f"No registry at `{registry_path}`. Run an optimisation first.")
        return

    registry = ExperimentRegistry(path)
    experiments = registry.list_experiments(limit=200)
    if not experiments:
        st.info("The registry is empty.")
        return

    st.dataframe(experiments, use_container_width=True, hide_index=True)

    identifiers = [row["experiment_id"] for row in experiments]
    chosen = st.selectbox("Inspect", identifiers)
    if chosen:
        result = registry.load_experiment(chosen)
        best = result.best
        if best is not None:
            columns = st.columns(4)
            columns[0].metric("Accuracy", f"{best.accuracy:.4f}")
            columns[1].metric("Spikes", f"{best.spike_count:,}")
            columns[2].metric("Backend", result.backend.get("backend", "?"))
            columns[3].metric("Seed", result.seed)
            st.pyplot(_pareto_figure(result))
            st.json({"params": best.params, "dataset": result.dataset})
        st.caption(
            "Trials: "
            f"{sum(1 for t in result.trials if t.status == 'ok')} ok / "
            f"{len(result.trials)} total"
        )


# ----------------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------------
if dataset is None:
    st.title("⚡ Sensor2Spike")
    st.info("Choose a data source in the sidebar to begin.")
    st.stop()

if page == "Dataset":
    page_dataset(dataset)
elif page == "Encode":
    page_encode(dataset)
elif page == "Optimise":
    page_optimise(dataset)
else:
    page_browse(registry_path)
