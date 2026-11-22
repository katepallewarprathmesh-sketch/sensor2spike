"""Spike-activity statistics and analytical compute cost.

The compute cost is an *analytical proxy* (operations per spike and per
neuron-step), not a measurement on neuromorphic hardware.  It is used to compare
candidate configurations against one another and is labelled as a proxy
everywhere it is reported.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..contracts import SpikeStats, SpikeTrain

__all__ = ["SPIKE_COST", "STEP_COST", "compute_cost", "compute_spike_stats"]

#: Arbitrary compute units charged per spike (synaptic event + neuron update).
SPIKE_COST = 10.0
#: Arbitrary compute units charged per neuron-step (membrane state update).
STEP_COST = 1.0


def compute_cost(spikes: SpikeTrain | Sequence[SpikeTrain]) -> float:
    """Return the analytical cost of processing ``spikes``.

    ``cost = spike_count * SPIKE_COST + n_steps * n_neurons * STEP_COST``.
    """
    trains = [spikes] if isinstance(spikes, SpikeTrain) else list(spikes)
    if not trains:
        return 0.0
    total_spikes = sum(train.spike_count for train in trains)
    total_steps = sum(train.n_steps for train in trains)
    n_neurons = trains[0].n_neurons
    return float(total_spikes * SPIKE_COST + total_steps * n_neurons * STEP_COST)


def compute_spike_stats(
    spikes: SpikeTrain | Sequence[SpikeTrain],
    *,
    information_loss: float | None = None,
    reconstruction_r2: float | None = None,
    compute_cost_override: float | None = None,
) -> SpikeStats:
    """Compute aggregate spike statistics for one or more spike trains."""
    trains = [spikes] if isinstance(spikes, SpikeTrain) else list(spikes)
    if not trains:
        raise ValueError("compute_spike_stats requires at least one spike train.")

    n_neurons = trains[0].n_neurons
    if any(train.n_neurons != n_neurons for train in trains):
        raise ValueError("All spike trains must share the same number of neurons.")

    total_spikes = sum(train.spike_count for train in trains)
    total_steps = sum(train.n_steps for train in trains)
    duration = sum(train.duration_seconds for train in trains)
    per_neuron = np.zeros(n_neurons, dtype=np.float64)
    intervals: list[np.ndarray] = []
    for train in trains:
        per_neuron += train.spikes.sum(axis=0, dtype=np.float64)
        isi = train.isi()
        if isi.size:
            intervals.append(isi)

    entries = max(1, total_steps * n_neurons)
    active_fraction = total_spikes / entries
    mean_isi_ms = None
    isi_cv = None
    if intervals:
        pooled = np.concatenate(intervals)
        if pooled.size:
            mean_isi_ms = float(pooled.mean() * 1000.0)
            std = float(pooled.std())
            mean = float(pooled.mean())
            isi_cv = float(std / mean) if mean > 0 else None

    return SpikeStats(
        spike_count=int(total_spikes),
        n_steps=int(total_steps),
        n_neurons=int(n_neurons),
        duration_seconds=float(duration),
        spike_rate_hz=float(total_spikes / duration) if duration > 0 else 0.0,
        spikes_per_second_per_neuron=float(total_spikes / duration / n_neurons) if duration > 0 else 0.0,
        active_fraction=float(active_fraction),
        temporal_sparsity=float(1.0 - active_fraction),
        mean_isi_ms=mean_isi_ms,
        isi_cv=isi_cv,
        firing_rates_hz=per_neuron / duration if duration > 0 else per_neuron,
        information_loss=None if information_loss is None else float(information_loss),
        reconstruction_r2=None if reconstruction_r2 is None else float(reconstruction_r2),
        compute_cost=(
            float(compute_cost_override)
            if compute_cost_override is not None
            else compute_cost(trains)
        ),
    )
