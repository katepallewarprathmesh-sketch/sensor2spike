"""Contract tests for every encoder in the registry."""

from __future__ import annotations

import numpy as np
import pytest

from sensor2spike.contracts import SpikeTrain
from sensor2spike.encoders import (
    ENCODERS,
    SpikeEncoder,
    build_encoder,
    encoder_space,
    list_encoders,
    validate_encoder_config,
)
from sensor2spike.exceptions import ConfigurationError, DataValidationError, EncodingError

ENCODER_NAMES = sorted(ENCODERS)


def _signal(n_samples: int = 500, n_channels: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 500.0
    data = np.stack(
        [np.sin(2 * np.pi * (2.0 + channel) * t) for channel in range(n_channels)], axis=1
    )
    return (data + 0.05 * rng.standard_normal(data.shape)).astype(np.float64)


# ---------------------------------------------------------------------------- registry
def test_registry_exposes_all_five_encoders():
    assert set(ENCODER_NAMES) == {"rate", "latency", "threshold", "delta", "population"}


def test_list_encoders_reports_parameters():
    entries = {entry["name"]: entry for entry in list_encoders()}
    assert set(entries) == set(ENCODER_NAMES)
    for entry in entries.values():
        assert entry["description"]
        assert isinstance(entry["parameters"], list)


@pytest.mark.parametrize("name", ENCODER_NAMES)
def test_build_encoder_returns_fitted_contract(name):
    encoder = build_encoder(name)
    assert isinstance(encoder, SpikeEncoder)
    assert encoder.name == name
    assert encoder.get_config()["encoder"] == name


def test_build_encoder_accepts_mapping_and_kwargs():
    from_config = build_encoder({"encoder": "rate", "rate_max_hz": 250.0})
    assert from_config.rate_max_hz == pytest.approx(250.0)
    with_kwargs = build_encoder("rate", rate_max_hz=250.0, dt_ms=2.0)
    assert with_kwargs.rate_max_hz == pytest.approx(250.0)
    assert with_kwargs.dt_ms == pytest.approx(2.0)


def test_build_encoder_rejects_unknown_name_and_parameter():
    with pytest.raises(ConfigurationError):
        build_encoder("does_not_exist")
    with pytest.raises(ConfigurationError):
        build_encoder("rate", not_a_parameter=1)


def test_validate_encoder_config_normalises_values():
    config = validate_encoder_config("rate", {"dt_ms": "2.0", "rate_max_hz": 50})
    assert config["dt_ms"] == pytest.approx(2.0)
    assert config["rate_max_hz"] == pytest.approx(50.0)
    # Omitted parameters fall back to the encoder defaults.
    assert config["mode"] == "deterministic"
    with pytest.raises(ConfigurationError):
        validate_encoder_config("rate", {"nonsense": 1})


# ------------------------------------------------------------------------- transform API
@pytest.mark.parametrize("name", ENCODER_NAMES)
def test_transform_produces_binary_spike_train(name):
    data = _signal()
    encoder = build_encoder(name, dt_ms=1.0, seed=0)
    spikes = encoder.fit_transform(data, fs=500.0)

    assert isinstance(spikes, SpikeTrain)
    assert spikes.spikes.ndim == 2
    assert set(np.unique(spikes.spikes)).issubset({0, 1})
    assert spikes.spikes.shape[0] == spikes.n_steps
    assert spikes.spikes.shape[1] == spikes.n_neurons
    assert spikes.encoder == name
    assert encoder.is_fitted


@pytest.mark.parametrize("name", ENCODER_NAMES)
def test_transform_requires_fit_first(name):
    encoder = build_encoder(name)
    with pytest.raises(EncodingError):
        encoder.transform(_signal(), fs=500.0)


@pytest.mark.parametrize("name", ENCODER_NAMES)
def test_transform_rejects_wrong_channel_count(name):
    encoder = build_encoder(name, seed=0).fit(_signal(n_channels=3), fs=500.0)
    with pytest.raises((EncodingError, DataValidationError)):
        encoder.transform(_signal(n_channels=4), fs=500.0)


@pytest.mark.parametrize("name", ENCODER_NAMES)
def test_fit_transform_is_deterministic_for_fixed_seed(name):
    data = _signal(seed=5)
    first = build_encoder(name, seed=42).fit_transform(data, fs=500.0)
    second = build_encoder(name, seed=42).fit_transform(data, fs=500.0)
    np.testing.assert_array_equal(first.spikes, second.spikes)


@pytest.mark.parametrize("name", ENCODER_NAMES)
def test_reproducibility_does_not_depend_on_the_process_rng(name):
    """Encoding must be driven only by the encoder's own seed.

    The runner relies on this: two runs of the same experiment must produce the
    same spikes even if unrelated code has advanced the global RNG.
    """
    data = _signal(seed=5)
    np.random.seed(0)
    np.random.random(1000)
    first = build_encoder(name, seed=42).fit_transform(data, fs=500.0)
    np.random.seed(1234)
    np.random.random(1000)
    second = build_encoder(name, seed=42).fit_transform(data, fs=500.0)
    np.testing.assert_array_equal(first.spikes, second.spikes)


@pytest.mark.parametrize("name", ENCODER_NAMES)
def test_transform_windows_matches_manual_loop(name):
    windows = np.stack([_signal(200, seed=seed) for seed in range(3)])
    encoder = build_encoder(name, seed=0)
    encoder.fit(np.concatenate(list(windows), axis=0), fs=500.0)
    trains = encoder.transform_windows(windows)
    manual = [encoder.transform(window, fs=500.0) for window in windows]
    assert len(trains) == 3
    for batched, expected in zip(trains, manual, strict=False):
        np.testing.assert_array_equal(batched.spikes, expected.spikes)


@pytest.mark.parametrize("name", ENCODER_NAMES)
def test_step_count_matches_dt(name):
    data = _signal(n_samples=1000)
    for dt_ms in (0.5, 1.0, 2.0, 5.0):
        spikes = build_encoder(name, dt_ms=dt_ms, seed=0).fit_transform(data, fs=500.0)
        expected = int(round((1000 / 500.0) / (dt_ms / 1000.0)))
        assert abs(spikes.n_steps - expected) <= 1


# ---------------------------------------------------------------------------- encodings
def test_rate_encoder_increases_spikes_with_amplitude():
    quiet = np.full((400, 1), 0.05)
    loud = np.full((400, 1), 1.0)
    # fit on both so the normalisation range is shared
    encoder = build_encoder("rate", rate_max_hz=200.0, dt_ms=1.0, mode="deterministic")
    encoder.fit(np.concatenate([quiet, loud]), fs=1000.0)
    quiet_spikes = encoder.transform(quiet, fs=1000.0).spike_count
    loud_spikes = encoder.transform(loud, fs=1000.0).spike_count
    assert loud_spikes > quiet_spikes


def test_latency_encoder_uses_one_spike_per_neuron_pair():
    spikes = build_encoder("latency", neurons_per_channel=4, frame_ms=50.0).fit_transform(
        _signal(500, 2), fs=500.0
    )
    # One neuron subset per channel: at most one spike per channel per window.
    per_channel_counts = spikes.spikes.sum(axis=1)
    assert per_channel_counts.max() <= 2


def test_threshold_encoder_respects_refractory_period():
    spikes = build_encoder(
        "threshold", threshold=0.1, refractory_ms=20.0, dt_ms=1.0, mode="signed"
    ).fit_transform(_signal(2000, 1, seed=2), fs=1000.0)
    assert spikes.n_neurons == 2  # up + down bank for a single channel
    # isi() reports seconds; refractory_ms=20 must be respected per neuron.
    isi = spikes.isi()
    if isi.size:
        assert isi.min() >= 20.0e-3 - 1e-9


def test_delta_encoder_emits_on_change_only():
    flat = np.zeros((500, 1))
    step = np.concatenate([np.zeros(250), np.ones(250)])[:, None]
    encoder = build_encoder("delta", threshold=0.1, dt_ms=1.0)
    encoder.fit(np.concatenate([flat, step]), fs=500.0)
    flat_spikes = encoder.transform(flat, fs=500.0).spike_count
    step_spikes = encoder.transform(step, fs=500.0).spike_count
    assert flat_spikes == 0
    assert step_spikes > 0


def test_population_encoder_activates_more_neurons_for_higher_amplitude():
    low = np.full((200, 1), 0.1)
    high = np.full((200, 1), 0.9)
    encoder = build_encoder("population", neurons_per_channel=8, sigma=0.2)
    encoder.fit(np.concatenate([low, high]), fs=1000.0)
    low_active = int(np.count_nonzero(encoder.transform(low, fs=1000.0).firing_rates_hz))
    high_active = int(np.count_nonzero(encoder.transform(high, fs=1000.0).firing_rates_hz))
    assert high_active >= low_active


# ---------------------------------------------------------------------- state / config IO
@pytest.mark.parametrize("name", ENCODER_NAMES)
def test_config_round_trip_rebuilds_encoder(name):
    encoder = build_encoder(name, dt_ms=2.0, seed=3).fit(_signal(), fs=500.0)
    config = encoder.get_config()
    rebuilt = build_encoder(config).fit(_signal(), fs=500.0)
    assert rebuilt.get_config() == config


@pytest.mark.parametrize("name", ENCODER_NAMES)
def test_get_state_from_state_round_trip(name):
    encoder = build_encoder(name, seed=7).fit(_signal(), fs=500.0)
    state = encoder.get_state()
    clone = build_encoder(name).from_state(state)
    assert clone.n_channels_ == encoder.n_channels_
    np.testing.assert_allclose(clone.scale_min_, encoder.scale_min_)


# ---------------------------------------------------------------------------- search space
@pytest.mark.parametrize("name", ENCODER_NAMES)
def test_search_space_is_sampleable_and_mergeable(name):
    space = encoder_space(name)
    rng = np.random.default_rng(0)
    for _ in range(5):
        sample = space.sample(rng)
        assert set(sample).issubset(set(space.keys()))
        # Every sampled value must build a usable encoder.
        build_encoder({**sample, "encoder": name})

    narrowed = encoder_space(name, {"dt_ms": 2.0})
    assert narrowed["dt_ms"].kind == "choice"
    assert narrowed["dt_ms"].choices == (2.0,)


@pytest.mark.parametrize("name", ENCODER_NAMES)
def test_default_config_omits_dt_ms_override(name):
    encoder = build_encoder(name)
    assert encoder.dt_ms == pytest.approx(1.0)


def test_delta_encoder_reconstructs_band_limited_signal():
    """Sigma-delta error feedback must make ``decode`` a true inverse.

    This is the property the encoder is documented to have, and the reason its
    information-loss term is meaningful: on a band-limited signal the cumulative
    sum of the spikes reproduces the waveform up to a constant.
    """
    from sensor2spike.analysis.dsp import reconstruction_r2, resample_to_steps

    fs = 1000.0
    t = np.arange(2000) / fs
    signal = (np.sin(2 * np.pi * 3 * t) + 0.3 * np.sin(2 * np.pi * 7 * t))[:, None]

    encoder = build_encoder("delta", threshold=0.02, dt_ms=1.0, max_steps_per_sample=8)
    encoder.fit(signal, fs=fs)
    spikes = encoder.transform(signal, fs=fs)
    decoded = encoder.decode(spikes)
    reference = resample_to_steps(signal, fs, spikes.dt, spikes.n_steps)
    assert reconstruction_r2(decoded, reference) > 0.99
    assert 0 < spikes.spike_count < 0.5 * spikes.n_steps


def test_delta_encoder_trades_accuracy_for_sparsity():
    """A finer step must track better and cost more spikes."""
    fs = 500.0
    t = np.arange(1000) / fs
    signal = np.sin(2 * np.pi * 2 * t)[:, None]
    results = {}
    for threshold in (0.005, 0.1):
        encoder = build_encoder("delta", threshold=threshold, dt_ms=1.0, max_steps_per_sample=8)
        encoder.fit(signal, fs=fs)
        spikes = encoder.transform(signal, fs=fs)
        results[threshold] = (encoder, spikes, encoder.information_loss(spikes, signal))
    fine, coarse = results[0.005], results[0.1]
    assert fine[1].spike_count > coarse[1].spike_count
    assert fine[2] < coarse[2]
