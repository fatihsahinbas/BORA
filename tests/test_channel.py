"""Channel impairments, and the receiver's ability to live with them."""

from __future__ import annotations

import numpy as np
import pytest

from bora import Config, channel


def test_awgn_hits_the_requested_ratio(rng) -> None:
    signal = rng.normal(0, 1, 200_000)
    noisy = channel.awgn(signal, 10.0, rng)

    noise = noisy - signal
    measured = 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))
    assert measured == pytest.approx(10.0, abs=0.2)


def test_awgn_leaves_a_silent_signal_alone(rng) -> None:
    silence = np.zeros(1_000)
    assert np.array_equal(channel.awgn(silence, 10.0, rng), silence)


def test_multipath_lengthens_by_the_longest_echo() -> None:
    signal = np.ones(1_000)
    out = channel.multipath(signal, delays=(0, 50, 300), gains=(1.0, 0.5, 0.2))
    assert len(out) == 1_300


def test_multipath_rejects_mismatched_arguments() -> None:
    with pytest.raises(ValueError):
        channel.multipath(np.ones(10), delays=(0, 1), gains=(1.0,))


@pytest.mark.parametrize("ppm", [-500.0, -60.0, 60.0, 500.0])
def test_clock_drift_changes_length_as_expected(ppm: float) -> None:
    signal = np.zeros(1_000_000)
    out = channel.clock_drift(signal, ppm)
    expected = int(len(signal) / (1 + ppm * 1e-6))
    assert abs(len(out) - expected) <= 1


def test_zero_drift_is_a_passthrough(rng) -> None:
    signal = rng.normal(0, 1, 1_000)
    assert np.array_equal(channel.clock_drift(signal, 0.0), signal)


def test_resample_inverts_clock_drift() -> None:
    """The receiver's correction must undo the transmitter's error.

    Linear interpolation is lossy, so this is not bit-exact.  The bound is set
    where it matters: 5% RMS on a mid-band tone leaves the constellation far
    inside a quadrant, which is the property the demodulator depends on.
    """
    signal = np.sin(2 * np.pi * 1_000 * np.arange(48_000) / 48_000)
    drifted = channel.clock_drift(signal, 200.0)
    restored = channel.resample(drifted, len(signal) / len(drifted))

    n = min(len(signal), len(restored))
    error = np.sqrt(np.mean((signal[:n] - restored[:n]) ** 2))
    assert error < 0.05


def test_resample_restores_the_original_length() -> None:
    signal = np.zeros(500_000)
    drifted = channel.clock_drift(signal, 300.0)
    restored = channel.resample(drifted, len(signal) / len(drifted))
    assert abs(len(restored) - len(signal)) <= 1


def test_resample_rejects_a_nonsense_ratio() -> None:
    with pytest.raises(ValueError):
        channel.resample(np.ones(10), 0.0)


def test_bandpass_tilt_suppresses_out_of_band_energy(cfg: Config) -> None:
    t = np.arange(48_000) / cfg.fs
    inside = np.sin(2 * np.pi * 2_000 * t)
    outside = np.sin(2 * np.pi * 15_000 * t)

    kept = channel.bandpass_tilt(inside, cfg.fs)
    cut = channel.bandpass_tilt(outside, cfg.fs)

    assert np.sqrt(np.mean(cut**2)) < 0.2 * np.sqrt(np.mean(kept**2))


def test_burst_noise_is_localised(rng) -> None:
    signal = np.ones(100_000)
    out = channel.burst_noise(signal, rng, n_bursts=2, duration=500)

    changed = np.flatnonzero(np.abs(out - signal) > 1e-9)
    assert 0 < len(changed) <= 2 * 500


def test_burst_noise_skips_a_signal_shorter_than_one_burst(rng) -> None:
    signal = np.ones(100)
    assert np.array_equal(channel.burst_noise(signal, rng, duration=500), signal)


def test_clip_bounds_the_output(rng) -> None:
    loud = rng.normal(0, 5, 10_000)
    assert np.max(np.abs(channel.clip(loud))) <= 1.0


def test_room_applies_every_impairment(cfg: Config, rng) -> None:
    signal = np.sin(2 * np.pi * 2_000 * np.arange(48_000) / cfg.fs) * 0.5
    out = channel.room(signal, cfg.fs, rng, snr_db=20, ppm=100)

    assert len(out) != len(signal)  # drift changed the length
    assert np.max(np.abs(out)) <= 1.0  # clipped
    assert not np.allclose(out[: len(signal)], signal)  # noise and echoes
