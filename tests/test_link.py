"""End to end: bytes in, sound, bytes out.

These are the tests that would catch a regression a user would actually
notice.  The ones that push a signal through a simulated room are the reason
the channel module exists -- they are slower than the unit tests but they are
the only ones that check the parts work *together*.
"""

from __future__ import annotations

import numpy as np
import pytest

from bora import Config, channel, link


def test_roundtrip_through_a_perfect_channel(cfg: Config, payload: bytes) -> None:
    result = link.receive(link.transmit(payload, cfg), cfg)
    assert result.ok
    assert result.payload == payload
    assert result.fec is not None and result.fec.failed == 0


@pytest.mark.parametrize("size", [0, 1, 222, 223, 224, 1_000])
def test_payload_sizes_around_the_block_boundary(cfg: Config, size: int) -> None:
    """223 bytes is the codeword payload size, so the edges live here."""
    payload = bytes(range(256)) * 10
    chunk = payload[:size]
    result = link.receive(link.transmit(chunk, cfg), cfg)
    assert result.ok
    assert result.payload == chunk


def test_roundtrip_without_error_correction(cfg: Config, payload: bytes) -> None:
    result = link.receive(link.transmit(payload, cfg, use_fec=False), cfg)
    assert result.ok
    assert result.payload == payload
    assert result.header is not None and not result.header.fec_enabled


def test_leading_silence_does_not_confuse_the_receiver(
    cfg: Config, payload: bytes
) -> None:
    wave = link.transmit(payload, cfg)
    delayed = np.concatenate([np.zeros(31_000), wave, np.zeros(9_000)])
    assert link.receive(delayed, cfg).payload == payload


def test_error_correction_beats_no_correction_under_noise(
    cfg: Config, payload: bytes
) -> None:
    """The headline claim: parity buys roughly five decibels.

    Checked at a noise level chosen to sit between the two thresholds, so the
    test fails if either one moves.
    """
    snr = 8.0
    coded = link.transmit(payload, cfg, use_fec=True)
    bare = link.transmit(payload, cfg, use_fec=False)

    with_fec = sum(
        link.receive(channel.awgn(coded, snr, np.random.default_rng(s)), cfg).ok
        for s in range(5)
    )
    without = sum(
        link.receive(channel.awgn(bare, snr, np.random.default_rng(s)), cfg).ok
        for s in range(5)
    )

    assert with_fec == 5
    assert without == 0


def test_survives_a_simulated_room(cfg: Config, payload: bytes) -> None:
    wave = link.transmit(payload, cfg)
    rx = channel.room(wave, cfg.fs, np.random.default_rng(7), snr_db=22, ppm=60)

    result = link.receive(rx, cfg)
    assert result.ok
    assert result.payload == payload


@pytest.mark.parametrize("ppm", [-400.0, -120.0, 120.0, 400.0])
def test_clock_drift_is_measured_and_undone(
    cfg: Config, payload: bytes, ppm: float
) -> None:
    """Without this correction the link does not work between two machines."""
    wave = link.transmit(payload, cfg)
    rx = channel.awgn(channel.clock_drift(wave, ppm), 25.0, np.random.default_rng(3))

    result = link.receive(rx, cfg)
    assert result.ok
    assert result.payload == payload
    assert result.drift_ppm == pytest.approx(ppm, abs=15.0)


def test_drift_correction_can_be_switched_off(cfg: Config, payload: bytes) -> None:
    wave = link.transmit(payload, cfg)
    rx = channel.clock_drift(wave, 400.0)

    assert link.receive(rx, cfg, correct_drift=False).drift_ppm == 0.0


def test_interference_bursts_are_absorbed(cfg: Config, payload: bytes) -> None:
    """What the interleaver is for: concentrated damage, spread thin."""
    wave = link.transmit(payload, cfg)
    rx = channel.room(
        wave, cfg.fs, np.random.default_rng(11), snr_db=26, ppm=40, bursts=4
    )

    result = link.receive(rx, cfg)
    assert result.ok
    assert result.fec is not None and result.fec.repaired > 0


def test_pure_noise_is_rejected_rather_than_decoded(cfg: Config, rng) -> None:
    """A receiver that invents a frame out of room noise is worse than useless."""
    result = link.receive(rng.normal(0, 0.1, 200_000), cfg)
    assert not result.ok
    assert result.reason


def test_silence_is_rejected(cfg: Config) -> None:
    result = link.receive(np.zeros(100_000), cfg)
    assert not result.ok


def test_truncated_transmission_is_reported(cfg: Config, payload: bytes) -> None:
    wave = link.transmit(payload, cfg)
    result = link.receive(wave[: len(wave) // 2], cfg)
    assert not result.ok
    assert result.reason


def test_signal_shorter_than_the_preamble(cfg: Config) -> None:
    result = link.receive(np.zeros(10), cfg)
    assert not result.ok


def test_summary_is_readable(cfg: Config, payload: bytes) -> None:
    ok = link.receive(link.transmit(payload, cfg), cfg)
    assert ok.summary().startswith("OK:")

    bad = link.receive(np.zeros(100_000), cfg)
    assert bad.summary().startswith("FAILED:")


def test_airtime_matches_the_waveform(cfg: Config) -> None:
    for size in (100, 1_000, 5_000):
        payload = bytes(size)
        wave = link.transmit(payload, cfg)
        assert link.airtime(size, cfg) == pytest.approx(len(wave) / cfg.fs, abs=1e-9)


def test_goodput_is_below_the_raw_bitrate(cfg: Config) -> None:
    assert 0 < link.goodput(10_000, cfg) < cfg.raw_bitrate


def test_works_with_a_non_default_configuration(fast_cfg: Config) -> None:
    payload = b"a smaller radio, same protocol" * 20
    result = link.receive(link.transmit(payload, fast_cfg), fast_cfg)
    assert result.ok
    assert result.payload == payload
