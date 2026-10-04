"""OFDM modulation, chirp synchronisation and constellation measurement."""

from __future__ import annotations

import numpy as np
import pytest

from bora import Config, modem


def test_bit_packing_roundtrip(payload: bytes) -> None:
    assert modem.bits_to_bytes(modem.bytes_to_bits(payload)) == payload


def test_modulate_demodulate_is_lossless(cfg: Config, rng) -> None:
    bits = rng.integers(0, 2, cfg.bits_per_symbol * 12).astype(np.uint8)
    body = modem.modulate_bits(cfg, bits)

    recovered = modem.demodulate_bits(cfg, body, 0, 12)
    assert np.array_equal(recovered, bits)


def test_gray_map_is_its_own_inverse() -> None:
    assert np.array_equal(modem.GRAY[modem.GRAY], np.arange(4))


def test_gray_map_makes_neighbours_differ_by_one_bit() -> None:
    """Adjacent quadrants must differ in exactly one bit.

    The dominant error is a slip into the neighbouring quadrant, so this
    property is what keeps one symbol error from costing two bit errors.
    """
    for quadrant in range(4):
        here = modem.GRAY[quadrant]
        there = modem.GRAY[(quadrant + 1) % 4]
        assert bin(here ^ there).count("1") == 1


def test_waveform_stays_within_the_amplitude_limit(cfg: Config, rng) -> None:
    bits = rng.integers(0, 2, cfg.bits_per_symbol * 5).astype(np.uint8)
    wave = modem.build_waveform(cfg, bits)
    assert np.max(np.abs(wave)) <= cfg.amplitude + 1e-9


def test_chirp_occupies_the_configured_band(cfg: Config) -> None:
    chirp = modem.make_chirp(cfg)
    spectrum = np.abs(np.fft.rfft(chirp))
    freqs = np.fft.rfftfreq(len(chirp), 1 / cfg.fs)

    inside = (freqs >= cfg.f_lo) & (freqs <= cfg.f_hi)
    assert spectrum[inside].sum() > 20 * spectrum[~inside].sum()


def test_descending_chirp_is_nearly_orthogonal_to_the_opening_one(cfg: Config) -> None:
    """The two markers must not be mistakable for each other."""
    up = modem.make_chirp(cfg)
    down = modem.make_chirp(cfg, descending=True)

    cross = np.abs(np.correlate(up, down, mode="full")).max()
    auto = np.abs(np.correlate(up, up, mode="full")).max()
    assert cross < 0.35 * auto


def test_preamble_lands_on_the_first_symbol(cfg: Config, rng) -> None:
    bits = rng.integers(0, 2, cfg.bits_per_symbol * 6).astype(np.uint8)
    wave = modem.build_waveform(cfg, bits)

    sync = modem.find_preamble(cfg, wave)
    expected = cfg.lead_silence + cfg.chirp_len + cfg.guard_len

    assert abs(sync.offset - expected) <= 1
    assert sync.confidence > 20


def test_preamble_found_after_arbitrary_leading_silence(cfg: Config, rng) -> None:
    bits = rng.integers(0, 2, cfg.bits_per_symbol * 4).astype(np.uint8)
    wave = modem.build_waveform(cfg, bits)
    delayed = np.concatenate([np.zeros(7_777), wave])

    sync = modem.find_preamble(cfg, delayed)
    assert abs(sync.offset - (7_777 + cfg.lead_silence + cfg.chirp_len
                              + cfg.guard_len)) <= 1


def test_postamble_sits_where_the_frame_says(cfg: Config, rng) -> None:
    n_symbols = 9
    bits = rng.integers(0, 2, cfg.bits_per_symbol * n_symbols).astype(np.uint8)
    wave = modem.build_waveform(cfg, bits)

    opening = cfg.lead_silence
    expected = opening + modem.postamble_position(cfg, n_symbols)
    found = modem.find_postamble(cfg, wave, expected, radius=200)

    assert abs(found.offset - expected) <= 1
    assert found.confidence > 10


def test_quality_is_perfect_on_a_clean_signal(cfg: Config, rng) -> None:
    bits = rng.integers(0, 2, cfg.bits_per_symbol * 8).astype(np.uint8)
    body = modem.modulate_bits(cfg, bits)

    quality = modem.measure_quality(cfg, body, 0, 8)
    assert quality.evm_degrees < 1e-6


def test_quality_degrades_with_noise(cfg: Config, rng) -> None:
    bits = rng.integers(0, 2, cfg.bits_per_symbol * 8).astype(np.uint8)
    body = modem.modulate_bits(cfg, bits)
    noisy = body + rng.normal(0, 0.1 * np.sqrt(np.mean(body**2)), len(body))

    clean_evm = modem.measure_quality(cfg, body, 0, 8).evm_degrees
    noisy_evm = modem.measure_quality(cfg, noisy, 0, 8).evm_degrees
    assert noisy_evm > clean_evm


@pytest.mark.parametrize("n_bits,expected", [(0, 0), (1, 1), (108, 1), (109, 2)])
def test_symbol_accounting(cfg: Config, n_bits: int, expected: int) -> None:
    assert modem.symbols_for_bits(cfg, n_bits) == expected


def test_demodulate_rejects_a_truncated_signal(cfg: Config) -> None:
    with pytest.raises(ValueError):
        modem.demodulate_bits(cfg, np.zeros(100), 0, 4)


def test_works_with_a_non_default_configuration(fast_cfg: Config, rng) -> None:
    bits = rng.integers(0, 2, fast_cfg.bits_per_symbol * 5).astype(np.uint8)
    body = modem.modulate_bits(fast_cfg, bits)
    assert np.array_equal(modem.demodulate_bits(fast_cfg, body, 0, 5), bits)
