"""Physical layer parameters and their derived quantities."""

from __future__ import annotations

import pytest

from bora import Config


def test_defaults_are_self_consistent(cfg: Config) -> None:
    cfg.validate()
    assert cfg.n_carriers == cfg.bin_hi - cfg.bin_lo + 1
    assert cfg.bits_per_symbol == cfg.n_carriers * 2
    assert cfg.sym_len == cfg.n_fft + cfg.cp_len


def test_carriers_stay_inside_the_requested_band(cfg: Config) -> None:
    lowest = cfg.bin_lo * cfg.carrier_spacing
    highest = cfg.bin_hi * cfg.carrier_spacing
    assert lowest >= cfg.f_lo
    assert highest <= cfg.f_hi


def test_raw_bitrate_matches_the_documented_figure(cfg: Config) -> None:
    assert cfg.raw_bitrate == pytest.approx(8_100.0)


def test_carriers_never_reach_nyquist(cfg: Config) -> None:
    assert cfg.bin_hi < cfg.n_fft // 2


@pytest.mark.parametrize(
    "broken",
    [
        {"f_lo": 6_000.0, "f_hi": 1_000.0},
        {"f_hi": 30_000.0},
        {"cp_len": 512},
        {"amplitude": 0.0},
        {"amplitude": 1.5},
    ],
)
def test_validate_rejects_unusable_parameters(broken: dict) -> None:
    with pytest.raises(ValueError):
        Config(**broken).validate()


def test_describe_mentions_the_key_numbers(cfg: Config) -> None:
    text = cfg.describe()
    assert "carriers" in text
    assert "cyclic prefix" in text
    assert str(cfg.fs) in text
