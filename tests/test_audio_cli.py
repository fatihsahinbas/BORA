"""WAV round-tripping and the command line entry points."""

from __future__ import annotations

import wave as wavelib
from pathlib import Path

import numpy as np
import pytest

from bora import Config, audio, cli, link


def test_wav_roundtrip(tmp_path: Path, cfg: Config, rng) -> None:
    signal = np.clip(rng.normal(0, 0.3, 48_000), -1.0, 1.0)
    path = str(tmp_path / "x.wav")

    audio.write_wav(path, signal, cfg)
    back, fs = audio.read_wav(path)

    assert fs == cfg.fs
    assert len(back) == len(signal)
    assert np.max(np.abs(back - signal)) < 1e-4  # 16-bit quantisation


def test_wav_clips_rather_than_wrapping(tmp_path: Path, cfg: Config) -> None:
    """Samples past full scale must saturate, not wrap around.

    Integer wraparound would turn a slightly too-loud transmission into
    something unrecognisable instead of merely distorted.
    """
    path = str(tmp_path / "loud.wav")
    audio.write_wav(path, np.array([2.0, -2.0, 0.0]), cfg)

    back, _ = audio.read_wav(path)
    assert back[0] == pytest.approx(1.0, abs=1e-3)
    assert back[1] == pytest.approx(-1.0, abs=1e-3)


def test_wav_survives_a_full_transmission(tmp_path: Path, cfg: Config,
                                          payload: bytes) -> None:
    path = str(tmp_path / "frame.wav")
    audio.write_wav(path, link.transmit(payload, cfg), cfg)

    back, _ = audio.read_wav(path)
    assert link.receive(back, cfg).payload == payload


def test_stereo_is_mixed_down(tmp_path: Path, cfg: Config) -> None:
    path = str(tmp_path / "stereo.wav")
    mono = (np.sin(np.arange(4_800) * 0.1) * 16_000).astype("<i2")
    stereo = np.repeat(mono, 2)

    with wavelib.open(path, "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(cfg.fs)
        fh.writeframes(stereo.tobytes())

    back, _ = audio.read_wav(path)
    assert len(back) == len(mono)


def test_eight_bit_wav_is_rejected(tmp_path: Path, cfg: Config) -> None:
    path = str(tmp_path / "8bit.wav")
    with wavelib.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(1)
        fh.setframerate(cfg.fs)
        fh.writeframes(b"\x80" * 100)

    with pytest.raises(ValueError):
        audio.read_wav(path)


# ------------------------------------------------------------------- the CLI


def test_cli_info(capsys) -> None:
    assert cli.main(["info"]) == 0
    assert "raw bitrate" in capsys.readouterr().out


def test_cli_send_then_recv(tmp_path: Path, capsys) -> None:
    source = tmp_path / "message.txt"
    source.write_bytes(b"sent across the room as sound" * 30)
    carrier = tmp_path / "out.wav"
    recovered = tmp_path / "back.txt"

    assert cli.main(["send", str(source), "-o", str(carrier)]) == 0
    assert cli.main(["recv", str(carrier), "-o", str(recovered)]) == 0
    assert recovered.read_bytes() == source.read_bytes()

    assert "OK" in capsys.readouterr().out


def test_cli_send_without_fec(tmp_path: Path) -> None:
    source = tmp_path / "m.bin"
    source.write_bytes(bytes(range(256)) * 4)
    carrier = tmp_path / "raw.wav"
    back = tmp_path / "raw.bin"

    assert cli.main(["send", str(source), "-o", str(carrier), "--no-fec"]) == 0
    assert cli.main(["recv", str(carrier), "-o", str(back)]) == 0
    assert back.read_bytes() == source.read_bytes()


def test_cli_loopback_succeeds_at_a_workable_noise_level(capsys) -> None:
    assert cli.main(["loopback", "--size", "600", "--snr", "15"]) == 0
    assert "identical    True" in capsys.readouterr().out


def test_cli_loopback_reports_failure_in_the_exit_code() -> None:
    assert cli.main(["loopback", "--size", "600", "--snr", "-10"]) == 1


def test_cli_sweep_writes_csv(tmp_path: Path) -> None:
    out = tmp_path / "ber.csv"
    code = cli.main(
        ["sweep", "--size", "300", "--trials", "1", "--snr-points", "14", "8",
         "--out", str(out)]
    )
    assert code == 0

    lines = out.read_text().strip().splitlines()
    assert lines[0] == "snr_db,fec,success,trials,ber"
    assert len(lines) == 5  # header plus two SNR points, twice


def test_cli_reports_a_missing_file(capsys) -> None:
    assert cli.main(["send", "does-not-exist.bin"]) == 2
    assert "error:" in capsys.readouterr().err
