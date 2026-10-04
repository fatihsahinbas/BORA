"""WAV file input and output, and optional live audio.

WAV support uses the standard library only.  Live playback and capture need
`sounddevice`, which pulls in PortAudio; it is an optional extra, imported
lazily so that everything else works on a machine with no sound card at all --
a CI runner, for instance.
"""

from __future__ import annotations

import wave

import numpy as np

from .config import DEFAULT, Config

__all__ = ["write_wav", "read_wav", "play", "record", "loopback_device"]


def write_wav(path: str, signal: np.ndarray, cfg: Config = DEFAULT) -> None:
    """Write a float signal to a 16-bit mono WAV file."""
    pcm = (np.clip(signal, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(cfg.fs)
        fh.writeframes(pcm.tobytes())


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a mono or multi-channel WAV file, returning floats and the rate.

    Multi-channel files are mixed down, because the modem has no use for
    stereo and a user recording on a laptop will often get two channels
    without asking for them.
    """
    with wave.open(path, "rb") as fh:
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        fs = fh.getframerate()
        raw = fh.readframes(fh.getnframes())

    if width != 2:
        raise ValueError(f"only 16-bit WAV files are supported, got {width * 8}-bit")

    samples = np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, fs


def _sounddevice():
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "live audio needs the optional 'sounddevice' package and a working "
            "PortAudio install: pip install 'bora[audio]'"
        ) from exc
    return sounddevice


def play(signal: np.ndarray, cfg: Config = DEFAULT, *, blocking: bool = True) -> None:
    """Play a signal through the default output device."""
    sd = _sounddevice()
    sd.play(np.clip(signal, -1.0, 1.0).astype(np.float32), cfg.fs)
    if blocking:
        sd.wait()


def record(seconds: float, cfg: Config = DEFAULT) -> np.ndarray:
    """Capture mono audio from the default input device."""
    sd = _sounddevice()
    frames = int(seconds * cfg.fs)
    buf = sd.rec(frames, samplerate=cfg.fs, channels=1, dtype="float32")
    sd.wait()
    return buf.reshape(-1).astype(np.float64)


def loopback_device(
    signal: np.ndarray, cfg: Config = DEFAULT, *, tail: float = 0.5
) -> np.ndarray:
    """Play and record at the same time, through the actual speaker and mic.

    This is the moment the simulated channel stops being a model and the real
    one takes over.  Expect it to fail the first time.
    """
    sd = _sounddevice()
    padded = np.concatenate([signal, np.zeros(int(tail * cfg.fs))])
    captured = sd.playrec(
        np.clip(padded, -1.0, 1.0).astype(np.float32),
        samplerate=cfg.fs,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return captured.reshape(-1).astype(np.float64)
