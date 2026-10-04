"""Channel impairments, so the hard parts can be debugged without a room.

Each function models one thing that a real acoustic link does to a signal.
They compose, and the order matters: a room echoes before a microphone adds
noise, and the receiver's clock drifts regardless of either.

Clock drift is the one that surprises people.  Two laptops nominally sampling
at 48 kHz differ by tens of parts per million, which sounds negligible until
you notice it accumulates: at 100 ppm a ten-second transmission ends a
millisecond out of step, the symbol window slides off the FFT boundary, and
the constellation smears.  It is the reason a link that works perfectly
through a WAV file falls apart between two machines.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "awgn",
    "multipath",
    "clock_drift",
    "resample",
    "bandpass_tilt",
    "burst_noise",
    "clip",
    "room",
]


def resample(signal: np.ndarray, ratio: float) -> np.ndarray:
    """Stretch or squeeze a signal by ``ratio``, keeping its content intact.

    ``ratio > 1`` lengthens.  This is the inverse of :func:`clock_drift` and
    the tool the receiver uses to undo a mismatched sample clock once it has
    measured one.
    """
    if ratio <= 0:
        raise ValueError("resampling ratio must be positive")
    if ratio == 1.0 or len(signal) == 0:
        return signal.copy()

    n_out = max(1, int(round(len(signal) * ratio)))
    positions = np.arange(n_out) / ratio
    return np.interp(positions, np.arange(len(signal)), signal)


def awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at a given signal-to-noise ratio.

    The ratio is measured against the mean power of the whole array, silence
    included, which makes it pessimistic compared with an SNR measured only
    over the active burst.
    """
    power = float(np.mean(signal**2))
    if power <= 0:
        return signal.copy()
    noise_power = power / (10 ** (snr_db / 10))
    return signal + rng.normal(0.0, np.sqrt(noise_power), signal.shape)


def multipath(
    signal: np.ndarray,
    delays: tuple[int, ...] = (0, 37, 112, 260),
    gains: tuple[float, ...] = (1.0, 0.45, 0.25, 0.12),
) -> np.ndarray:
    """Convolve with a sparse impulse response: a direct path plus echoes.

    The defaults are a small hard-walled room.  The longest echo here is 260
    samples, about 5.4 ms, which deliberately exceeds the default cyclic
    prefix -- real rooms do not respect your parameter choices.
    """
    if len(delays) != len(gains):
        raise ValueError("delays and gains must have matching lengths")

    out = np.zeros(len(signal) + max(delays), dtype=np.float64)
    for delay, gain in zip(delays, gains, strict=True):
        out[delay : delay + len(signal)] += gain * signal
    return out


def clock_drift(signal: np.ndarray, ppm: float) -> np.ndarray:
    """Resample as if the receiver's clock ran ``ppm`` parts per million fast.

    Linear interpolation is crude but the error it introduces is far below the
    effect being modelled, and it keeps the function dependency-free.
    """
    if ppm == 0.0:
        return signal.copy()

    scale = 1.0 + ppm * 1e-6
    n_out = int(len(signal) / scale)
    positions = np.arange(n_out) * scale
    return np.interp(positions, np.arange(len(signal)), signal)


def bandpass_tilt(
    signal: np.ndarray,
    fs: int,
    f_lo: float = 300.0,
    f_hi: float = 8_000.0,
    slope_db: float = 6.0,
) -> np.ndarray:
    """Apply the frequency response of a cheap speaker and microphone.

    Rolls off outside the passband and tilts the response across it, because
    no transducer in this price range is flat.
    """
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / fs)

    response = np.ones_like(freqs)
    response[freqs < f_lo] = 0.05
    response[freqs > f_hi] = 0.05

    inside = (freqs >= f_lo) & (freqs <= f_hi)
    if inside.any():
        span = np.clip((freqs[inside] - f_lo) / (f_hi - f_lo), 0.0, 1.0)
        response[inside] = 10 ** (-slope_db * span / 20)

    return np.fft.irfft(spectrum * response, n=len(signal))


def burst_noise(
    signal: np.ndarray,
    rng: np.random.Generator,
    n_bursts: int = 3,
    duration: int = 900,
    amplitude: float = 1.5,
) -> np.ndarray:
    """Drop short, loud interference on top of the signal.

    This is the impairment interleaving exists for: a chair scraping wipes out
    several consecutive OFDM symbols, which without interleaving would destroy
    whole codewords rather than scratching many.
    """
    out = signal.copy()
    if len(signal) <= duration:
        return out

    rms = float(np.sqrt(np.mean(signal**2))) or 1.0
    for _ in range(n_bursts):
        at = int(rng.integers(0, len(signal) - duration))
        out[at : at + duration] += rng.normal(0.0, amplitude * rms, duration)
    return out


def clip(signal: np.ndarray, ceiling: float = 1.0) -> np.ndarray:
    """Hard-limit the signal, as an overdriven sound card would."""
    return np.clip(signal, -ceiling, ceiling)


def room(
    signal: np.ndarray,
    fs: int,
    rng: np.random.Generator,
    *,
    snr_db: float = 20.0,
    ppm: float = 60.0,
    bursts: int = 0,
) -> np.ndarray:
    """Everything a real room does, applied in physical order.

    Echoes happen in the air, the transducers colour what survives, the
    microphone adds thermal noise, interference arrives whenever it likes, and
    the receiver's clock was never right to begin with.
    """
    out = multipath(signal)
    out = bandpass_tilt(out, fs)
    out = awgn(out, snr_db, rng)
    if bursts:
        out = burst_noise(out, rng, n_bursts=bursts)
    out = clock_drift(out, ppm)
    return clip(out)
