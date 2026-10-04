"""OFDM physical layer: bits in, waveform out, and back again.

Modulation is differential QPSK across time.  Each carrier compares its phase
against the *same carrier in the previous symbol*, so the receiver never needs
to know the absolute phase the room imposed on it.  That costs roughly 3 dB
against coherent detection and buys a demodulator with no channel estimator,
no pilot tones and no equaliser -- which is the right trade for a first
working link.

The cyclic prefix is what makes this survive a room at all.  Copying the tail
of each symbol onto its front means an echo arriving within the prefix window
lands on a copy of the signal rather than on the previous symbol.  Multipath
stops being intersymbol interference and becomes a per-carrier phase rotation,
which differential detection then cancels for free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config

__all__ = [
    "GRAY",
    "make_chirp",
    "bytes_to_bits",
    "bits_to_bytes",
    "symbols_for_bits",
    "symbols_for_bytes",
    "modulate_bits",
    "build_waveform",
    "postamble_position",
    "Sync",
    "find_preamble",
    "find_postamble",
    "demodulate_bits",
    "SignalQuality",
    "measure_quality",
]

GRAY = np.array([0, 1, 3, 2], dtype=np.int64)
"""Gray map between dibit value and QPSK quadrant.  Happens to be its own
inverse, so the same table serves the modulator and the demodulator.

A Gray map matters here because the most likely error is a slip into an
adjacent quadrant, and adjacent quadrants differ by exactly one bit."""

_ROT = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j])
"""Phase step applied for each quadrant index."""


# ------------------------------------------------------------------- sync


def make_chirp(cfg: Config, *, descending: bool = False) -> np.ndarray:
    """Build a linear frequency sweep used as a frame marker.

    A chirp is used rather than a tone burst because its autocorrelation is a
    sharp spike: sweeping across the band means every frequency component
    lines up at exactly one lag.  That gives sample-accurate timing from a
    single correlation, with no search.

    The descending variant marks the end of a frame.  Sweeping the other way
    makes it nearly orthogonal to the opening chirp, so neither can be
    mistaken for the other.
    """
    n = cfg.chirp_len
    t = np.arange(n) / cfg.fs
    duration = n / cfg.fs

    lo, hi = (cfg.f_hi, cfg.f_lo) if descending else (cfg.f_lo, cfg.f_hi)
    rate = (hi - lo) / duration
    phase = 2 * np.pi * (lo * t + 0.5 * rate * t**2)
    return np.cos(phase) * np.hanning(n)


def _fft_correlate(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Matched filter via FFT, indexed by the template's start position."""
    span = len(signal) - len(template) + 1
    if span <= 0:
        return np.empty(0)
    n_fft = 1 << (len(signal) + len(template) - 2).bit_length()
    acc = np.fft.rfft(signal, n_fft) * np.conj(np.fft.rfft(template, n_fft))
    return np.fft.irfft(acc, n_fft)[:span]


@dataclass(frozen=True)
class Sync:
    """Result of preamble detection.

    Attributes
    ----------
    offset:
        Sample index where the first OFDM symbol begins.
    peak:
        Correlation magnitude at the detected position.
    confidence:
        Peak divided by the median correlation.  Above roughly 10 the
        detection is solid; near 1 there is no preamble in the signal.
    """

    offset: int
    peak: float
    confidence: float


def find_preamble(cfg: Config, rx: np.ndarray) -> Sync:
    """Locate the opening chirp and return where the payload starts."""
    corr = np.abs(_fft_correlate(rx, make_chirp(cfg)))
    if corr.size == 0:
        raise ValueError("signal is shorter than the preamble")

    at = int(np.argmax(corr))
    peak = float(corr[at])
    floor = float(np.median(corr)) or 1e-12

    return Sync(
        offset=at + cfg.chirp_len + cfg.guard_len,
        peak=peak,
        confidence=peak / floor,
    )


def find_postamble(cfg: Config, rx: np.ndarray, expect_at: int, radius: int) -> Sync:
    """Locate the closing chirp within ``radius`` samples of ``expect_at``.

    The search is windowed because the answer is only interesting if it lands
    near where the frame says it should; a match far away is a reflection or a
    coincidence, not a frame boundary.
    """
    corr = np.abs(_fft_correlate(rx, make_chirp(cfg, descending=True)))
    if corr.size == 0:
        raise ValueError("signal is shorter than the postamble")

    lo = max(0, expect_at - radius)
    hi = min(len(corr), expect_at + radius + 1)
    if hi <= lo:
        raise ValueError("postamble search window falls outside the signal")

    window = corr[lo:hi]
    at = lo + int(np.argmax(window))
    peak = float(corr[at])
    floor = float(np.median(corr)) or 1e-12

    return Sync(offset=at, peak=peak, confidence=peak / floor)


# -------------------------------------------------------------- bit helpers


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Unpack bytes into a big-endian bit array."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack a big-endian bit array back into bytes."""
    return np.packbits(bits).tobytes()


def symbols_for_bits(cfg: Config, n_bits: int) -> int:
    """OFDM symbols needed to carry ``n_bits`` bits."""
    return -(-n_bits // cfg.bits_per_symbol)


def symbols_for_bytes(cfg: Config, n_bytes: int) -> int:
    """OFDM symbols needed to carry ``n_bytes`` bytes."""
    return symbols_for_bits(cfg, n_bytes * 8)


# --------------------------------------------------------------- modulator


def modulate_bits(cfg: Config, bits: np.ndarray) -> np.ndarray:
    """Turn a bit array into the OFDM body, without preamble or silence.

    The body opens with an unmodulated reference symbol.  Differential
    detection needs something to differentiate against, and spending one
    symbol on it is cheaper than embedding pilots in every symbol.
    """
    pad = (-len(bits)) % cfg.bits_per_symbol
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    dibits = bits.reshape(-1, 2).astype(np.int64)
    steps = _ROT[GRAY[dibits[:, 0] * 2 + dibits[:, 1]]].reshape(-1, cfg.n_carriers)

    cells = np.empty((steps.shape[0] + 1, cfg.n_carriers), dtype=complex)
    cells[0] = 1.0
    cells[1:] = np.cumprod(steps, axis=0)

    spectrum = np.zeros((cells.shape[0], cfg.n_fft // 2 + 1), dtype=complex)
    spectrum[:, cfg.carriers] = cells
    frames = np.fft.irfft(spectrum, n=cfg.n_fft, axis=1)

    with_cp = np.concatenate([frames[:, -cfg.cp_len :], frames], axis=1)
    body = with_cp.reshape(-1)

    peak = np.max(np.abs(body))
    if peak > 0:
        body = body / peak
    return body


def build_waveform(cfg: Config, bits: np.ndarray) -> np.ndarray:
    """Wrap the OFDM body in preamble, postamble, guards and silence.

    The closing chirp is what makes clock-drift correction possible: the
    receiver measures the distance between the two markers, compares it with
    the distance the frame says it should be, and learns exactly how fast its
    own clock is running.
    """
    cfg.validate()
    return np.concatenate(
        [
            np.zeros(cfg.lead_silence),
            make_chirp(cfg) * cfg.amplitude,
            np.zeros(cfg.guard_len),
            modulate_bits(cfg, bits) * cfg.amplitude,
            np.zeros(cfg.guard_len),
            make_chirp(cfg, descending=True) * cfg.amplitude,
            np.zeros(cfg.lead_silence),
        ]
    )


def postamble_position(cfg: Config, n_symbols: int) -> int:
    """Where the closing chirp starts, measured from the opening chirp.

    ``n_symbols`` counts data symbols; the reference symbol is added here.
    """
    return (
        cfg.chirp_len
        + cfg.guard_len
        + (n_symbols + 1) * cfg.sym_len
        + cfg.guard_len
    )


# ------------------------------------------------------------- demodulator


def _cells(cfg: Config, rx: np.ndarray, offset: int, n_frames: int) -> np.ndarray:
    """Extract the active carriers from ``n_frames`` OFDM symbols."""
    need = offset + n_frames * cfg.sym_len
    if offset < 0 or need > len(rx):
        raise ValueError(
            f"signal ends early: need {need} samples, have {len(rx)}"
        )
    span = rx[offset : offset + n_frames * cfg.sym_len]
    frames = span.reshape(n_frames, cfg.sym_len)[:, cfg.cp_len :]
    return np.fft.rfft(frames, axis=1)[:, cfg.carriers]


def _differential(cells: np.ndarray) -> np.ndarray:
    """Phase difference between consecutive symbols, per carrier."""
    return cells[1:] * np.conj(cells[:-1])


def demodulate_bits(
    cfg: Config, rx: np.ndarray, offset: int, n_symbols: int
) -> np.ndarray:
    """Recover ``n_symbols`` worth of data bits starting at ``offset``.

    ``offset`` points at the reference symbol, so ``n_symbols + 1`` OFDM
    symbols are consumed.
    """
    if n_symbols < 1:
        return np.empty(0, dtype=np.uint8)

    diff = _differential(_cells(cfg, rx, offset, n_symbols + 1))
    quadrant = np.mod(np.rint(np.angle(diff) / (np.pi / 2)).astype(np.int64), 4)
    dibit = GRAY[quadrant]

    return np.stack([dibit >> 1, dibit & 1], axis=-1).reshape(-1).astype(np.uint8)


# ----------------------------------------------------------- link quality


@dataclass(frozen=True)
class SignalQuality:
    """Constellation health, measured on the received symbols.

    Attributes
    ----------
    evm_degrees:
        RMS phase error against the nearest ideal quadrant.  Below about 10
        degrees the link is comfortable; past 25 the decisions start slipping.
    est_snr_db:
        Signal-to-noise ratio implied by that phase error.
    """

    evm_degrees: float
    est_snr_db: float


def measure_quality(
    cfg: Config, rx: np.ndarray, offset: int, n_symbols: int
) -> SignalQuality:
    """Estimate link quality from the spread of the received constellation."""
    diff = _differential(_cells(cfg, rx, offset, n_symbols + 1))
    angle = np.angle(diff)
    error = angle - np.rint(angle / (np.pi / 2)) * (np.pi / 2)

    rms = float(np.sqrt(np.mean(error**2)))
    evm_deg = np.degrees(rms)
    snr = float("inf") if rms <= 0 else -10.0 * np.log10(2.0 * rms**2)

    return SignalQuality(evm_degrees=evm_deg, est_snr_db=snr)
