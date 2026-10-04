"""The full stack: bytes to waveform and back.

This is the layer users touch.  Everything below it -- :mod:`bora.modem` for
the physical layer, :mod:`bora.fec` for error correction, :mod:`bora.framing`
for the wire format -- is separately testable and separately replaceable.

Reception runs in three passes, and each exists because the one before it
cannot finish the job:

1. Find the opening chirp and decode the header.  The header is only a handful
   of symbols long, so it is still readable even when the rest of the frame is
   not: whatever is wrong with the clock has had almost no time to accumulate.

2. The header gives the payload length, which gives the exact position the
   closing chirp should occupy.  Comparing that with where the closing chirp
   actually is measures the sample-rate error between the two machines
   directly, in parts per million, with no tracking loop.  Resample by that
   ratio and the error is gone.

3. Decode the frame properly against the corrected signal.

The middle pass is the difference between a modem that works through a WAV
file and one that works between two laptops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import framing, modem
from .channel import resample
from .config import DEFAULT, Config
from .fec import DecodeReport, FecError, coded_length
from .framing import FrameError, Header

__all__ = ["Reception", "transmit", "receive", "airtime", "goodput"]

_DRIFT_SEARCH_PPM = 3_000.0
"""How far off the closing chirp may be before the match is disbelieved.

Three thousand ppm is far wider than any real sound card -- consumer crystals
land within a few hundred -- but a wide window costs nothing and a narrow one
silently gives up on a machine that genuinely is that bad."""

_DRIFT_MIN_CONFIDENCE = 6.0
"""Correlation peak-to-median below which the closing chirp is not believed."""

_DRIFT_MIN_PPM = 1.0
"""Below this, correcting costs an interpolation and buys nothing."""


@dataclass
class Reception:
    """Everything the receiver learned about one transmission.

    Attributes
    ----------
    payload:
        Recovered bytes.  Present even when ``ok`` is false, so a partially
        corrupted image can still be inspected.
    ok:
        True when the payload matches the CRC the sender recorded.
    reason:
        Why the reception failed, or an empty string on success.
    header:
        The decoded frame header, if it survived.
    sync:
        Opening chirp detection result.
    quality:
        Constellation health across the whole frame.
    fec:
        Per-codeword correction statistics.
    drift_ppm:
        Measured sample clock error between transmitter and receiver.  Zero
        when no closing chirp was found or correction was disabled.
    """

    payload: bytes = b""
    ok: bool = False
    reason: str = ""
    header: Header | None = None
    sync: modem.Sync | None = None
    quality: modem.SignalQuality | None = None
    fec: DecodeReport | None = None
    drift_ppm: float = 0.0

    def summary(self) -> str:
        """One-line human readable verdict."""
        if self.ok:
            detail = f"{len(self.payload)} bytes"
            if self.fec and self.fec.repaired:
                detail += (
                    f", repaired {self.fec.corrected_bytes} byte errors "
                    f"in {self.fec.repaired}/{self.fec.blocks} blocks"
                )
            return f"OK: {detail}"
        return f"FAILED: {self.reason}"


def transmit(
    payload: bytes, cfg: Config = DEFAULT, *, use_fec: bool = True
) -> np.ndarray:
    """Encode ``payload`` into a transmittable waveform."""
    frame = framing.build_frame(payload, use_fec=use_fec)
    return modem.build_waveform(cfg, modem.bytes_to_bits(frame))


def _read_header(
    cfg: Config, rx: np.ndarray, sync: modem.Sync
) -> tuple[Header, int]:
    """Decode the header and return it with the frame's total symbol count."""
    available = (len(rx) - sync.offset) // cfg.sym_len - 1
    if available < 1:
        raise FrameError("no OFDM symbols follow the preamble")

    header_syms = modem.symbols_for_bytes(cfg, framing.HEADER_CODED_LEN)
    if available < header_syms:
        raise FrameError("signal ends inside the header")

    bits = modem.demodulate_bits(cfg, rx, sync.offset, header_syms)
    coded = modem.bits_to_bytes(bits)[: framing.HEADER_CODED_LEN]
    header = framing.parse_header(coded)

    on_air = framing.HEADER_CODED_LEN + header.payload_bytes_on_air()
    return header, modem.symbols_for_bytes(cfg, on_air)


def _measure_drift(
    cfg: Config, rx: np.ndarray, sync: modem.Sync, total_syms: int
) -> tuple[float, float]:
    """Return ``(ppm, ratio)`` describing the receiver's sample clock error.

    A ratio of exactly 1.0 means either that the clocks agree or that the
    closing chirp could not be trusted; the caller treats both the same way.
    """
    opening_at = sync.offset - cfg.chirp_len - cfg.guard_len
    nominal = modem.postamble_position(cfg, total_syms)
    radius = max(cfg.chirp_len // 2, int(nominal * _DRIFT_SEARCH_PPM * 1e-6))

    try:
        closing = modem.find_postamble(cfg, rx, opening_at + nominal, radius)
    except ValueError:
        return 0.0, 1.0

    if closing.confidence < _DRIFT_MIN_CONFIDENCE:
        return 0.0, 1.0

    measured = closing.offset - opening_at
    if measured <= 0:
        return 0.0, 1.0

    ratio = nominal / measured
    return (ratio - 1.0) * 1e6, ratio


def receive(
    rx: np.ndarray, cfg: Config = DEFAULT, *, correct_drift: bool = True
) -> Reception:
    """Decode a received waveform back into bytes."""
    cfg.validate()
    rx = np.asarray(rx, dtype=np.float64)

    try:
        sync = modem.find_preamble(cfg, rx)
    except ValueError as exc:
        return Reception(reason=str(exc))

    try:
        header, total_syms = _read_header(cfg, rx, sync)
    except (FecError, FrameError) as exc:
        return Reception(sync=sync, reason=f"header unreadable ({exc})")

    drift_ppm = 0.0
    if correct_drift:
        drift_ppm, ratio = _measure_drift(cfg, rx, sync, total_syms)
        if abs(drift_ppm) >= _DRIFT_MIN_PPM:
            rx = resample(rx, ratio)
            try:
                sync = modem.find_preamble(cfg, rx)
                header, total_syms = _read_header(cfg, rx, sync)
            except (ValueError, FecError, FrameError) as exc:
                return Reception(
                    sync=sync,
                    drift_ppm=drift_ppm,
                    reason=f"header unreadable after drift correction ({exc})",
                )

    available = (len(rx) - sync.offset) // cfg.sym_len - 1
    truncated = total_syms > available
    read_syms = max(1, min(total_syms, available))

    bits = modem.demodulate_bits(cfg, rx, sync.offset, read_syms)
    stream = modem.bits_to_bytes(bits)
    on_air = framing.HEADER_CODED_LEN + header.payload_bytes_on_air()
    body = stream[framing.HEADER_CODED_LEN : on_air]

    quality = modem.measure_quality(cfg, rx, sync.offset, read_syms)

    try:
        payload, report = framing.parse_payload(header, body)
    except FecError as exc:
        return Reception(
            sync=sync,
            header=header,
            quality=quality,
            drift_ppm=drift_ppm,
            reason=f"payload unreadable ({exc})",
        )

    ok = framing.crc_matches(header, payload)
    if ok:
        reason = ""
    elif truncated:
        reason = "signal ends before the frame does"
    elif report.failed:
        reason = f"{report.failed}/{report.blocks} blocks beyond repair"
    else:
        reason = "CRC mismatch"

    return Reception(
        payload=payload,
        ok=ok,
        reason=reason,
        header=header,
        sync=sync,
        quality=quality,
        fec=report,
        drift_ppm=drift_ppm,
    )


def frame_samples(n_bytes: int, cfg: Config = DEFAULT, *, use_fec: bool = True) -> int:
    """Total waveform length in samples for a payload of ``n_bytes`` bytes."""
    on_air = framing.HEADER_CODED_LEN + (
        coded_length(n_bytes) if use_fec else n_bytes
    )
    symbols = modem.symbols_for_bytes(cfg, on_air) + 1
    return (
        2 * cfg.lead_silence
        + 2 * cfg.chirp_len
        + 2 * cfg.guard_len
        + symbols * cfg.sym_len
    )


def airtime(n_bytes: int, cfg: Config = DEFAULT, *, use_fec: bool = True) -> float:
    """Seconds on air for a payload of ``n_bytes`` bytes."""
    return frame_samples(n_bytes, cfg, use_fec=use_fec) / cfg.fs


def goodput(n_bytes: int, cfg: Config = DEFAULT, *, use_fec: bool = True) -> float:
    """Useful bits per second, counting preamble and parity as overhead."""
    seconds = airtime(n_bytes, cfg, use_fec=use_fec)
    return n_bytes * 8 / seconds if seconds > 0 else 0.0
