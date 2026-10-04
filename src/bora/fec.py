"""Forward error correction: Reed-Solomon over GF(256) plus a block interleaver.

Why both
--------
Reed-Solomon fixes *byte* errors, and it fixes at most ``nsym // 2`` of them
per codeword.  An acoustic channel does not spread its errors politely: a
notch in the room response kills the same handful of carriers in every OFDM
symbol, and a door slam wipes out several consecutive symbols.  Either way the
damage lands in one place.

The interleaver spreads that damage.  Codewords are stacked as the rows of a
matrix and transmitted column by column, so bytes that travel next to each
other belong to *different* codewords.  A burst that would have destroyed one
codeword instead scratches many, and each scratch is well inside what RS can
repair.

That pairing is the whole trick, and it is why the corrected error rate falls
off a cliff rather than degrading smoothly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - exercised by the import, not by the tests
    import reedsolo
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "bora requires the 'reedsolo' package for forward error correction; "
        "install it with: pip install reedsolo"
    ) from exc

__all__ = [
    "BLOCK_LEN",
    "PAYLOAD_PARITY",
    "PAYLOAD_DATA",
    "HEADER_PARITY",
    "FecError",
    "DecodeReport",
    "encode_blocks",
    "decode_blocks",
    "encode_header",
    "decode_header",
    "interleave",
    "deinterleave",
    "coded_length",
]

BLOCK_LEN = 255
"""Reed-Solomon codeword length in bytes.  Fixed by GF(256)."""

PAYLOAD_PARITY = 32
"""Parity bytes per payload codeword.  Corrects 16 byte errors."""

PAYLOAD_DATA = BLOCK_LEN - PAYLOAD_PARITY
"""Data bytes per payload codeword."""

HEADER_PARITY = 42
"""Parity bytes on the 12-byte header.  Corrects 21 byte errors."""


class FecError(Exception):
    """Raised when a codeword carries more damage than the code can repair."""


@dataclass
class DecodeReport:
    """Outcome of a payload decode.

    Attributes
    ----------
    data:
        The recovered bytes, truncated to the requested length.
    blocks:
        Number of codewords processed.
    repaired:
        Number of codewords that needed correction.
    failed:
        Number of codewords the code could not repair.
    corrected_bytes:
        Total byte errors fixed across all codewords.
    """

    data: bytes
    blocks: int
    repaired: int
    failed: int
    corrected_bytes: int

    @property
    def ok(self) -> bool:
        """True when every codeword decoded cleanly."""
        return self.failed == 0


def _codec(parity: int) -> reedsolo.RSCodec:
    return reedsolo.RSCodec(parity)


# ------------------------------------------------------------- interleaving


def interleave(blocks: np.ndarray) -> bytes:
    """Read a ``(n_blocks, block_len)`` array out column by column.

    Byte ``i`` of every codeword is emitted before byte ``i + 1`` of any of
    them, so a burst of ``n_blocks`` consecutive channel errors costs each
    codeword exactly one byte.
    """
    return blocks.T.reshape(-1).tobytes()


def deinterleave(stream: bytes, n_blocks: int, block_len: int = BLOCK_LEN) -> np.ndarray:
    """Inverse of :func:`interleave`."""
    expected = n_blocks * block_len
    if len(stream) < expected:
        raise FecError(
            f"interleaved stream is short: got {len(stream)}, need {expected}"
        )
    flat = np.frombuffer(stream[:expected], dtype=np.uint8)
    return flat.reshape(block_len, n_blocks).T


# ------------------------------------------------------------- payload path


def block_count(length: int) -> int:
    """Number of codewords needed to carry ``length`` payload bytes."""
    if length == 0:
        return 0
    return -(-length // PAYLOAD_DATA)


def coded_length(length: int) -> int:
    """On-air byte count for a payload of ``length`` bytes."""
    return block_count(length) * BLOCK_LEN


def encode_blocks(payload: bytes) -> bytes:
    """Reed-Solomon encode and interleave a payload.

    The final codeword is zero-padded up to :data:`PAYLOAD_DATA`; the receiver
    trims it using the length carried in the header.
    """
    n_blocks = block_count(len(payload))
    if n_blocks == 0:
        return b""

    padded = payload.ljust(n_blocks * PAYLOAD_DATA, b"\x00")
    codec = _codec(PAYLOAD_PARITY)

    blocks = np.empty((n_blocks, BLOCK_LEN), dtype=np.uint8)
    for i in range(n_blocks):
        chunk = padded[i * PAYLOAD_DATA : (i + 1) * PAYLOAD_DATA]
        blocks[i] = np.frombuffer(bytes(codec.encode(chunk)), dtype=np.uint8)

    return interleave(blocks)


def decode_blocks(stream: bytes, length: int) -> DecodeReport:
    """Deinterleave and Reed-Solomon decode a payload.

    A codeword that cannot be repaired is passed through with its parity
    stripped rather than aborting the whole frame.  The caller still has the
    CRC to tell it whether the result is trustworthy, and a partially correct
    image is more useful than an exception.
    """
    n_blocks = block_count(length)
    if n_blocks == 0:
        return DecodeReport(b"", 0, 0, 0, 0)

    blocks = deinterleave(stream, n_blocks)
    codec = _codec(PAYLOAD_PARITY)

    out = bytearray()
    repaired = failed = corrected = 0

    for row in blocks:
        raw = row.tobytes()
        try:
            data, _, errata = codec.decode(raw)
        except reedsolo.ReedSolomonError:
            failed += 1
            out.extend(raw[:PAYLOAD_DATA])
            continue
        if errata:
            repaired += 1
            corrected += len(errata)
        out.extend(bytes(data))

    return DecodeReport(bytes(out[:length]), n_blocks, repaired, failed, corrected)


# -------------------------------------------------------------- header path


def encode_header(header: bytes) -> bytes:
    """Protect the fixed-size header with a heavy Reed-Solomon code.

    The header is what tells the receiver how much payload to expect, so it
    gets far more parity than it would need on merit: losing it loses the
    frame, and it costs only a few OFDM symbols.
    """
    return bytes(_codec(HEADER_PARITY).encode(header))


def decode_header(coded: bytes, plain_len: int) -> bytes:
    """Recover the header, raising :class:`FecError` when it is unreadable."""
    want = plain_len + HEADER_PARITY
    if len(coded) < want:
        raise FecError(f"header is short: got {len(coded)}, need {want}")
    try:
        data, _, _ = _codec(HEADER_PARITY).decode(bytes(coded[:want]))
    except reedsolo.ReedSolomonError as exc:
        raise FecError("header is beyond repair") from exc
    return bytes(data)
