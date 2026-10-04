"""Frame construction and parsing.

Wire format
-----------
Every transmission carries a fixed 12-byte header, heavily protected, followed
by the payload.  The header exists so the receiver can learn the payload
length before it has decoded the payload -- without it there is no way to know
where the frame ends, and trailing room noise would be decoded as data.

    header (plaintext, big endian)
        magic    u16   0xB07A
        version  u8
        flags    u8    bit 0: payload is Reed-Solomon coded
        length   u32   payload size in bytes
        crc32    u32   CRC-32 of the payload

The header is then Reed-Solomon coded to 54 bytes.  The payload follows,
either RS-coded and interleaved or raw, depending on the flag.

The CRC is deliberately kept *separate* from the error correction.  RS tells
you whether it could repair what it saw; the CRC tells you whether the result
is actually the message that was sent.  Those are different questions, and a
miscorrecting codeword answers the first one wrongly.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

from . import fec

__all__ = [
    "MAGIC",
    "VERSION",
    "HEADER_LEN",
    "HEADER_CODED_LEN",
    "FLAG_FEC",
    "Header",
    "FrameError",
    "build_frame",
    "header_bytes_on_air",
]

MAGIC = 0xB07A
"""Frame marker.  Reads as 'BORA' if you squint at the hex."""

VERSION = 1
"""Wire format version."""

_STRUCT = struct.Struct(">HBBII")

HEADER_LEN = _STRUCT.size
"""Plaintext header size: 12 bytes."""

HEADER_CODED_LEN = HEADER_LEN + fec.HEADER_PARITY
"""Header size on air: 54 bytes."""

FLAG_FEC = 0x01
"""Set when the payload carries Reed-Solomon parity."""


class FrameError(Exception):
    """Raised when a received frame is malformed beyond recovery."""


@dataclass(frozen=True)
class Header:
    """Decoded frame header."""

    version: int
    flags: int
    length: int
    crc: int

    @property
    def fec_enabled(self) -> bool:
        """True when the payload is Reed-Solomon coded."""
        return bool(self.flags & FLAG_FEC)

    def pack(self) -> bytes:
        """Serialise to the 12-byte plaintext form."""
        return _STRUCT.pack(MAGIC, self.version, self.flags, self.length, self.crc)

    @classmethod
    def unpack(cls, raw: bytes) -> Header:
        """Parse the plaintext form, validating the magic and version."""
        if len(raw) < HEADER_LEN:
            raise FrameError("header is truncated")
        magic, version, flags, length, crc = _STRUCT.unpack(raw[:HEADER_LEN])
        if magic != MAGIC:
            raise FrameError(f"bad magic 0x{magic:04X}; this is not a BORA frame")
        if version != VERSION:
            raise FrameError(f"unsupported wire version {version}")
        return cls(version=version, flags=flags, length=length, crc=crc)

    def payload_bytes_on_air(self) -> int:
        """How many payload bytes the receiver should expect to demodulate."""
        if self.fec_enabled:
            return fec.coded_length(self.length)
        return self.length


def header_bytes_on_air() -> int:
    """Size of the coded header in bytes.  Constant, but named for clarity."""
    return HEADER_CODED_LEN


def build_frame(payload: bytes, *, use_fec: bool = True) -> bytes:
    """Assemble the complete on-air byte stream for ``payload``."""
    header = Header(
        version=VERSION,
        flags=FLAG_FEC if use_fec else 0,
        length=len(payload),
        crc=zlib.crc32(payload) & 0xFFFFFFFF,
    )
    body = fec.encode_blocks(payload) if use_fec else payload
    return fec.encode_header(header.pack()) + body


def parse_header(coded: bytes) -> Header:
    """Recover and validate the header from its coded form."""
    return Header.unpack(fec.decode_header(coded, HEADER_LEN))


def parse_payload(header: Header, body: bytes) -> tuple[bytes, fec.DecodeReport]:
    """Recover the payload described by ``header`` from the demodulated body."""
    if header.fec_enabled:
        report = fec.decode_blocks(body, header.length)
        return report.data, report

    data = body[: header.length]
    report = fec.DecodeReport(data, 0, 0, 0, 0)
    return data, report


def crc_matches(header: Header, payload: bytes) -> bool:
    """True when ``payload`` hashes to the CRC the sender recorded."""
    if len(payload) != header.length:
        return False
    return zlib.crc32(payload) & 0xFFFFFFFF == header.crc
