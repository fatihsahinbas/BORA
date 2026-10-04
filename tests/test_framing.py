"""Wire format: header packing, framing, and the checks that guard them."""

from __future__ import annotations

import zlib

import pytest

from bora import framing
from bora.fec import FecError
from bora.framing import FLAG_FEC, Header


def test_header_roundtrip() -> None:
    header = Header(version=framing.VERSION, flags=FLAG_FEC, length=1234, crc=0xDEADBEEF)
    assert Header.unpack(header.pack()) == header


def test_header_is_twelve_bytes() -> None:
    assert framing.HEADER_LEN == 12
    assert framing.HEADER_CODED_LEN == 54


def test_bad_magic_is_rejected() -> None:
    raw = bytearray(
        Header(version=1, flags=0, length=1, crc=2).pack()
    )
    raw[0] ^= 0xFF
    with pytest.raises(framing.FrameError):
        Header.unpack(bytes(raw))


def test_unknown_version_is_rejected() -> None:
    raw = bytearray(Header(version=99, flags=0, length=1, crc=2).pack())
    with pytest.raises(framing.FrameError):
        Header.unpack(bytes(raw))


def test_truncated_header_is_rejected() -> None:
    with pytest.raises(framing.FrameError):
        Header.unpack(b"\xb0\x7a")


def test_build_and_parse_frame(payload: bytes) -> None:
    frame = framing.build_frame(payload)
    header = framing.parse_header(frame[: framing.HEADER_CODED_LEN])

    assert header.length == len(payload)
    assert header.fec_enabled
    assert header.crc == zlib.crc32(payload) & 0xFFFFFFFF

    body = frame[framing.HEADER_CODED_LEN :]
    recovered, report = framing.parse_payload(header, body)

    assert recovered == payload
    assert report.ok
    assert framing.crc_matches(header, recovered)


def test_frame_without_fec_carries_the_payload_verbatim(payload: bytes) -> None:
    frame = framing.build_frame(payload, use_fec=False)
    header = framing.parse_header(frame[: framing.HEADER_CODED_LEN])

    assert not header.fec_enabled
    assert header.payload_bytes_on_air() == len(payload)
    assert frame[framing.HEADER_CODED_LEN :] == payload


def test_crc_catches_a_single_flipped_bit(payload: bytes) -> None:
    frame = framing.build_frame(payload)
    header = framing.parse_header(frame[: framing.HEADER_CODED_LEN])

    damaged = bytearray(payload)
    damaged[0] ^= 0x01
    assert not framing.crc_matches(header, bytes(damaged))


def test_crc_catches_a_truncated_payload(payload: bytes) -> None:
    frame = framing.build_frame(payload)
    header = framing.parse_header(frame[: framing.HEADER_CODED_LEN])
    assert not framing.crc_matches(header, payload[:-1])


def test_empty_payload_is_a_valid_frame() -> None:
    frame = framing.build_frame(b"")
    header = framing.parse_header(frame[: framing.HEADER_CODED_LEN])

    assert header.length == 0
    recovered, _ = framing.parse_payload(header, b"")
    assert recovered == b""
    assert framing.crc_matches(header, recovered)


def test_noise_does_not_parse_as_a_header(rng) -> None:
    noise = bytes(rng.integers(0, 256, framing.HEADER_CODED_LEN, dtype="uint8"))
    with pytest.raises((FecError, framing.FrameError)):
        framing.parse_header(noise)
