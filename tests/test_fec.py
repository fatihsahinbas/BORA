"""Reed-Solomon coding and the block interleaver.

The interesting tests here are the ones that damage the codewords on purpose:
a code is only worth having if you know exactly where it stops working.
"""

from __future__ import annotations

import numpy as np
import pytest

from bora import fec


def test_roundtrip_without_damage(payload: bytes) -> None:
    report = fec.decode_blocks(fec.encode_blocks(payload), len(payload))
    assert report.data == payload
    assert report.ok
    assert report.repaired == 0


def test_empty_payload_encodes_to_nothing() -> None:
    assert fec.encode_blocks(b"") == b""
    assert fec.decode_blocks(b"", 0).data == b""


@pytest.mark.parametrize("size", [1, 222, 223, 224, 500, 1_000])
def test_block_count_covers_the_payload(size: int) -> None:
    assert fec.coded_length(size) == fec.block_count(size) * fec.BLOCK_LEN
    assert fec.block_count(size) * fec.PAYLOAD_DATA >= size


def test_interleave_is_reversible(rng: np.random.Generator) -> None:
    blocks = rng.integers(0, 256, (7, fec.BLOCK_LEN), dtype=np.uint8)
    restored = fec.deinterleave(fec.interleave(blocks), 7)
    assert np.array_equal(restored, blocks)


def test_interleave_spreads_adjacent_bytes_across_codewords() -> None:
    """Consecutive on-air bytes must belong to different codewords.

    This is the whole point of the interleaver; if it ever stops holding, a
    burst error goes back to destroying one codeword instead of scratching
    many, and the FEC stops helping where it matters most.
    """
    n_blocks = 5
    blocks = np.arange(n_blocks * fec.BLOCK_LEN, dtype=np.uint8).reshape(
        n_blocks, fec.BLOCK_LEN
    )
    stream = np.frombuffer(fec.interleave(blocks), dtype=np.uint8)

    # The first n_blocks bytes on air are byte 0 of every codeword.
    assert np.array_equal(stream[:n_blocks], blocks[:, 0])


def test_corrects_damage_up_to_the_design_limit(payload: bytes, rng) -> None:
    """Sixteen byte errors per codeword is the documented limit."""
    stream = bytearray(fec.encode_blocks(payload))
    n_blocks = fec.block_count(len(payload))

    # Interleaved, so a run of n_blocks consecutive bytes is one byte in each
    # codeword.  Sixteen such runs damages every codeword exactly 16 times.
    for run in range(16):
        start = run * n_blocks
        for i in range(start, start + n_blocks):
            stream[i] ^= 0xFF

    report = fec.decode_blocks(bytes(stream), len(payload))
    assert report.data == payload
    assert report.failed == 0
    assert report.repaired == n_blocks


def test_reports_failure_past_the_design_limit(payload: bytes) -> None:
    stream = bytearray(fec.encode_blocks(payload))
    n_blocks = fec.block_count(len(payload))

    for run in range(30):  # comfortably past 16
        start = run * n_blocks
        for i in range(start, start + n_blocks):
            stream[i] ^= 0xFF

    report = fec.decode_blocks(bytes(stream), len(payload))
    assert not report.ok
    assert report.failed > 0
    # It still returns bytes rather than raising: a damaged image beats none.
    assert len(report.data) == len(payload)


def test_a_burst_shorter_than_the_interleaver_depth_is_free(payload: bytes) -> None:
    """A contiguous wipe-out costs each codeword at most a byte or two."""
    stream = bytearray(fec.encode_blocks(payload))
    n_blocks = fec.block_count(len(payload))

    burst = 8 * n_blocks  # eight bytes per codeword once deinterleaved
    for i in range(500, 500 + burst):
        stream[i] ^= 0xA5

    report = fec.decode_blocks(bytes(stream), len(payload))
    assert report.data == payload
    assert report.failed == 0


def test_header_survives_heavy_damage() -> None:
    header = bytes(range(12))
    coded = bytearray(fec.encode_header(header))
    assert len(coded) == 12 + fec.HEADER_PARITY

    for i in range(0, 21):  # right at the correction limit
        coded[i * 2] ^= 0xFF

    assert fec.decode_header(bytes(coded), 12) == header


def test_header_decode_raises_when_unrecoverable() -> None:
    coded = bytearray(fec.encode_header(bytes(range(12))))
    for i in range(len(coded)):
        coded[i] ^= 0xFF

    with pytest.raises(fec.FecError):
        fec.decode_header(bytes(coded), 12)


def test_deinterleave_rejects_a_short_stream() -> None:
    with pytest.raises(fec.FecError):
        fec.deinterleave(b"\x00" * 10, n_blocks=3)
