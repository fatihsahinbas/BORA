"""BORA - Byte Over Resonant Air.

An acoustic modem: it turns bytes into sound, sends them across a room, and
turns them back into the same bytes.  No radio, no network, no pairing --
a speaker at one end and a microphone at the other.

    >>> import bora
    >>> wave = bora.transmit(b"hello over the air")
    >>> bora.receive(wave).payload
    b'hello over the air'

The stack, bottom to top:

``bora.config``
    Physical layer parameters.  Everything derives from here.
``bora.modem``
    OFDM with differential QPSK, chirp preamble synchronisation.
``bora.fec``
    Reed-Solomon over GF(256) with a block interleaver.
``bora.framing``
    Wire format: magic, length, CRC.
``bora.link``
    The two functions above, wired together.
``bora.channel``
    Simulated impairments, so the hard parts are debuggable offline.
``bora.audio``
    WAV files, and live playback and capture when PortAudio is present.
"""

from __future__ import annotations

from .channel import awgn, clock_drift, multipath, room
from .config import DEFAULT, Config
from .fec import DecodeReport
from .framing import Header
from .link import Reception, airtime, goodput, receive, transmit
from .modem import SignalQuality, Sync

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Config",
    "DEFAULT",
    "transmit",
    "receive",
    "Reception",
    "airtime",
    "goodput",
    "Header",
    "DecodeReport",
    "Sync",
    "SignalQuality",
    "awgn",
    "multipath",
    "clock_drift",
    "room",
]
