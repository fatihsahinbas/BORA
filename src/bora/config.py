"""Physical layer parameters.

Everything downstream derives from :class:`Config`.  Changing a field here
changes the waveform; nothing else in the codebase hard-codes a number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Config", "DEFAULT"]


@dataclass(frozen=True)
class Config:
    """OFDM parameters for the acoustic channel.

    The defaults target a laptop speaker talking to a laptop microphone
    across a quiet room.  The 1-6 kHz band avoids the worst of the
    low-frequency room modes while staying below the roll-off of cheap
    transducers.
    """

    fs: int = 48_000
    """Sample rate in hertz."""

    n_fft: int = 512
    """FFT size.  Carrier spacing is ``fs / n_fft``."""

    cp_len: int = 128
    """Cyclic prefix length in samples.  Sets the echo tolerance."""

    f_lo: float = 1_000.0
    """Lowest carrier frequency in hertz."""

    f_hi: float = 6_000.0
    """Highest carrier frequency in hertz."""

    chirp_len: int = 2_048
    """Preamble sweep length in samples."""

    guard_len: int = 256
    """Silence between preamble and first OFDM symbol."""

    lead_silence: int = 2_400
    """Silence padded at both ends of the transmission."""

    amplitude: float = 0.8
    """Peak amplitude of the emitted waveform, 0..1."""

    # ---------------------------------------------------------- derived

    @property
    def bin_lo(self) -> int:
        """Index of the lowest active FFT bin."""
        return int(np.ceil(self.f_lo * self.n_fft / self.fs))

    @property
    def bin_hi(self) -> int:
        """Index of the highest active FFT bin."""
        return int(np.floor(self.f_hi * self.n_fft / self.fs))

    @property
    def carriers(self) -> np.ndarray:
        """Active FFT bin indices."""
        return np.arange(self.bin_lo, self.bin_hi + 1)

    @property
    def n_carriers(self) -> int:
        """Number of active carriers."""
        return self.bin_hi - self.bin_lo + 1

    @property
    def sym_len(self) -> int:
        """OFDM symbol length in samples, cyclic prefix included."""
        return self.n_fft + self.cp_len

    @property
    def bits_per_symbol(self) -> int:
        """Payload bits carried by one OFDM symbol.  DQPSK: 2 per carrier."""
        return self.n_carriers * 2

    @property
    def carrier_spacing(self) -> float:
        """Spacing between carriers in hertz."""
        return self.fs / self.n_fft

    @property
    def symbol_rate(self) -> float:
        """OFDM symbols per second."""
        return self.fs / self.sym_len

    @property
    def raw_bitrate(self) -> float:
        """Channel bitrate before forward error correction, in bits/s."""
        return self.bits_per_symbol * self.symbol_rate

    @property
    def cp_millis(self) -> float:
        """Echo tolerance in milliseconds."""
        return self.cp_len / self.fs * 1_000.0

    def validate(self) -> None:
        """Raise :class:`ValueError` if the parameter set is unusable."""
        if self.f_hi <= self.f_lo:
            raise ValueError("f_hi must exceed f_lo")
        if self.f_hi >= self.fs / 2:
            raise ValueError("f_hi must stay below the Nyquist frequency")
        if self.n_carriers < 1:
            raise ValueError("the configured band contains no carriers")
        if self.cp_len >= self.n_fft:
            raise ValueError("cyclic prefix must be shorter than the FFT")
        if not 0.0 < self.amplitude <= 1.0:
            raise ValueError("amplitude must fall in (0, 1]")

    def describe(self) -> str:
        """Human readable summary of the derived parameters."""
        return "\n".join(
            [
                f"sample rate      {self.fs} Hz",
                f"band             {self.f_lo:.0f}-{self.f_hi:.0f} Hz",
                f"carriers         {self.n_carriers} (bins {self.bin_lo}-{self.bin_hi})",
                f"carrier spacing  {self.carrier_spacing:.2f} Hz",
                f"symbol           {self.sym_len} samples "
                f"({self.symbol_rate:.2f} sym/s)",
                f"cyclic prefix    {self.cp_len} samples ({self.cp_millis:.2f} ms)",
                f"bits per symbol  {self.bits_per_symbol}",
                f"raw bitrate      {self.raw_bitrate:.0f} bps",
            ]
        )


DEFAULT = Config()
