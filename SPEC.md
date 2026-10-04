# BORA wire specification

Version 1.

BORA carries an arbitrary byte string from a loudspeaker to a microphone.  It
assumes nothing about the channel except that it passes roughly 1-6 kHz and
that both ends agree on a nominal sample rate.  It does not assume the two
clocks actually match, and it does not assume the room is quiet.

All multi-byte integers are big endian.

---

## 1. Physical layer

| Parameter | Default | Notes |
|---|---|---|
| Sample rate | 48000 Hz | Both ends nominal; real rates may differ |
| FFT size | 512 | Carrier spacing 93.75 Hz |
| Active band | 1000-6000 Hz | FFT bins 11 to 64 inclusive |
| Active carriers | 54 | |
| Cyclic prefix | 128 samples | 2.67 ms of echo tolerance |
| Symbol length | 640 samples | 75 symbols per second |
| Modulation | DQPSK | 2 bits per carrier, Gray coded |
| Raw bitrate | 8100 bps | Before framing and parity |

### 1.1 Carrier mapping

Carrier `k` occupies FFT bin `k`, for `bin_lo <= k <= bin_hi`, where

    bin_lo = ceil(f_lo * n_fft / fs)
    bin_hi = floor(f_hi * n_fft / fs)

All other bins, including DC and Nyquist, are zero.  The time-domain symbol is
the real inverse FFT of that spectrum, so the waveform is real by
construction.

### 1.2 Differential encoding

Bits are taken two at a time, most significant first, and mapped to a phase
step through a Gray code:

| Dibit | Quadrant | Phase step |
|---|---|---|
| 00 | 0 | 0 deg |
| 01 | 1 | +90 deg |
| 11 | 2 | 180 deg |
| 10 | 3 | +270 deg |

The step is applied relative to the same carrier in the *previous* symbol.
The body therefore opens with a reference symbol in which every active carrier
carries `1 + 0j`.

The Gray code is chosen so that adjacent quadrants differ in exactly one bit.
The dominant error mode is a slip into a neighbouring quadrant, and this
mapping makes such a slip cost one bit rather than two.

Differential encoding means the receiver never needs to know the absolute
phase the room imposed.  It costs about 3 dB against coherent detection and
removes the need for a channel estimator, pilot carriers and an equaliser.

### 1.3 Cyclic prefix

Each symbol is transmitted as its final `cp_len` samples followed by the full
`n_fft` samples.  An echo arriving within the prefix window overlaps a copy of
the same symbol rather than the previous one, which turns multipath from
intersymbol interference into a per-carrier phase rotation -- and differential
detection then removes that rotation for free.

Echoes longer than the prefix are *not* handled.  They raise the noise floor
and are left to the error correction.

---

## 2. Frame structure

    [ lead silence ]
    [ opening chirp      2048 samples, 1000 -> 6000 Hz ]
    [ guard              256 samples of silence        ]
    [ reference symbol   640 samples                   ]
    [ data symbols       640 samples each              ]
    [ guard              256 samples of silence        ]
    [ closing chirp      2048 samples, 6000 -> 1000 Hz ]
    [ trail silence ]

### 2.1 Chirps

Both markers are linear frequency sweeps windowed by a Hann function.  A chirp
is used rather than a tone burst because its autocorrelation is a single sharp
spike: every frequency component aligns at exactly one lag, giving
sample-accurate timing from one correlation with no search.

The closing chirp sweeps downward, which makes it nearly orthogonal to the
opening one -- cross-correlation stays below 35% of either autocorrelation
peak -- so neither marker can be mistaken for the other.

### 2.2 Byte stream

The symbols carry one continuous bit stream, most significant bit first,
consisting of:

    [ coded header   54 bytes ]
    [ coded payload  variable ]

The stream is zero-padded to a whole number of symbols.  The header does
**not** need to end on a symbol boundary; the receiver decodes enough symbols
to cover 54 bytes and takes the first 54.

---

## 3. Header

Twelve bytes, plaintext:

| Offset | Size | Field | Value |
|---|---|---|---|
| 0 | 2 | magic | `0xB07A` |
| 2 | 1 | version | `1` |
| 3 | 1 | flags | bit 0: payload is FEC coded |
| 4 | 4 | length | payload size in bytes |
| 8 | 4 | crc32 | CRC-32 of the payload |

The header is then Reed-Solomon coded with 42 parity bytes to 54 bytes total,
correcting up to 21 byte errors.  That is far more protection than the header
needs on merit, but losing the header loses the whole frame, and the extra
parity costs only a few symbols.

A receiver MUST reject a frame whose magic or version does not match.

---

## 4. Forward error correction

### 4.1 Payload coding

Reed-Solomon over GF(256), codeword length 255, 32 parity bytes.  Each
codeword carries 223 data bytes and corrects up to 16 byte errors.

The payload is split into `ceil(length / 223)` chunks, the last zero-padded.
The receiver trims the result using the header's length field.

### 4.2 Interleaving

Codewords are stacked as the rows of an `n_blocks x 255` matrix and
transmitted **column by column**: byte 0 of every codeword, then byte 1 of
every codeword, and so on.

This is what makes the code useful on this channel.  Acoustic errors arrive in
clusters -- a notch in the room response kills the same carriers in every
symbol, a chair scrape wipes out several consecutive symbols.  Without
interleaving a burst destroys one codeword outright; with it, the same burst
costs each codeword a byte or two, well inside what the code repairs.

A burst of `n` consecutive on-air bytes costs each codeword at most
`ceil(n / n_blocks)` byte errors.

### 4.3 Integrity

The CRC-32 in the header is checked against the decoded payload and is
deliberately independent of the error correction.  Reed-Solomon reports
whether it could repair what it saw; the CRC reports whether the result is
actually the message that was sent.  A miscorrecting codeword answers the
first question wrongly, and only the second catches it.

A receiver MUST NOT report success on a CRC mismatch, even when every codeword
decoded cleanly.

---

## 5. Reception

### 5.1 Three passes

1. **Find the opening chirp** by matched filter.  The first OFDM symbol begins
   `chirp_len + guard_len` samples later.

2. **Decode the header, measure the clock, correct it.**  The header is only a
   few symbols long, so it survives even when the rest of the frame does not:
   whatever is wrong with the sample clock has had almost no time to
   accumulate.  The header gives the payload length, which gives the exact
   position the closing chirp should occupy:

       nominal = chirp_len + guard_len + (n_symbols + 1) * sym_len + guard_len

   Correlating for the closing chirp near that position gives the measured
   distance.  The ratio is the sample-rate error:

       ppm = (nominal / measured - 1) * 1e6

   Resampling by `nominal / measured` removes it.

3. **Decode the frame** against the corrected signal, reading exactly the
   symbols the header says the frame occupies and no more.

### 5.2 Why pass 2 exists

Two laptops nominally sampling at 48 kHz differ by tens to hundreds of parts
per million.  At 100 ppm, a ten-second transmission ends a millisecond out of
step; the FFT window slides off the symbol boundary and the constellation
smears until the decisions fail.  Measured here: the same frame that decodes
cleanly through a file fails through a simulated room until this correction is
applied, after which it tolerates at least +/- 400 ppm.

A receiver SHOULD ignore a closing-chirp match whose correlation peak is less
than 6 times the median, and SHOULD skip correction below 1 ppm.

### 5.3 Rejecting non-frames

A receiver MUST NOT emit a payload for a signal that contains no frame.  The
magic, the version check and the CRC together make a false positive from room
noise vanishingly unlikely.

---

## 6. Throughput

For a payload of `L` bytes with FEC enabled:

    on_air_bytes = 54 + ceil(L / 223) * 255
    symbols      = ceil(on_air_bytes * 8 / 108) + 1
    samples      = 2*lead_silence + 2*chirp_len + 2*guard_len + symbols*640

Measured goodput at the default settings: about 7000 bps for payloads over a
few kilobytes, against a raw channel rate of 8100 bps.

---

## 7. Conformance

An implementation conforms to version 1 if it produces and accepts the
waveform described above at the default parameters.  Other parameter sets are
valid but are not interoperable; both ends must be configured alike.
