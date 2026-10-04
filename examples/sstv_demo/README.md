# SSTV warm-up

BORA's starting point was SSTV, the way amateur radio sends pictures as
sound with modes such as Robot36. This folder is a small modem in that style:
a single FM tone sweeping 1500–2300 Hz with pixel brightness. It is here
because it makes the step up to OFDM easy to see.

| | FM (this folder) | BORA |
|---|---|---|
| Carriers | 1 | 54 |
| What a symbol carries | one pixel's brightness, as a frequency | 108 bits, as phase changes |
| A 160×120 colour image | ~28 s | ~66 s raw, but bit-exact |
| Errors | visible noise, nothing repairs it | Reed-Solomon repairs, CRC verifies |
| Clock drift | line-start fit (slant correction) | closing chirp measures it in ppm |

FM degrades gracefully, which is why SSTV still suits radio: a noisy picture
is still a picture. It cannot carry a file, though, because a file has no
tolerance for a wrong byte. That difference is the whole reason BORA exists.

```bash
pip install numpy scipy pillow
python sstv.py      # bora_tx.wav, src.png, dec.png, state.npz
python cover.py     # sstv_cover.png
python render.py    # sstv_demo.mp4, needs ffmpeg
```

The channel is simulated: 12 dB SNR plus a 4 ms echo. The colour fringing in
the decoded image is that echo. Fonts are loaded from the DejaVu paths of a
typical Linux install.
