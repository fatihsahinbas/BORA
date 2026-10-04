"""Send a picture through the air and watch it degrade.

    python examples/image_demo.py --out docs/image_demo.png

Raw pixels are used rather than a compressed format on purpose.  A damaged
JPEG usually refuses to open at all, which tells you nothing; a damaged
bitmap shows you exactly where the channel bit you.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from bora import Config, channel, link

WIDTH, HEIGHT = 96, 72


def make_test_image() -> Image.Image:
    """A synthetic image with enough structure to make damage obvious."""
    img = Image.new("RGB", (WIDTH, HEIGHT), (14, 26, 48))
    draw = ImageDraw.Draw(img)

    for x in range(0, WIDTH, 8):
        draw.line([(x, 0), (x, HEIGHT)], fill=(24, 44, 78))
    for radius, colour in ((30, (222, 86, 62)), (20, (86, 198, 178)),
                           (11, (240, 200, 74))):
        box = [WIDTH // 2 - radius, HEIGHT // 2 - radius,
               WIDTH // 2 + radius, HEIGHT // 2 + radius]
        draw.ellipse(box, outline=colour, width=3)
    draw.text((5, 4), "BORA", fill=(255, 255, 255))
    return img


def to_bytes(img: Image.Image) -> bytes:
    return img.tobytes()


def from_bytes(data: bytes) -> Image.Image:
    need = WIDTH * HEIGHT * 3
    data = data.ljust(need, b"\x00")[:need]
    return Image.frombytes("RGB", (WIDTH, HEIGHT), data)


def receive_at(payload: bytes, cfg: Config, snr_db: float, *, use_fec: bool,
               seed: int = 0) -> tuple[Image.Image, str]:
    wave = link.transmit(payload, cfg, use_fec=use_fec)
    rx = channel.awgn(wave, snr_db, np.random.default_rng(seed))
    result = link.receive(rx, cfg)

    got = result.payload.ljust(len(payload), b"\x00")
    wrong = sum(a != b for a, b in zip(payload, got, strict=True))
    label = f"{snr_db:.0f} dB  {'FEC' if use_fec else 'raw'}  {wrong} bad bytes"
    return from_bytes(result.payload), label


def build_sheet(panels: list[tuple[Image.Image, str]], scale: int = 3) -> Image.Image:
    from PIL import ImageDraw as D

    pad, caption_h = 10, 16
    cell_w, cell_h = WIDTH * scale, HEIGHT * scale + caption_h
    cols = len(panels)

    sheet = Image.new("RGB", (cols * cell_w + (cols + 1) * pad,
                              cell_h + 2 * pad), (255, 255, 255))
    draw = D.Draw(sheet)

    for i, (img, label) in enumerate(panels):
        x = pad + i * (cell_w + pad)
        sheet.paste(img.resize((cell_w, HEIGHT * scale), Image.NEAREST), (x, pad))
        draw.text((x, pad + HEIGHT * scale + 3), label, fill=(30, 30, 30))

    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/image_demo.png")
    parser.add_argument("--wav", default="docs/image.wav")
    args = parser.parse_args()

    cfg = Config()
    original = make_test_image()
    payload = to_bytes(original)

    print(f"raw bitmap    {len(payload)} bytes ({WIDTH}x{HEIGHT} RGB)")
    print(f"airtime       {link.airtime(len(payload), cfg):.1f} s")
    print(f"goodput       {link.goodput(len(payload), cfg):.0f} bps")
    print()

    panels = [(original, "original")]
    for snr, use_fec in ((9, False), (6, False), (6, True), (4, True), (3, True)):
        img, label = receive_at(payload, cfg, snr, use_fec=use_fec)
        panels.append((img, label))
        print(f"  {label}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    build_sheet(panels).save(args.out)
    print(f"\nwrote {args.out}")

    if args.wav:
        from bora import audio

        audio.write_wav(args.wav, link.transmit(payload, cfg), cfg)
        print(f"wrote {args.wav}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
