"""Render a short video and a cover image of one transmission through a room.

    python examples/video_demo.py --out-dir docs/

A 96x72 bitmap goes through the full stack -- Reed-Solomon, interleaver,
OFDM, chirps -- then through :func:`bora.channel.room` (echoes, transducer
roll-off, noise, interference bursts, 60 ppm clock drift) and back.

The video shows two constellations side by side.  The receiver itself only
ever uses the corrected one; the uncorrected one is the same signal decoded
without the closing-chirp drift measurement, drawn so the reason for that
measurement is visible.

Needs ``pillow`` and, for the video, ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bora import DEFAULT, channel, framing, link, modem

CFG = DEFAULT
IMG_W, IMG_H = 96, 72
FW, FH, FPS = 1080, 1350, 30
HOLD = 5.0

BG = (14, 15, 20)
FG = (232, 230, 240)
MUTED = (138, 135, 152)
DIM = (90, 88, 105)
EDGE = (40, 40, 52)
ACCENT = (255, 214, 120)
BAD = (255, 92, 78)
GOOD = (120, 220, 170)


# ------------------------------------------------------------------ fonts


def _font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.ImageFont:
    if mono:
        names = ["DejaVuSansMono.ttf", "consola.ttf", "Menlo.ttc"]
    elif bold:
        names = ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"]
    else:
        names = ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


# ------------------------------------------------------------- the payload


def make_picture() -> Image.Image:
    """An original test picture: mountains, a sun, the project name."""
    scale = 4
    w, h = IMG_W * scale, IMG_H * scale
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        draw.line(
            [(0, y), (w, y)],
            fill=(int(25 + 230 * t**1.2), int(40 + 120 * t**1.6), int(110 - 60 * t)),
        )
    draw.ellipse([230, 105, 330, 205], fill=(255, 214, 120))
    draw.polygon(
        [
            (0, 288),
            (0, 200),
            (75, 150),
            (140, 195),
            (205, 135),
            (290, 200),
            (340, 170),
            (384, 195),
            (384, 288),
        ],
        fill=(70, 40, 90),
    )
    draw.polygon(
        [(0, 288), (0, 235), (95, 200), (180, 240), (260, 210), (384, 245), (384, 288)],
        fill=(30, 20, 50),
    )
    draw.text((26, 18), "BORA", font=_font(78, bold=True), fill=(255, 255, 255))
    return img.resize((IMG_W, IMG_H), Image.LANCZOS)


def as_image(data: bytes) -> Image.Image:
    need = IMG_W * IMG_H * 3
    return Image.frombytes("RGB", (IMG_W, IMG_H), data.ljust(need, b"\x00")[:need])


# --------------------------------------------------------- one transmission


@dataclass
class Run:
    picture: Image.Image
    rx: np.ndarray
    frame: bytes
    n_syms: int
    on_air: bytes
    diff_raw: np.ndarray
    diff_fixed: np.ndarray
    good: link.Reception
    naive: link.Reception
    tx_len: int


def _diffs(rx: np.ndarray, n_syms: int) -> np.ndarray:
    sync = modem.find_preamble(CFG, rx)
    span = rx[sync.offset : sync.offset + (n_syms + 1) * CFG.sym_len]
    frames = span.reshape(n_syms + 1, CFG.sym_len)[:, CFG.cp_len :]
    cells = np.fft.rfft(frames, axis=1)[:, CFG.carriers]
    return cells[1:] * np.conj(cells[:-1])


def transmit_once(seed: int = 3) -> Run:
    picture = make_picture()
    payload = picture.tobytes()
    tx = link.transmit(payload, CFG)
    rx = channel.room(
        tx, CFG.fs, np.random.default_rng(seed), snr_db=20, ppm=60, bursts=2
    )

    good = link.receive(rx, CFG)
    naive = link.receive(rx, CFG, correct_drift=False)

    frame = framing.build_frame(payload, use_fec=True)
    n_syms = modem.symbols_for_bytes(CFG, len(frame))
    fixed = channel.resample(rx, 1.0 + good.drift_ppm * 1e-6)
    sync = modem.find_preamble(CFG, fixed)
    bits = modem.demodulate_bits(CFG, fixed, sync.offset, n_syms)
    on_air = modem.bits_to_bytes(bits)[: len(frame)]

    return Run(
        picture=picture,
        rx=rx,
        frame=frame,
        n_syms=n_syms,
        on_air=on_air,
        diff_raw=_diffs(rx, n_syms),
        diff_fixed=_diffs(fixed, n_syms),
        good=good,
        naive=naive,
        tx_len=len(tx),
    )


def evm(diff: np.ndarray) -> float:
    angle = np.angle(diff)
    err = angle - np.rint(angle / (np.pi / 2)) * (np.pi / 2)
    return float(np.degrees(np.sqrt(np.mean(err**2))))


# ------------------------------------------------------------ spectrogram

SPEC_FMIN, SPEC_FMAX = 500.0, 6500.0
SPEC_HOP = 256


def spectrogram(x: np.ndarray) -> np.ndarray:
    n = 1024
    win = np.hanning(n)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::SPEC_HOP] * win
    mag = np.abs(np.fft.rfft(frames, axis=1)).T
    freqs = np.fft.rfftfreq(n, 1 / CFG.fs)
    band = (freqs >= SPEC_FMIN) & (freqs <= SPEC_FMAX)
    db = 20 * np.log10(mag[band] + 1e-9)
    norm = np.clip((db - (db.max() - 50)) / 50, 0, 1)[::-1]
    stops = np.array(
        [[14, 15, 20], [60, 30, 90], [200, 90, 90], [255, 214, 120], [255, 250, 230]],
        float,
    )
    pos = norm * (len(stops) - 1)
    i = np.clip(pos.astype(int), 0, len(stops) - 2)
    t = (pos - i)[..., None]
    return (stops[i] * (1 - t) + stops[i + 1] * t).astype(np.uint8)


# --------------------------------------------------------------- drawing


def draw_constellation(
    img: Image.Image, box: tuple[int, int, int], points: np.ndarray
) -> None:
    x0, y0, size = box
    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x0 + size, y0 + size], outline=EDGE, width=2)
    c = size // 2
    draw.line([x0 + c, y0 + 6, x0 + c, y0 + size - 6], fill=EDGE)
    draw.line([x0 + 6, y0 + c, x0 + size - 6, y0 + c], fill=EDGE)
    if points.size == 0:
        return
    # DQPSK carries everything in the phase, so draw phase only
    p = points / np.maximum(np.abs(points), 1e-12) * (size * 0.38)
    for z in p:
        x, y = x0 + c + z.real, y0 + c - z.imag
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=ACCENT)


def bytes_grid(data: bytes, errors: np.ndarray, upto: int) -> Image.Image:
    """On-air bytes as grey pixels in arrival order; bytes that arrived wrong in red."""
    width = 192
    rows = -(-len(data) // width)
    vals = np.zeros(rows * width, dtype=np.uint8)
    arr = np.frombuffer(data, dtype=np.uint8)[:upto]
    vals[: len(arr)] = arr
    px = np.repeat((40 + vals * 0.55).astype(np.uint8)[:, None], 3, axis=1)
    bad = np.zeros(rows * width, dtype=bool)
    bad[: len(arr)] = errors[: len(arr)]
    px[bad] = BAD
    px[len(arr) :] = BG
    return Image.fromarray(px.reshape(rows, width, 3))


def paste_fit(dst: Image.Image, src: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, w, h = box
    k = min(w / src.width, h / src.height)
    sw, sh = int(src.width * k), int(src.height * k)
    dst.paste(
        src.resize((sw, sh), Image.NEAREST), (x0 + (w - sw) // 2, y0 + (h - sh) // 2)
    )


# ----------------------------------------------------------------- video


class Video:
    def __init__(self, run: Run) -> None:
        self.run = run
        self.total = len(run.rx) / CFG.fs
        k = len(run.rx) / run.tx_len
        lead = CFG.lead_silence
        self.t_body = (lead + CFG.chirp_len + CFG.guard_len) * k / CFG.fs
        self.t_close = (
            (lead + modem.postamble_position(CFG, run.n_syms) - CFG.guard_len)
            * k
            / CFG.fs
        )
        self.t_end = self.t_close + (CFG.guard_len + CFG.chirp_len) * k / CFG.fs
        failed = run.naive.reason.split()[0] if run.naive.reason else "?"
        self.naive_note = f"{failed} blok bozuk"
        self.header_syms = modem.symbols_for_bytes(CFG, framing.HEADER_CODED_LEN)

        self.errors = np.frombuffer(run.on_air, np.uint8) != np.frombuffer(
            run.frame, np.uint8
        )
        self.spec = Image.fromarray(spectrogram(run.rx))
        self.cols_per_sec = CFG.fs / SPEC_HOP
        self.f = {
            "title": _font(46, bold=True),
            "sub": _font(26),
            "label": _font(20, mono=True),
            "caption": _font(27),
            "stat": _font(27, mono=True),
            "big": _font(34, bold=True),
        }

    def symbol_at(self, t: float) -> int:
        if t <= self.t_body:
            return 0
        frac = (t - self.t_body) / (self.t_close - self.t_body)
        return int(np.clip(frac, 0, 1) * self.run.n_syms)

    def caption(self, t: float, sym: int) -> str:
        if t < self.t_body:
            return "Açılış chirp'i: alıcı zamanlamayı buluyor"
        if sym <= self.header_syms:
            return "Başlık: alıcı veri boyunu öğreniyor"
        if t < self.t_close:
            return "Veri: 54 taşıyıcı, sembol başına 108 bit"
        if t < self.t_end + 0.3:
            return "Kapanış chirp'i: iki saat arasındaki fark ölçülüyor"
        return "Reed-Solomon onardı, CRC doğruladı"

    def frame(self, i: int) -> Image.Image:
        run, f = self.run, self.f
        t = i / FPS
        done = t > self.t_end + 0.3
        sym = self.symbol_at(t)

        img = Image.new("RGB", (FW, FH), BG)
        d = ImageDraw.Draw(img)
        d.text((60, 52), "BORA", font=f["title"], fill=FG)
        d.text((222, 70), "Byte Over Resonant Air", font=f["sub"], fill=MUTED)

        # picture / on-air bytes
        box = (60, 180, 600, 450)
        d.rectangle([58, 178, 661, 631], outline=EDGE, width=2)
        if done:
            a = min((t - self.t_end - 0.3) / 0.6, 1.0)
            grid = bytes_grid(run.on_air, self.errors, len(run.on_air))
            canvas = Image.new("RGB", (600, 450), BG)
            paste_fit(canvas, grid, (0, 0, 600, 450))
            final = Image.new("RGB", (600, 450), BG)
            paste_fit(final, as_image(run.good.payload), (0, 0, 600, 450))
            img.paste(Image.blend(canvas, final, a), (60, 180))
            d.text((60, 148), "ALINAN GÖRÜNTÜ", font=f["label"], fill=FG)
        else:
            upto = min(int(sym * CFG.bits_per_symbol / 8), len(run.on_air))
            if upto:
                paste_fit(img, bytes_grid(run.on_air, self.errors, upto), box)
            d.text(
                (60, 148),
                "HAVADAKİ BAYTLAR  ·  kırmızı: hatalı gelen",
                font=f["label"],
                fill=MUTED,
            )

        # constellations
        lo = max(0, sym - 6)
        for k, (name, diff, y0) in enumerate(
            [
                ("SAAT DÜZELTMESİ YOK", run.diff_raw, 180),
                ("SAAT DÜZELTMELİ", run.diff_fixed, 412),
            ]
        ):
            d.text((690, y0), name, font=f["label"], fill=MUTED)
            pts = diff[lo:sym].ravel() if sym else np.empty(0, complex)
            draw_constellation(img, (690, y0 + 30, 190), pts)
            win = diff[max(0, sym - 40) : sym]
            if sym > 5:
                e = evm(win)
                colour = BAD if (k == 0 and e > 12) else FG
                d.text((900, y0 + 70), "EVM", font=f["label"], fill=MUTED)
                d.text((900, y0 + 100), f"{e:4.1f}°", font=f["big"], fill=colour)

        d.text((60, 660), self.caption(t, sym), font=f["caption"], fill=FG)

        # spectrogram
        sx, sy, sw, sh = 60, 712, 900, 270
        win_sec = 2.0
        scale = sw / (win_sec * self.cols_per_sec)
        right = int(min(t, self.total) * self.cols_per_sec)
        left = max(0, right - int(win_sec * self.cols_per_sec))
        strip = Image.new("RGB", (sw, sh), BG)
        if right > left:
            piece = self.spec.crop((left, 0, right, self.spec.height))
            piece = piece.resize((max(1, int(piece.width * scale)), sh), Image.BILINEAR)
            strip.paste(piece, (sw - piece.width, 0))
        img.paste(strip, (sx, sy))
        d.rectangle([sx - 2, sy - 2, sx + sw + 1, sy + sh + 1], outline=EDGE, width=2)
        d.line([sx + sw - 1, sy, sx + sw - 1, sy + sh], fill=FG, width=2)
        for hz, name in [(6000, "6 kHz"), (1000, "1 kHz")]:
            y = sy + int((SPEC_FMAX - hz) / (SPEC_FMAX - SPEC_FMIN) * sh)
            for x in range(sx, sx + sw, 14):
                d.line([x, y, x + 6, y], fill=DIM)
            d.text((sx + sw + 10, y), name, font=f["label"], fill=MUTED, anchor="lm")

        # stats
        sent = min(len(run.frame), int(sym * CFG.bits_per_symbol / 8))
        mm = lambda s: f"{int(s) // 60:02d}:{int(s) % 60:02d}"  # noqa: E731
        rows = [
            ("süre", f"{mm(min(t, self.total))} / {mm(self.total)}", FG),
            ("gönderilen", f"{sent / 1000:4.1f} / {len(run.frame) / 1000:.1f} KB", FG),
        ]
        if t >= self.t_end:
            rows.append(("saat farkı", f"{run.good.drift_ppm:+.1f} ppm, düzeltildi", FG))
        else:
            rows.append(("saat farkı", "kapanış chirp'i bekleniyor", DIM))
        if done:
            fixed = run.good.fec.corrected_bytes
            rows.append(("hatalı bayt", f"{fixed} → 0", FG))
            rows.append(("CRC", "OK" if run.good.ok else "HATA", GOOD))
        for k, (key, val, colour) in enumerate(rows):
            y = 1030 + k * 46
            d.text((60, y), key, font=f["stat"], fill=MUTED)
            d.text((300, y), val, font=f["stat"], fill=colour)

        if done:
            a = min((t - self.t_end - 1.2) / 0.6, 1.0)
            if a > 0:
                thumb = as_image(run.naive.payload).resize((192, 144), Image.NEAREST)
                ov = Image.new("RGB", (192, 144), BG)
                img.paste(Image.blend(ov, thumb, a), (828, 1080))
                d.text((828, 1050), "düzeltmesiz:", font=f["label"], fill=MUTED)
                d.text((828, 1232), "CRC: HATA", font=f["label"], fill=BAD)
                d.text((828, 1258), self.naive_note, font=f["label"], fill=BAD)
        return img

    def n_frames(self) -> int:
        return int((self.total + HOLD) * FPS)


def write_wav(path: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(CFG.fs)
        w.writeframes(pcm.tobytes())


def render_video(run: Run, out: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")
    video = Video(run)
    audio = out.with_suffix(".wav")
    write_wav(audio, np.concatenate([run.rx, np.zeros(int(HOLD * CFG.fs))]))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{FW}x{FH}", "-r", str(FPS),
        "-i", "-", "-i", str(audio),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
        str(out),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(video.n_frames()):
        proc.stdin.write(video.frame(i).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    audio.unlink()


# ----------------------------------------------------------------- cover


def render_cover(run: Run, out: Path) -> None:
    img = Image.new("RGB", (FW, FH), BG)
    d = ImageDraw.Draw(img)
    title, label = _font(50, bold=True), _font(22, mono=True)

    d.text((60, 60), "Bir dosyayı sese çevirdim,", font=title, fill=FG)
    d.text((60, 125), "sonra sesten geri okudum.", font=title, fill=ACCENT)

    w, h, y0 = 465, 349, 250
    d.text((60, y0), "GÖNDERİLEN", font=label, fill=MUTED)
    d.text((FW - 60 - w, y0), "ALINAN", font=label, fill=MUTED)
    img.paste(run.picture.resize((w, h), Image.NEAREST), (60, y0 + 36))
    img.paste(
        as_image(run.good.payload).resize((w, h), Image.NEAREST), (FW - 60 - w, y0 + 36)
    )

    # opening of the frame: chirp, then the carriers switch on
    sy = 700
    d.text(
        (60, sy - 40),
        "İLK 0,3 SANİYE  ·  chirp, sonra 54 taşıyıcı",
        font=label,
        fill=MUTED,
    )
    spec = spectrogram(run.rx[: int(0.3 * CFG.fs)])
    img.paste(Image.fromarray(spec).resize((960, 170), Image.BILINEAR), (60, sy))
    d.rectangle([58, sy - 2, 1021, sy + 171], outline=EDGE, width=2)

    cy, size = 940, 250
    for k, (name, diff, colour) in enumerate(
        [
            ("saat düzeltmesi yok", run.diff_raw, BAD),
            ("düzeltilmiş", run.diff_fixed, GOOD),
        ]
    ):
        x = 60 + k * 500
        tail = diff[-8:]
        d.text((x, cy - 40), name.upper(), font=label, fill=MUTED)
        draw_constellation(img, (x, cy, size), tail.ravel())
        d.text((x + size + 18, cy + 60), "EVM", font=label, fill=MUTED)
        big = _font(36, bold=True)
        d.text((x + size + 18, cy + 92), f"{evm(diff[-40:]):.1f}°", font=big, fill=colour)

    fec = run.good.fec
    foot = (
        f"8100 bps  ·  {run.good.drift_ppm:+.0f} ppm düzeltildi  ·  "
        f"{fec.corrected_bytes} bayt onarıldı"
    )
    d.text((60, FH - 70), foot, font=_font(25), fill=MUTED)
    img.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run = transmit_once()
    print(run.good.summary(), f"| drift {run.good.drift_ppm:+.1f} ppm")
    print("without drift correction:", run.naive.summary())
    render_cover(run, args.out_dir / "bora_cover.png")
    if not args.no_video:
        render_video(run, args.out_dir / "bora_demo.mp4")


if __name__ == "__main__":
    main()
