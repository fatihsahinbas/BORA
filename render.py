import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import stft
from scipy.io import wavfile
from modem import SR, H, T_SYNC, T_PORCH, T_PIX, W

FW, FH, FPS = 1080, 1350, 30
BG = (14, 15, 20)
FG = (232, 230, 240)
MUTED = (138, 135, 152)
ACCENT = (255, 214, 120)
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
f = lambda p, s: ImageFont.truetype(p, s)

st = np.load("state.npz")
rx, starts, src, dec = st["rx"], st["starts"], st["src"], st["dec"]
line_len = int(T_SYNC * SR) + int(T_PORCH * SR) + 3 * (int(round(W * T_PIX * SR)) + int(T_PORCH * SR))

# --- spektrogram (1000-2600 Hz) ---
FMIN, FMAX = 1000, 2600
fr, tt, Z = stft(rx, SR, nperseg=1024, noverlap=1024 - 256)
band = (fr >= FMIN) & (fr <= FMAX)
S = 20 * np.log10(np.abs(Z[band]) + 1e-9)
S = np.clip((S - (S.max() - 48)) / 48, 0, 1)[::-1]  # yüksek frekans üstte


def cmap(v):
    # koyu -> mor -> amber
    stops = np.array([[14, 15, 20], [60, 30, 90], [200, 90, 90], [255, 214, 120], [255, 250, 230]], float)
    x = v * (len(stops) - 1)
    i = np.clip(x.astype(int), 0, len(stops) - 2)
    t = (x - i)[..., None]
    return (stops[i] * (1 - t) + stops[i + 1] * t).astype(np.uint8)


spec_rgb = cmap(S)                       # (freq_bins, cols)
cols_per_sec = SR / 256
SPEC_W, SPEC_H = 900, 230
WIN_SEC = 2.5
spec_full = Image.fromarray(spec_rgb)
spec_scaled = spec_full.resize((int(spec_rgb.shape[1] * SPEC_W / (WIN_SEC * cols_per_sec)), SPEC_H), Image.BILINEAR)
px_per_sec = spec_scaled.width / (spec_rgb.shape[1] / cols_per_sec)


def fy(freq):
    return int((FMAX - freq) / (FMAX - FMIN) * SPEC_H)


IMG_W, IMG_H = 960, 720
dec_big = Image.fromarray(dec.astype(np.uint8)).resize((IMG_W, IMG_H), Image.BICUBIC)
src_img = Image.fromarray(src.astype(np.uint8))

total = len(rx) / SR
HOLD = 3.0
n_frames = int((total + HOLD) * FPS)


def frame(i):
    t = i / FPS
    im = Image.new("RGB", (FW, FH), BG)
    d = ImageDraw.Draw(im)
    d.text((60, 52), "BORA", font=f(FONT_B, 46), fill=FG)
    d.text((60, 110), "görüntü  →  ses  →  görüntü", font=f(FONT_R, 26), fill=MUTED)

    # görüntü alanı
    ix, iy = 60, 170
    d.rectangle([ix - 2, iy - 2, ix + IMG_W + 1, iy + IMG_H + 1], outline=(40, 40, 52), width=2)
    idx = t * SR
    done = int(np.sum(starts + line_len <= idx))
    if done > 0:
        h = int(done * IMG_H / H)
        im.paste(dec_big.crop((0, 0, IMG_W, h)), (ix, iy))
    if 0 < done < H or (done == 0 and idx > starts[0]):
        y = iy + int(done * IMG_H / H)
        d.rectangle([ix, y, ix + IMG_W - 1, y + int(IMG_H / H)], fill=ACCENT)
    if done == 0 and idx < starts[0]:
        d.text((ix + IMG_W // 2, iy + IMG_H // 2), "1900 Hz başlangıç tonu...", font=f(FONT_R, 30),
               fill=MUTED, anchor="mm")

    # spektrogram
    sx, sy = 60, 950
    t_now = min(t, total)
    right = int(t_now * px_per_sec)
    left = right - SPEC_W
    crop = Image.new("RGB", (SPEC_W, SPEC_H), BG)
    src_l = max(left, 0)
    if right > 0:
        piece = spec_scaled.crop((src_l, 0, right, SPEC_H))
        crop.paste(piece, (SPEC_W - piece.width, 0))
    im.paste(crop, (sx, sy))
    d.rectangle([sx - 2, sy - 2, sx + SPEC_W + 1, sy + SPEC_H + 1], outline=(40, 40, 52), width=2)
    d.line([sx + SPEC_W - 1, sy, sx + SPEC_W - 1, sy + SPEC_H], fill=FG, width=2)
    lab = f(FONT_M, 19)
    for fq, name in [(2300, "2300 Hz  beyaz"), (1500, "1500 Hz  siyah"), (1200, "1200 Hz  senkron")]:
        yy = sy + fy(fq)
        for xx in range(sx, sx + SPEC_W, 14):
            d.line([xx, yy, xx + 6, yy], fill=(90, 88, 105), width=1)
        d.text((sx + SPEC_W + 10, yy), name.split()[0] + " " + name.split()[1], font=lab, fill=MUTED, anchor="lm")
        d.text((sx + SPEC_W + 10, yy + 20), name.split()[2], font=lab, fill=(100, 98, 115), anchor="lm")

    # alt bilgi
    mm = lambda s: f"{int(s) // 60:02d}:{int(s) % 60:02d}"
    d.text((60, 1235), f"{mm(t_now)} / {mm(total)}", font=f(FONT_M, 30), fill=FG)
    d.text((FW - 60, 1235), f"satır {min(done, H)}/{H}", font=f(FONT_M, 30), fill=FG, anchor="ra")
    d.text((60, 1285), "simüle kanal: gürültü + yankı (hoparlör → mikrofon)", font=f(FONT_R, 22), fill=MUTED)

    if t > total + 0.4:
        a = min((t - total - 0.4) / 0.6, 1)
        ov = Image.new("RGBA", (FW, FH), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        od.rectangle([ix, iy + IMG_H - 110, ix + IMG_W, iy + IMG_H], fill=(10, 10, 14, int(200 * a)))
        od.text((ix + 30, iy + IMG_H - 55), f"{int(total)} saniye. Kablo yok, sadece ses.", font=f(FONT_B, 36),
                fill=FG + (int(255 * a),), anchor="lm")
        im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    return im


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        for t in [0.1, 8.0, 18.0, total + 2.5]:
            frame(int(t * FPS)).save(f"prev_{t:.0f}.png")
        sys.exit()
    pad = np.zeros(int(HOLD * SR))
    wavfile.write("audio_full.wav", SR, (np.concatenate([rx, pad]) * 32767).astype(np.int16))
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{FW}x{FH}", "-r", str(FPS), "-i", "-", "-i", "audio_full.wav",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                          "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                          "bora_demo.mp4"], stdin=subprocess.PIPE)
    for i in range(n_frames):
        p.stdin.write(frame(i).tobytes())
    p.stdin.close()
    p.wait()
    print("ok", n_frames)
