"""BORA demo: görüntüyü sese çevir (SSTV benzeri FM), sesten geri çöz."""
import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SR = 44100
W, H = 160, 120
F_SYNC, F_BLACK, F_WHITE = 1200.0, 1500.0, 2300.0
T_SYNC = 0.006          # satır başı senkron darbesi
T_PORCH = 0.001
T_PIX = 0.00045         # piksel başına süre
VIS_T = 0.3             # başlangıç tonu (1900 Hz)


def make_source():
    img = Image.new("RGB", (W * 4, H * 4))
    d = ImageDraw.Draw(img)
    # gökyüzü gradyanı
    for y in range(H * 4):
        t = y / (H * 4)
        r = int(25 + 230 * t ** 1.2)
        g = int(40 + 120 * t ** 1.6)
        b = int(110 - 60 * t)
        d.line([(0, y), (W * 4, y)], fill=(r, g, b))
    # güneş
    d.ellipse([370, 170, 530, 330], fill=(255, 214, 120))
    # dağ siluetleri
    d.polygon([(0, 480), (0, 330), (120, 250), (230, 320), (330, 220), (470, 330),
               (560, 280), (640, 320), (640, 480)], fill=(70, 40, 90))
    d.polygon([(0, 480), (0, 390), (160, 330), (300, 400), (430, 350), (640, 410),
               (640, 480)], fill=(30, 20, 50))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 120)
    d.text((48, 40), "BORA", font=font, fill=(255, 255, 255))
    img = img.filter(ImageFilter.GaussianBlur(0.6)).resize((W, H), Image.LANCZOS)
    return np.asarray(img).astype(np.float64)


def encode(img):
    """Her satır: senkron + R, G, B taramaları. Sürekli faz FM."""
    freqs, line_starts = [], []
    freqs.append(np.full(int(VIS_T * SR), 1900.0))
    n_sync, n_porch, n_pix = int(T_SYNC * SR), int(T_PORCH * SR), T_PIX * SR
    pos = len(freqs[0])
    for y in range(H):
        line_starts.append(pos)
        seg = [np.full(n_sync, F_SYNC), np.full(n_porch, F_BLACK)]
        for c in range(3):
            n = int(round(W * n_pix))
            idx = np.minimum((np.arange(n) / n_pix).astype(int), W - 1)
            vals = img[y, idx, c] / 255.0
            seg.append(F_BLACK + vals * (F_WHITE - F_BLACK))
            seg.append(np.full(n_porch, F_BLACK))
        seg = np.concatenate(seg)
        freqs.append(seg)
        pos += len(seg)
    freqs.append(np.full(int(0.3 * SR), 1900.0))
    f = np.concatenate(freqs)
    phase = 2 * np.pi * np.cumsum(f) / SR
    audio = 0.6 * np.sin(phase)
    return audio, f, np.array(line_starts)


def channel(audio, snr_db=12, seed=7):
    """Hoparlör-mikrofon arası: gürültü + hafif yankı."""
    rng = np.random.default_rng(seed)
    echo = np.zeros_like(audio)
    d = int(0.004 * SR)
    echo[d:] = 0.25 * audio[:-d]
    x = audio + echo
    p = np.mean(x ** 2)
    noise = rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(x))
    return np.clip(x + noise, -1, 1)


def demod(audio):
    sos = butter(6, [1000, 2500], btype="band", fs=SR, output="sos")
    x = sosfiltfilt(sos, audio)
    phase = np.unwrap(np.angle(hilbert(x)))
    inst = np.diff(phase) * SR / (2 * np.pi)
    inst = np.append(inst, inst[-1])
    sos2 = butter(3, 1200, fs=SR, output="sos")
    return sosfiltfilt(sos2, inst)


def find_syncs(freq):
    """1200 Hz bölgelerini bul -> satır başları."""
    is_sync = (freq > 1050) & (freq < 1350)
    k = int(T_SYNC * SR * 0.6)
    run = np.convolve(is_sync.astype(float), np.ones(k), "same") > 0.9 * k
    edges = np.flatnonzero(np.diff(run.astype(int)) == 1)
    starts, last = [], -10 ** 9
    for e in edges:
        if e - last > 0.05 * SR:
            starts.append(e - k // 2)
            last = e
    starts = np.array(starts[:H], dtype=float)
    # eğim düzeltme: satır başları doğrusal olmalı, gürültülü tespitlere doğru uydur
    i = np.arange(len(starts))
    for _ in range(3):
        a, b = np.polyfit(i, starts, 1)
        res = starts - (a * i + b)
        keep = np.abs(res) < 2.5 * np.median(np.abs(res)) + 1
        a, b = np.polyfit(i[keep], starts[keep], 1)
    base = np.round(a * i + b).astype(int)
    # ince ayar: senkron+porch şablonuna en iyi oturan ortak ofset
    n_sync, n_porch = int(T_SYNC * SR), int(T_PORCH * SR)
    tmpl = np.concatenate([np.full(n_sync, F_SYNC), np.full(n_porch, F_BLACK)])
    best, best_lag = np.inf, 0
    for lag in range(-300, 301, 2):
        err = 0.0
        for s in base[::4]:
            seg = freq[s + lag:s + lag + len(tmpl)]
            if len(seg) == len(tmpl):
                err += np.mean((seg - tmpl) ** 2)
        if err < best:
            best, best_lag = err, lag
    return base + best_lag


def decode(freq, starts):
    out = np.zeros((H, W, 3))
    n_sync, n_porch, n_pix = int(T_SYNC * SR), int(T_PORCH * SR), T_PIX * SR
    n_chan = int(round(W * n_pix))
    for y, s in enumerate(starts):
        p = s + n_sync + n_porch
        for c in range(3):
            seg = freq[p:p + n_chan]
            if len(seg) < n_chan:
                break
            px = np.array([seg[int(i * n_pix):int((i + 1) * n_pix)].mean() for i in range(W)])
            out[y, :, c] = np.clip((px - F_BLACK) / (F_WHITE - F_BLACK), 0, 1) * 255
            p += n_chan + n_porch
    return out


if __name__ == "__main__":
    from scipy.io import wavfile
    src = make_source()
    audio, f_true, line_starts = encode(src)
    rx = channel(audio)
    f = demod(rx)
    starts = find_syncs(f)
    dec = decode(f, starts)
    print("süre", len(audio) / SR, "sn | satır", len(starts),
          "| sync hata(örnek)", np.abs(starts - line_starts[:len(starts)]).max())
    print("PSNR", 10 * np.log10(255 ** 2 / np.mean((dec - src) ** 2)))
    wavfile.write("bora_tx.wav", SR, (rx * 32767).astype(np.int16))
    Image.fromarray(src.astype(np.uint8)).save("src.png")
    Image.fromarray(dec.astype(np.uint8)).save("dec.png")
    np.savez("state.npz", rx=rx, f=f, starts=starts, src=src, dec=dec, line_starts=line_starts)
