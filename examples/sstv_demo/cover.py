import numpy as np
from PIL import Image, ImageDraw
from render import (
    ACCENT,
    BG,
    FG,
    FMAX,
    FMIN,
    FONT_B,
    FONT_M,
    FONT_R,
    MUTED,
    dec,
    f,
    spec_rgb,
    src,
    total,
)

FW, FH = 1080, 1350
im = Image.new("RGB", (FW, FH), BG)
d = ImageDraw.Draw(im)

d.text((60, 60), "Bir görüntüyü sese çevirdim,", font=f(FONT_B, 50), fill=FG)
d.text((60, 125), "sonra sesten geri okudum.", font=f(FONT_B, 50), fill=ACCENT)

# gönderilen / alınan
w, h = 465, 349
y0 = 250
sent = Image.fromarray(src.astype(np.uint8)).resize((w, h), Image.BICUBIC)
recv = Image.fromarray(dec.astype(np.uint8)).resize((w, h), Image.BICUBIC)
im.paste(sent, (60, y0 + 40))
im.paste(recv, (FW - 60 - w, y0 + 40))
lab = f(FONT_M, 22)
d.text((60, y0), "GÖNDERİLEN", font=lab, fill=MUTED)
d.text((FW - 60 - w, y0), "ALINAN", font=lab, fill=MUTED)

# tüm iletimin spektrogramı (zaman sıkıştırılmış)
sy = y0 + 40 + h + 90
d.text(
    (60, sy - 45),
    f"ARADAKİ SES  ·  {int(total)} sn  ·  {FMIN}-{FMAX} Hz",
    font=lab,
    fill=MUTED,
)
SW, SH = FW - 120, 250
spec = Image.fromarray(spec_rgb).resize((SW, SH), Image.LANCZOS)
im.paste(spec, (60, sy))
d.rectangle([58, sy - 2, 60 + SW + 1, sy + SH + 1], outline=(40, 40, 52), width=2)
# zoom şerit: 0.7 sn yakın plan
zy = sy + SH + 40
d.text((60, zy), "YAKIN PLAN  ·  3 satır", font=lab, fill=MUTED)
cps = 44100 / 256
a, b = int(9.0 * cps), int(9.7 * cps)
zoom = Image.fromarray(spec_rgb[:, a:b]).resize((SW, 150), Image.BICUBIC)
im.paste(zoom, (60, zy + 40))
d.rectangle([58, zy + 38, 60 + SW + 1, zy + 191], outline=(40, 40, 52), width=2)

d.text(
    (60, FH - 62),
    "SSTV warm-up  ·  the FM idea that started BORA",
    font=f(FONT_R, 26),
    fill=MUTED,
)
im.save("sstv_cover.png")
print("ok")
