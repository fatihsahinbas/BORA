# BORA

Görüntüyü sese çevirip sesten geri okuyan deneysel bir akustik modem.

Arada kablo, Wi-Fi ya da Bluetooth yok. Gönderen taraf görüntüyü hoparlörden çalıyor, alan taraf mikrofonla dinleyip görüntüyü geri çiziyor. Fikir, radyo amatörlerinin kullandığı SSTV (Slow Scan Television) modlarından, özellikle Robot36'dan geliyor.

| Gönderilen | Alınan |
|---|---|
| ![src](docs/src.png) | ![dec](docs/dec.png) |

> Bu repo şimdilik Python ile yazılmış bir prototip. Rust tarafı yolda.

## Nasıl çalışıyor?

Her pikselin parlaklığı bir frekansa dönüşüyor. Telefonda birine resmi satır satır tarif ettiğinizi düşünün, ama renk adı söylemek yerine ıslık çalıyorsunuz: koyu piksel için kalın, açık piksel için ince bir ıslık.

```python
f = 1500 + piksel / 255 * 800      # siyah 1500 Hz, beyaz 2300 Hz
faz = 2 * np.pi * np.cumsum(f) / sr  # sürekli faz, frekans geçişlerinde tık sesi olmaz
ses = np.sin(faz)
```

### Sinyal formatı

| Bölüm | Frekans | Süre |
|---|---|---|
| Başlangıç tonu | 1900 Hz | 300 ms |
| Satır senkronu | 1200 Hz | 6 ms |
| Porch | 1500 Hz | 1 ms |
| Kırmızı tarama | 1500-2300 Hz | 72 ms (160 piksel × 0,45 ms) |
| Porch | 1500 Hz | 1 ms |
| Yeşil tarama | 1500-2300 Hz | 72 ms |
| Porch | 1500 Hz | 1 ms |
| Mavi tarama | 1500-2300 Hz | 72 ms |
| Porch | 1500 Hz | 1 ms |
| Bitiş tonu | 1900 Hz | 300 ms |

- Çözünürlük: 160×120, RGB
- Satır süresi: 226 ms
- Toplam iletim: yaklaşık 27,7 sn
- Örnekleme hızı: 44,1 kHz

### Kanal simülasyonu

Hoparlör ile mikrofon arasındaki ortamı taklit etmek için sinyale iki bozulma ekleniyor:

- 4 ms gecikmeli, 0,25 genlikli yankı
- 12 dB SNR beyaz gürültü

Alınan görüntüdeki renk saçaklanması bu yankının izi.

### Çözücü

1. **Bant geçiren filtre:** 1000-2500 Hz dışı atılıyor.
2. **Anlık frekans:** Hilbert dönüşümü ile analitik sinyal çıkarılıyor, fazın türevi frekansı veriyor.
3. **Yumuşatma:** 1200 Hz alçak geçiren filtre.
4. **Senkron tespiti:** 1050-1350 Hz aralığında yeterince uzun kalan bölgeler satır başı kabul ediliyor.
5. **Eğim düzeltme:** Tespit edilen satır başlarına doğrusal regresyon uyduruluyor, aykırı noktalar atılıyor.
6. **İnce ayar:** Senkron+porch şablonuna en iyi oturan ortak ofset ±300 örnek içinde aranıyor.
7. **Piksel okuma:** Her piksel aralığındaki frekans ortalaması parlaklığa geri çevriliyor.

#### Neden eğim düzeltme?

Zor olan sesi üretmek değil, karşı tarafın her satırın tam nerede başladığını bilmesi. İki cihazın ses kartı saati birebir aynı çalışmıyor. Fark çok küçük olsa bile satırlar birikerek kayıyor ve görüntü eğik çıkıyor.

Bu yüzden çözücü senkron darbelerine tek tek güvenmiyor. Hepsine birden düz bir çizgi oturtuyor ve hem gürültüden gelen titremeyi hem de saat farkından gelen kaymayı aynı anda düzeltiyor.

## Çalıştırma

```bash
pip install -r requirements.txt
python modem.py
```

Çıktılar:

| Dosya | İçerik |
|---|---|
| `bora_tx.wav` | Kanaldan geçmiş (gürültülü) ses, dinlenebilir |
| `src.png` | Gönderilen görüntü |
| `dec.png` | Sesten çözülen görüntü |
| `state.npz` | Ara veriler (video/görsel üretimi için) |

Örnek çıktı:

```
süre 27.7156462585034 sn | satır 120 | sync hata(örnek) 23
PSNR 24.18213803843762
```

### Video ve kapak görseli (opsiyonel)

`ffmpeg` gerekir.

```bash
python render.py   # bora_demo.mp4: satır satır çizim + kayan spektrogram, sesli
python cover.py    # bora_kapak.png: gönderilen/alınan + spektrogram
```

## Sınırlar

- Kanal şimdilik simüle. Gerçek hoparlör-mikrofon testi yapılmadı.
- Hata düzeltme kodu yok. Bozulan piksel bozuk kalıyor.
- Senkron tespiti, satır süresinin önceden bilindiğini varsayıyor.
- Saat farkı (ppm sapması) ayrıca simüle edilmedi. Eğim düzeltme bunu kapsıyor ama test edilmedi.

## Yol haritası

- [ ] Gerçek hoparlör → mikrofon testi
- [ ] Başlangıç tonunda mod bilgisi (VIS kodu)
- [ ] Görüntü dışında rastgele dosya gönderimi
- [ ] Hata düzeltme (Reed-Solomon)
- [ ] Rust implementasyonu
- [ ] Canlı alıcı (mikrofondan gerçek zamanlı çözme)
# BORA
