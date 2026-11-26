# Eğitilmiş Bir YOLO Modeli Nasıl Kullanılır?

Bu rehber, hazır veya kendi eğittiğin bir nesne tespiti modelini kullanmayı öğretir. Örneklerde, [veri seti ve eğitim rehberindeki](veri_seti_ve_model_egitimi.md) **kupa tespiti** modelinin eğitildiğini varsayıyoruz.

```text
Eğitilmiş modeli yükle → Görüntü ver → Tahmini al
                     → Sonucu yorumla → Göster veya kaydet
```

## 1. Elindeki model ne işe yarar?

Bir model, eğitimde öğrendiği sayısal bilgileri kullanarak yeni görüntülerde tahmin yapar. Bu bilgilere **ağırlıklar**, modelin yeni görüntüyü işlemesine **çıkarım (inference)** denir.

YOLO'nun nesne tespiti modelleri, görüntüde hangi nesnelerin bulunduğunu ve nerede olduklarını tahmin eder. **YOLO26n**, YOLO26 ailesinin küçük, nano sürümüdür. [YOLO26 hakkında](https://docs.ultralytics.com/models/yolo26/)

Eğitimden gelen `best.pt`, model yapısını ve öğrenilmiş ağırlıkları taşıyan bir dosyadır. Örneğin kupa için eğittiğin model, yeni görüntülerde kupaları arar. Dosyanın adını değiştirmek modelin öğrendiği sınıfları değiştirmez.

Modeli kullanmak için yeniden eğitim yapman gerekmez. Eğitimin yapıldığı bilgisayardan başka bir cihaza taşıyabilirsin; yeni cihazda uyumlu yazılım ortamı gerekir.

## 2. Kullanım için ne gerekir?

- Python ve modeli çalıştıracak kütüphane.
- Eğitilmiş model dosyası: bu örnekte `best.pt`.
- İşlenecek fotoğraf, video veya erişilebilir kamera.

Python ortamına Ultralytics kütüphanesini kur:

```bash
python -m pip install ultralytics
```

Örnekleri kolay takip etmek için dosyaları şöyle düzenleyebilirsin:

```text
model_kullanimi/
├── best.pt
├── fotograf.jpg
└── video.mp4
```

`best.pt` dosyasını eğitim çıktındaki `weights` klasöründen buraya kopyala. Aşağıdaki kodları bu klasörde çalıştır. Farklı bir konum kullanırsan dosya yolunu değiştir.

Tahmin üretirken eğitim fotoğrafları, etiket dosyaları veya `data.yaml` gerekmez; bunlar veri hazırlama, eğitim ve değerlendirme aşamalarına aittir.

## 3. Modeli belleğe yükle

```python
from ultralytics import YOLO

model = YOLO("best.pt")
print(model.names)
```

`YOLO("best.pt")`, modeli kullanıma hazırlar. Her yeni görüntü için tekrar yüklemek yerine modeli bir kez yükleyip birçok görüntüde kullanabilirsin.

Tek sınıflı kupa modelinde beklenen sınıf listesi şöyledir:

```text
{0: 'kupa'}
```

Buradaki `0`, sınıf numarasıdır. Bu numaranın anlamı modele bağlıdır; başka bir modelde `0` farklı bir nesne olabilir. Bir modelin hangi nesneleri bildiğini dosya adından tahmin etmek yerine `model.names` üzerinden kontrol et.

## 4. Bir fotoğrafta tahmin yap

```python
from ultralytics import YOLO

model = YOLO("best.pt")
sonuclar = model.predict(source="fotograf.jpg", conf=0.5)
sonuclar[0].save(filename="sonuc.jpg")
```

Bu kodu `tahmin.py` adıyla aynı klasöre kaydedip terminalde çalıştırabilirsin:

```bash
python tahmin.py
```

`source`, işlenecek görüntüyü belirtir. `predict`, görüntüyü modelin girişine hazırlar, modeli çalıştırır ve sonuçları döndürür. Modelin ağırlıkları bu işlem sırasında yeniden öğrenilmez.

Tek fotoğraf gönderdiğimiz için ilk sonucu `sonuclar[0]` ile alıyoruz. `save`, tespit kutuları çizilmiş fotoğrafı `sonuc.jpg` olarak kaydeder. Modelin tahmin üretmesi ile sonucu çizmek veya kaydetmek ayrı işlemlerdir. [Tahmin kullanımı](https://docs.ultralytics.com/modes/predict/)

## 5. Sonuçlar ne anlama gelir?

Bir tespit üç temel bilgi içerir:

| Bilgi | Anlamı | Örnek |
| --- | --- | --- |
| Sınıf | Nesnenin türü | `kupa` |
| Güven skoru | Modelin o tespit için ürettiği güven değeri | `0.91` |
| Konum kutusu | Nesneyi çevreleyen dikdörtgen | `[100, 80, 240, 300]` |

Bu kutu örneği `x1, y1, x2, y2` biçimindedir: sol üst köşe `(100, 80)`, sağ alt köşe `(240, 300)`. Bu değerler görüntüdeki **piksel koordinatlarıdır**. Sol üst köşe başlangıçtır; x sağa, y aşağı doğru artar.

Yukarıdaki fotoğraf örneğinin devamında tespitleri okuyabilirsin:

```python
for sonuc in sonuclar:
    print("Bulunan nesne sayısı:", len(sonuc.boxes))

    for kutu in sonuc.boxes:
        sinif_no = int(kutu.cls[0].item())
        guven = float(kutu.conf[0].item())
        koordinatlar = kutu.xyxy[0].tolist()
        print(sonuc.names[sinif_no], guven, koordinatlar)
```

Hiç nesne bulunmazsa kutu döngüsü çalışmaz. Bir sonuç nesnesi dönmesi, mutlaka bir nesne bulunduğu anlamına gelmez. [Sonuç ve kutu alanları](https://docs.ultralytics.com/reference/engine/results/)

## 6. Ayarları nasıl seçersin?

| Ayar | Ne yapar? |
| --- | --- |
| `conf=0.5` | Bu eşikten düşük güvenli tespitleri eler. |
| `classes=[0]` | Yalnızca belirtilen sınıfın tespitlerini tutar. |
| `imgsz=640` | Modelin görüntüyü işlerken kullandığı giriş boyutunu belirler. |
| `device="cpu"` | Hesaplamayı işlemcide yapar. |
| `device=0` | Uyumlu GPU ve yazılım kurulumu varsa ilk GPU'yu kullanır. |
| `save=True` | Çizilmiş sonuçları çıktı klasörüne kaydeder. |
| `show=True` | Desteklenen grafik ortamında görüntü penceresi açar. |

**Güven skoru, modelin genel doğruluk oranı değildir.** `0.91` görmek, bütün modelin %91 doğru çalıştığı anlamına gelmez. Eşiği düşürmek daha fazla sonuç gösterir; yanlış tespitleri de artırabilir. Eşiği yükseltmek bazı yanlışları eleyebilir; gerçek nesneleri kaçırabilir.

`classes=[0]` kullanmak yeni bir sınıf öğretmez. Modelin zaten bildiği sonuçları filtreler. Modelin bilmediği bir nesneyi bulması gerekiyorsa uygun veriyle eğitim gerekir. [Ayarların açıklamaları](https://docs.ultralytics.com/usage/cfg/)

## 7. Video üzerinde kullan

Video, art arda gelen fotoğraf kareleridir. Her kare modelden geçirilir:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

for sonuc in model.predict(
    source="video.mp4",
    conf=0.5,
    stream=True,
    save=True,
    project="ciktilar",
    name="video",
):
    print("Bu karedeki nesne sayısı:", len(sonuc.boxes))
```

`stream=True`, bütün video sonuçlarını bellekte biriktirmek yerine sırayla üretir. Bu yüzden sonuçları `for` döngüsüyle tüketiyoruz. Çizilmiş video belirtilen çıktı klasörüne kaydedilir; aynı ad yeniden kullanılırsa klasöre numara eklenebilir. Gerçek kayıt yolunu terminalden kontrol et. [Video ve akış kullanımı](https://docs.ultralytics.com/modes/predict/)

Her karede iki kupa görülmesi, video boyunca toplam iki farklı kupa görüldüğünü kanıtlamaz. **Nesne takibi**, karelerdeki tespitleri eşleştirip takip numarası vermeye çalışan ek bir işlemdir. Normal tespit bu eşleştirmeyi yapmaz. [Tespit ve takip farkı](https://docs.ultralytics.com/modes/track/)

## 8. Canlı kamera üzerinde kullan

Terminalden temel kullanım:

```bash
yolo detect predict model=best.pt source=0 conf=0.5 show=True
```

`source=0`, sistemin ilk kamera kaynağını belirtir; başka bir kamera için `source=1` denenebilir. Görüntü penceresi için masaüstü oturumu ve pencere desteği olan OpenCV gerekir. Ekransız bir cihazda veya `headless` OpenCV ile `show=True` pencere açamaz; bu durumda kaydetme ya da ayrıca bir yayın uygulaması gerekir. Durdurmak için terminalde `Ctrl+C` kullanabilirsin.

## 9. Bilgisayar ve Raspberry Pi arasında ne değişir?

**Görüntü → model → tahmin** akışı değişmez. Aynı `.pt` dosyası uyumlu Python/PyTorch ortamında iki cihazda da kullanılabilir. Fark, işlem gücü ve kameraya erişimdir.

| Konu | Bilgisayar | Raspberry Pi |
| --- | --- | --- |
| Hesaplama | CPU veya uygun kurulumda desteklenen GPU | Genellikle ARM işlemcili CPU |
| Kamera | Dahili veya USB kamera | USB ya da uygun sürücü/kütüphane ile CSI kamera |
| Hız | Donanıma bağlıdır | Güçlü bir bilgisayara göre daha düşük FPS verebilir |
| Kurulum | İşletim sistemine uygun paketler | ARM mimarisi ve Pi işletim sistemiyle uyumlu paketler |

**FPS**, saniyede işlenen kare sayısıdır. Yüksek FPS, tespitlerin doğru olduğu anlamına gelmez. Küçük bir model veya daha küçük giriş boyutu daha az işlem gerektirebilir; özellikle küçük nesnelerde doğruluk değişebilir. Değişiklikten sonra sonuçları yeniden değerlendirmek gerekir.

Eğitimi bilgisayarda yapıp modeli Pi'ye taşımak mümkündür. Hızlandırıcıya uygun model formatına dönüştürme, eğitimden ayrı bir adımdır; Pi'ye taşımak tek başına yeniden eğitim gerektirmez. [Raspberry Pi rehberi](https://docs.ultralytics.com/guides/raspberry-pi/)

## 10. Sonuç beklediğin gibi değilse neye bakarsın?

- **Dosya bulunamıyor:** Modelin ve giriş görüntüsünün yolunu kontrol et. Göreli yollar, terminalin çalıştığı klasöre göre değerlendirilir.
- **Yanlış nesneler çıkıyor:** Doğru modeli yüklediğinden ve sınıf numaralarını doğru yorumladığından emin ol.
- **Nesne kaçıyor:** Işık, görüntü boyutu, örtülme ve güven eşiğini incele. Eğitimde görülmeyen koşullar için yeni veri gerekebilir.
- **Görüntü yavaş:** Donanımı, model boyutunu ve giriş boyutunu birlikte değerlendir.

Modeli birkaç kolay görüntüyle değerlendirmek yerine farklı koşullardaki yeni görüntülerde incele. Böylece modeli yalnızca çalıştırmış değil, nerede işe yaradığını ve nerede hata yaptığını da anlamış olursun.
