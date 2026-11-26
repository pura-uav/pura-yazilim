# Kendi Veri Setini Oluşturmak ve YOLO Modeli Eğitmek

Bu rehber, tek sınıflı bir **kupa tespiti** örneği üzerinden ilerler. Amaç, bir görüntüde kupanın nerede olduğunu bulan bir model hazırlamaktır. Temel kavramlar için önce [model rehberini](readme.md) okuyabilirsiniz.

```text
Amacı belirle → Görüntü topla → Etiketle → Veriyi böl
             → Eğit → Değerlendir → Eksikleri düzelt
```

## 1. Neyi öğreteceğine karar ver

Bir **sınıf**, modelin bulmasını istediğin nesne türüdür. Bu örnekte tek sınıfımız `kupa`; sınıf numarası `0` olacak. Birden fazla sınıf kullanırsan numaralar `0, 1, 2...` şeklinde ilerler.

Önce görevi seç:

| Görev | Verilecek etiket | Örnek soru |
| --- | --- | --- |
| Sınıflandırma | Görüntünün sınıfı | Fotoğrafta kupa var mı? |
| Nesne tespiti | Her nesne için dikdörtgen kutu ve sınıf | Kupalar nerede? |
| Segmentasyon | Nesnenin sınırlarını izleyen çokgen/maske | Tam olarak hangi pikseller kupa? |

Bu rehberde **nesne tespiti** kullanıyoruz. Kutuyla etiketlenmiş veri hazırlayacağız.

## 2. Görüntüleri topla

Kendi fotoğraflarını çekebilir, videolarından kareler alabilir veya kullanım koşulları uygun açık veri setlerinden yararlanabilirsin.

- Farklı kupalar, açılar, arka planlar, ışık koşulları ve uzaklıklar kullan.
- Modelin gerçek kullanımda göreceği koşulları temsil et. Sadece beyaz fonda çekilmiş fotoğraflarla yetinme.
- Aynı videonun peş peşe gelen neredeyse aynı karelerini çoğaltmak yerine çeşitlilik oluştur.
- Kupa bulunmayan sahneler de ekle; bunlar arka plan örnekleridir.

Kesin bir yeterli fotoğraf sayısı yoktur. Birkaç yüz çeşitli görüntüyle bir ilk deneme yapılabilir; yeterliliği, yeni görüntülerdeki sonuçlar belirler. Daha çok kopya yerine daha çeşitli ve doğru etiketli veri toplamaya çalış.

## 3. Etiketleme nasıl yapılır?

**Etiket**, fotoğraftaki doğru cevaptır. Nesne tespitinde her kupanın çevresine kutu çizip `kupa` sınıfını seçersin.

Etiketleme iki şekilde yürütülebilir:

- **Elle:** Her nesnenin kutusunu kendin çizersin.
- **Yarı otomatik:** Bir araç veya mevcut model kutu önerir; her öneriyi kontrol edip hataları düzeltirsin.

Örneğin [CVAT](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/) gibi bir araçta görüntüleri yükleyip `kupa` etiketini tanımlayabilir, kutuları çizebilir ve **Ultralytics YOLO Detection** biçiminde dışa aktarabilirsin. Otomatik etiketler kontrol edilmeden doğru cevap kabul edilmemelidir.

Her görüntüdeki **tüm kupaları** etiketle. Kutuyu nesneye yakın çiz; gereksiz arka plan ekleme. Kısmen örtülmüş nesneler için baştan ortak bir kural belirle ve tutarlı uygula. Görünen bir kupayı etiketsiz bırakmak, eğitim verisinde hatalı bir örnek oluşturur.

### YOLO etiket dosyası

`foto_001.jpg` görüntüsünün etiket dosyası `foto_001.txt` olur. Her nesne ayrı bir satırdır:

```text
sinif_no merkez_x merkez_y genislik yukseklik
```

Örneğin:

```text
0 0.50 0.50 0.20 0.40
```

Bu satır, görüntünün ortasında bulunan, genişliği görüntünün %20'si ve yüksekliği %40'ı kadar olan bir kupa kutusunu anlatır. Koordinatlar **piksel değil, 0–1 aralığında oranlardır**. Yatay merkez ve genişlik görüntü genişliğine; dikey merkez ve yükseklik görüntü yüksekliğine bölünür. Etiketleme aracı bu dönüşümü yapar. [Etiket satırının yapısı](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo/)

Örneğin 1000×500 piksellik fotoğrafta bu kutunun merkezi `(500, 250)`, genişliği 200 ve yüksekliği 200 pikseldir: `500/1000=0.5`, `250/500=0.5`, `200/1000=0.2`, `200/500=0.4`.

Kupa olmayan görüntü için boş bir etiket dosyası kullanılabilir. Kupa bulunan bir görüntüde etiketi unutmakla bunu karıştırma. [YOLO veri biçimi](https://docs.ultralytics.com/datasets/detect/)

## 4. Veriyi eğitim, doğrulama ve test olarak ayır

| Bölüm | Görevi | Örnek oran |
| --- | --- | --- |
| `train` | Ağırlıkların öğrenildiği bölüm | %70 |
| `val` | Eğitim sırasında modeli değerlendirme ve seçim | %20 |
| `test` | Son değerlendirme için ayrı tutulan bölüm | %10 |

Bu oranlar bir başlangıç örneğidir. Aynı görüntüyü veya çok benzer kopyalarını farklı bölümlere koyma. Özellikle videoları **kayıt/çekim grubu bazında** ayır; aynı kaydın ardışık karelerini hem eğitime hem teste dağıtma. Aksi hâlde sonuçlar olduğundan iyi görünebilir.

Veri artırma (**augmentation**), eğitim görüntülerinin uygun biçimde döndürülmesi, kırpılması veya renklerinin değiştirilmesidir. Yeni çekimlerin yerini tamamen tutmaz. Bu işlemleri bölme sonrasında eğitim bölümüne uygula; kutular da görüntüyle birlikte doğru dönüşmelidir.

## 5. Klasörleri ve veri tanımını hazırla

```text
kupa_veri_seti/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Örneğin `images/train/foto_001.jpg` ile `labels/train/foto_001.txt` eşleşmelidir. Farklı bölümlerde farklı çekimler bulunmalıdır.

`data.yaml`, verinin nerede olduğunu ve sınıf adlarını söyler:

```yaml
path: /tam/yol/kupa_veri_seti
train: images/train
val: images/val
test: images/test

names:
  0: kupa
```

`path` değerini kendi veri klasörünün **tam yolu** ile değiştir. `train`, `val` ve `test` bu klasöre göre yazılır. YAML içindeki sınıf numaraları etiketlerdeki numaralarla aynı olmalıdır. [Veri seti yapılandırması](https://docs.ultralytics.com/datasets/detect/)

## 6. Model eğitimde ne öğrenir?

Model önce görüntüde kutular ve sınıflar tahmin eder. Tahminlerle etiketler arasındaki farktan bir **kayıp (loss)** hesaplanır. Eğitim algoritması bu hatayı azaltmak için ağırlıkları küçük adımlarla değiştirir. Bu döngü tekrarlandıkça model örneklerdeki ortak özellikleri öğrenir.

Başlangıçta **önceden eğitilmiş** bir modelin bilgilerini kullanıp kendi verinle uyarlamak, sıfırdan öğrenmeye göre pratik bir başlangıçtır. Buna **transfer öğrenme / ince ayar (fine-tuning)** denir. [Eğitim mantığı](https://docs.ultralytics.com/guides/model-training-tips/)

## 7. Eğitimi başlat

Python ortamına kütüphaneyi kur:

```bash
python -m pip install ultralytics
```

`kupa_veri_seti` klasörünün bulunduğu dizinde örnek komut:

```bash
yolo detect train model=yolo26n.pt data=kupa_veri_seti/data.yaml epochs=50 imgsz=640 batch=8 project=egitim_ciktilari name=kupa_deneme
```

Burada resmi `yolo26n.pt` başlangıç ağırlıkları kullanılır; dosya yoksa ilk indirme için internet gerekir. Öğrenilecek sınıf, hazırladığın `data.yaml` içindeki `kupa` sınıfıdır.

| Ayar | Anlamı |
| --- | --- |
| `epochs=50` | Eğitim verisinin en fazla 50 tur işlenmesi; erken durdurma daha önce bitirebilir |
| `imgsz=640` | Eğitimde kullanılan giriş görüntüsü boyutu |
| `batch=8` | Bir eğitim adımında birlikte işlenen görüntü sayısı |
| `project`, `name` | Çıktıların kaydedileceği klasör ve denemenin adı |

Bu sayılar örnektir; başarı garantisi vermez. Bellek yetmezse `batch` azaltılabilir. CPU ile eğitim mümkündür; uygun GPU kurulumu işlemi hızlandırabilir. [Eğitim komutları](https://docs.ultralytics.com/modes/train/)

İlk çalıştırmanın ağırlıkları normalde burada bulunur:

```text
egitim_ciktilari/kupa_deneme/weights/best.pt
egitim_ciktilari/kupa_deneme/weights/last.pt
```

`best.pt`, doğrulama ölçütüne göre seçilen en iyi ağırlıklardır; `last.pt`, son kaydedilen ağırlıklardır. Aynı deneme adını tekrar kullanırsan klasöre numara eklenebilir; terminalde yazan gerçek çıktı yoluna bak.

## 8. Sonucun iyi olduğunu nasıl anlarsın?

- **Precision:** Modelin “kupa” dediklerinin ne kadarı gerçekten kupa?
- **Recall:** Görüntülerdeki gerçek kupaların ne kadarını buluyor?
- **mAP:** Kutuların konumunu ve tespit başarısını birlikte değerlendiren ölçüt ailesi. `mAP50` ve daha sıkı `mAP50-95` aynı ölçüm değildir.

Sadece eğitim kaybının düşmesine güvenme. Eğitim iyi giderken doğrulama başarısı kötüleşiyorsa model örnekleri ezberliyor olabilir; buna **aşırı öğrenme (overfitting)** denir. [Başarı ölçütleri](https://docs.ultralytics.com/guides/yolo-performance-metrics/)

Model ve ayar seçimi bittikten sonra ayırdığın test bölümünü değerlendir:

```bash
yolo detect val model=egitim_ciktilari/kupa_deneme/weights/best.pt data=kupa_veri_seti/data.yaml split=test
```

Test sonuçlarına tekrar tekrar bakıp ayar seçersen bu bölüm bağımsız bir son değerlendirme olmaktan çıkar. Ayrıca yeni çekimlerde yanlış ve kaçırılan tespitleri gözle incele. [Değerlendirme komutu](https://docs.ultralytics.com/modes/val/)

## 9. Eğitilmiş modeli kullan

```python
from ultralytics import YOLO

model = YOLO("egitim_ciktilari/kupa_deneme/weights/best.pt")
model.predict(source="yeni_fotograf.jpg", conf=0.5, save=True)
```

Artık yeni fotoğrafta kupaları arayan modelini kullanıyorsun. Hataları gruplandır, eksik koşullardan yeni görüntüler topla, etiketleri düzelt ve yeni bir eğitim denemesi yap. Amaç eğitim fotoğraflarını ezberlemek değil, **daha önce görülmemiş görüntülerde de doğru sonuç vermektir**.
