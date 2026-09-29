
# 📰 Türkçe Haber Güvenilirliği Analiz Sistemi

**Türkçe haber metinlerini doğal dil işleme (NLP) ve derin öğrenme yöntemleriyle analiz eden, Turkish ELECTRA tabanlı haber sınıflandırma sistemi.**

Bu proje, internet üzerindeki Türkçe haber içeriklerini metinsel özelliklerine göre analiz ederek **gerçek** veya **şüpheli** olarak sınıflandırmayı amaçlayan bir yapay zekâ uygulamasıdır. Sistem; ince ayar (fine-tuning) uygulanmış Turkish ELECTRA modeli, FastAPI tabanlı backend ve Google Chrome tarayıcı eklentisinden oluşmaktadır.

> ⚠️ **Önemli Uyarı:** Bu proje deneysel bir metin sınıflandırma sistemidir; haberlerin veya iddiaların doğruluğunu kesin olarak kanıtlayan bir doğrulama mekanizması değildir. Modelin tahminleri bağımsız ve güvenilir kaynaklardan kontrol edilmelidir.

---

## 📌 Proje Hakkında

Türkçe Haber Güvenilirliği Analiz Sistemi, kullanıcıların çevrim içi haber içeriklerini metinsel açıdan değerlendirmelerine yardımcı olmak amacıyla geliştirilmiştir.

Proje üç temel bileşenden oluşmaktadır:

- **Yapay Zekâ Modeli:** Türkçe haber metinlerini gerçek ve sahte sınıflarına ayırmak üzere eğitilmiş Turkish ELECTRA Base modeli.
- **Backend API:** Haber metnini alan, ön işleme ve analiz işlemlerini gerçekleştiren, sonuçları API üzerinden döndüren FastAPI servisi.
- **Chrome Eklentisi:** Desteklenen haber sitelerinde içerikleri otomatik algılayan, analiz isteği sunan ve sonuçları tarayıcı üzerinde gösteren eklenti.

Sistem, uzun haber metinlerini parçalara ayırarak analiz edebilmekte ve çıkarım sırasında kullanılan GPU/CPU bilgisini gösterebilmektedir.

## ✨ Temel Özellikler

- 🤖 **Türkçe Doğal Dil İşleme:** Turkish ELECTRA Base ile haber metni sınıflandırma.
- 🌐 **Chrome Entegrasyonu:** Haber sayfalarında çalışan tarayıcı eklentisi.
- ⚡ **FastAPI Backend:** Haber analizi ve sağlık kontrolü için REST API.
- 📄 **Uzun Metin Analizi:** Uzun haberleri örtüşmeli parçalara ayırarak analiz etme ve sonuçları birleştirme.
- 🔍 **Otomatik Haber Algılama:** Uygun haber sayfalarını tespit ederek analiz seçeneği sunma.
- 🧹 **İçerik Çıkarma:** Desteklenen haber sitelerine özel içerik çıkarma kuralları.
- 🖥️ **GPU/CPU Durumu:** Aktif çıkarım donanımını eklenti üzerinde gösterme.
- 🛡️ **Sayfa Filtreleme:** Ana sayfa, arama, kategori, etiket, yazar ve arşiv sayfaları gibi haber olmayan sayfaları analiz dışında tutma.
- 🧪 **Test Altyapısı:** Model, API, davranış, entegrasyon ve performans testleri.

## 🏗️ Sistem Mimarisi

```text
┌─────────────────────────────────┐
│          Chrome Tarayıcı        │
│                                 │
│  ┌───────────────────────────┐  │
│  │       Chrome Eklentisi    │  │
│  │                           │  │
│  │ • Haber sayfası algılama  │  │
│  │ • İçerik çıkarma          │  │
│  │ • Analiz isteği           │  │
│  │ • Sonuç paneli            │  │
│  └─────────────┬─────────────┘  │
└────────────────┼────────────────┘
                 │ HTTP İsteği
                 ▼
┌─────────────────────────────────┐
│          FastAPI Backend        │
│                                 │
│ • Girdi doğrulama               │
│ • Metin ön işleme               │
│ • Tokenization                  │
│ • Uzun metin parçalama          │
│ • Tahmin sonuçlarını birleştirme│
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       Turkish ELECTRA Base      │
│                                 │
│ • İnce ayar uygulanmış model    │
│ • GPU/CPU çıkarımı              │
│ • Gerçek / Sahte sınıflandırma  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│           Analiz Sonucu         │
│                                 │
│ • Sınıflandırma                 │
│ • Güven skoru                   │
│ • İşlem bilgileri               │
└─────────────────────────────────┘
```

## 🧠 Yapay Zekâ Modeli

Projenin nihai modeli, Türkçe haber sınıflandırma görevi için ince ayar uygulanmış **Turkish ELECTRA Base** modelidir.

| Özellik | Değer |
|---|---|
| Model | Turkish ELECTRA Base |
| Önceden eğitilmiş model | `dbmdz/electra-base-turkish-mc4-cased-discriminator` |
| Görev | İkili metin sınıflandırma |
| Sınıflar | Gerçek (0), Sahte (1) |
| Toplam veri kümesi | 60.000 kayıt |
| Eğitim verisi | 48.000 |
| Doğrulama verisi | 6.000 |
| Test verisi | 6.000 |
| Yerel çıkarım donanımı | NVIDIA RTX 5070 Laptop GPU |
| Maksimum dizi uzunluğu | 512 token |
| Uzun metin yöntemi | Örtüşmeli parçalara ayırma |

### 📊 Nihai Model Performansı

Aşağıdaki metrikler, modelin ayrılmış 6.000 kayıtlık test kümesindeki değerlendirmesine dayanmaktadır.

| Metrik | Sonuç |
|---|---:|
| Accuracy (Doğruluk) | %95,25 |
| Macro F1 | %95,25 |
| ROC-AUC | %99,09 |
| Gerçek Haber Recall | %96,43 |
| Sahte Haber Recall | %94,13 |

### Karışıklık Matrisi (Confusion Matrix)

| Gerçek / Tahmin | Gerçek | Sahte |
|---|---:|---:|
| Gerçek haber | 2.891 | 109 |
| Sahte haber | 176 | 2.824 |

Test kümesinde 3.000 gerçek haberin 2.891'i, 3.000 sahte haberin ise 2.824'ü doğru sınıflandırılmıştır.

Bu sonuçlar, modelin hazırlanan test kümesindeki performansını göstermektedir. Farklı kaynaklardan gelen, daha önce görülmemiş veya eğitim verisinden farklı dağılıma sahip haberlerde aynı performansın elde edileceği garanti edilemez.

## 🧰 Kullanılan Teknolojiler

| Bileşen | Teknoloji |
|---|---|
| Programlama dili | Python 3.10 |
| Derin öğrenme | PyTorch |
| NLP ve Transformer | Hugging Face Transformers |
| Yapay zekâ modeli | Turkish ELECTRA Base |
| Backend | FastAPI, Uvicorn, Pydantic |
| Tarayıcı eklentisi | JavaScript, Chrome Extensions Manifest V3 |
| Veri işleme | Pandas, NumPy |
| Makine öğrenmesi | Scikit-learn, TF-IDF, Logistic Regression, Linear SVM |
| Geliştirme ortamı | PyCharm |
| Yerel GPU | NVIDIA RTX 5070 Laptop GPU |

## 📁 Proje Dosya Yapısı

```text
news-reliability-project/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py
│   │   ├── schemas/
│   │   │   └── prediction.py
│   │   ├── services/
│   │   │   └── model_service.py
│   │   └── main.py
│
├── extension/
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   └── site_rules.js
│
├── training/
│   ├── baselines/
│   ├── data_preparation/
│   ├── train_article_baselines.py
│   ├── train_claim_baselines.py
│   └── ...
│
├── tests/
│   ├── fake_news_test_site.py
│   ├── final_project_tests.py
│   ├── test_electra_behavior.py
│   ├── test_final_electra_model.py
│   └── test_model_service.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Not:** Eğitilmiş model ağırlıkları, yerel veri kümeleri, eski arşivler ve analiz raporları dosya boyutunu düşük tutmak amacıyla GitHub deposuna eklenmemiştir.

Yerel modelin beklenen dizini:

```text
models/electra_turkish_article_60k_final/
```

Bu klasörde model ağırlıkları, yapılandırma ve tokenizer dosyalarının bulunması gerekir.

## ⚙️ Kurulum

### 1. Projeyi Klonlama

```bash
git clone https://github.com/Seyhmus-engineer/news-reliability-project.git
cd news-reliability-project
```

### 2. Sanal Ortam Oluşturma

```bash
python -m venv .venv
```

Windows PowerShell üzerinde etkinleştirme:

```powershell
.\.venv\Scripts\Activate.ps1
```

Git Bash üzerinde etkinleştirme:

```bash
source .venv/Scripts/activate
```

### 3. Gerekli Kütüphaneleri Yükleme

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU ile çıkarım yapmak için işletim sistemi, NVIDIA sürücüsü, GPU ve CUDA ortamıyla uyumlu PyTorch sürümü kurulmalıdır. Kurulum komutu kullanılan donanıma göre değişebilir.

### 4. Eğitilmiş Modeli Yerleştirme

Model ağırlıkları dosya boyutları nedeniyle GitHub deposuna dahil edilmemiştir.

Model dosyaları aşağıdaki dizine yerleştirilmelidir:

```text
models/electra_turkish_article_60k_final/
```

Backend'in modeli bu dizinden yükleyecek şekilde yapılandırılmış olması gerekir.

## 🚀 Backend'i Çalıştırma

Proje ana dizinindeyken FastAPI servisini başlatmak için:

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Servis başarıyla başlatıldığında yerel API adresi:

```text
http://127.0.0.1:8000
```

Uygulamada etkinleştirilmişse etkileşimli API dokümantasyonuna şu adresten erişilebilir:

```text
http://127.0.0.1:8000/docs
```

### API Uç Noktaları

| Metot | Uç Nokta | Açıklama |
|---|---|---|
| GET | `/` | Backend ana uç noktası |
| GET | `/health` | Servis sağlık kontrolü |
| POST | `/predict` | Haber içeriğini analiz etme |

Tahmin uç noktası, backend şemasında tanımlanan kurallara bağlı olarak başlık, içerik, URL ve kaynak gibi haber bilgilerini kabul eder.

**Girdi doğrulama:** Servis, yapılandırılmış minimum token koşulunu karşılamayan metinleri reddeder. Uzun haberler parçalara ayrılır, her parça analiz edilir ve sonuçlar birleştirilerek nihai tahmin oluşturulur.

## 🧩 Chrome Eklentisini Yükleme

1. FastAPI backend servisini başlatın.
2. Google Chrome tarayıcısında aşağıdaki adresi açın:

   ```text
   chrome://extensions
   ```

3. Sağ üst köşeden **Geliştirici modu** seçeneğini etkinleştirin.
4. **Paketlenmemiş öğe yükle** seçeneğine tıklayın.
5. Proje içindeki `extension/` klasörünü seçin.
6. Desteklenen haber sitelerinden birinde haber sayfasını açın.
7. Eklenti uygun haber içeriğini algıladığında analiz isteğini kullanın.
8. Analiz sonucunu sağ tarafta açılan panelden görüntüleyin.

Eklenti; TRT Haber, Anadolu Ajansı, Sözcü, Cumhuriyet, Hürriyet ve Milliyet gibi desteklenen haber siteleri için siteye özel içerik çıkarma kuralları içerir.

Ana sayfalar, arama sonuçları, kategori, etiket, yazar ve arşiv sayfaları gibi haber niteliği taşımayan sayfalar analiz dışında tutulur.

## 🧪 Test ve Doğrulama

Proje; backend işlevleri, model davranışı, metin işleme, entegrasyon ve performans kontrollerini içeren otomatik test dosyalarına sahiptir.

### Nihai Sistem Testi

| Sonuç | Test Sayısı |
|---|---:|
| Başarılı | 24 |
| Başarısız | 0 |
| Atlanan | 1 |

Atlanan kontrol, test ortamında Node.js bulunmadığı için JavaScript sözdizimi doğrulamasıdır.

### Kontrollü Tarayıcı Gösterimi

Dört gerçek ve dört sahte haber örneğiyle gerçekleştirilen kontrollü gösterimde:

| Test Grubu | Doğru Sonuç |
|---|---:|
| Gerçek haber örnekleri | 4 / 4 |
| Sahte haber örnekleri | 4 / 4 |
| **Toplam** | **8 / 8** |

Bu sonuç yalnızca kontrollü gösterime aittir; modelin gerçek dünyadaki genel doğruluk oranı olarak değerlendirilmemelidir.

### Performans Ölçümleri

Yerel test ortamında elde edilen ölçümler:

| Metin Uzunluğu | Parça Sayısı | Ortalama Analiz Süresi |
|---|---:|---:|
| Kısa — 140 token | 1 | 0,0205 sn |
| Orta — 252 token | 1 | 0,0260 sn |
| Uzun — 1.520 token | 4 | 0,0974 sn |

Bu süreler kullanılan yerel donanım ve test koşullarına bağlıdır; farklı sistemlerde aynı performans garanti edilmez.

## 🔬 Veri Kümesi ve Değerlendirme

Proje, eğitim, doğrulama ve test kümelerine ayrılmış toplam 60.000 kayıtlık hazırlanmış veri kümesini kullanmaktadır.

| Veri Kümesi | Kayıt Sayısı |
|---|---:|
| Eğitim | 48.000 |
| Doğrulama | 6.000 |
| Test | 6.000 |
| **Toplam** | **60.000** |

Veri hazırlama ve değerlendirme süreçlerinde şu çalışmalar gerçekleştirilmiştir:

- Veri standartlaştırma ve ön işleme.
- Tekrarlı ve birbirine çok benzeyen kayıtların denetlenmesi.
- Veri bölme ve veri sızıntısı kontrolleri.
- Kaynak örüntüsü ve model davranışı analizleri.
- TF-IDF tabanlı Logistic Regression ve Linear SVM modelleriyle temel karşılaştırmalar.
- Ayrılmış test kümesinde sınıflandırma metrikleri ve karışıklık matrisi ile değerlendirme.

Ham veri kümeleri ve kaynak haber içerikleri bu GitHub deposuna dahil edilmemiştir. Harici veri kümelerini indirirken ilgili veri lisansları ve kullanım koşulları incelenmelidir.

## ⚠️ Sınırlamalar

Bu proje bir makine öğrenmesi tabanlı metin sınıflandırıcısıdır; **otomatik ve kesin bir doğruluk teyit sistemi değildir.**

- Yüksek güven skoru, haberin veya iddianın doğru olduğunu kanıtlamaz.
- Düşük güven skoru, haberin yanlış olduğunu kanıtlamaz.
- Model; yazım biçimi, kelime seçimi, haber uzunluğu, konu ve eğitim verisinden farklı kaynak özelliklerinden etkilenebilir.
- Yeni gelişen olaylar ve eğitim dağılımı dışındaki haberlerde hatalı tahminler oluşabilir.
- İçerik çıkarma başarısı, ziyaret edilen web sayfasının yapısına bağlıdır.
- Test kümesi ve kontrollü gösterim sonuçları, gerçek dünyadaki performansı tek başına temsil etmez.

Önemli haber ve iddialar, birden fazla güvenilir ve bağımsız kaynak üzerinden doğrulanmalıdır.

## 🔐 Gizlilik ve Güvenlik

Proje, analiz için gönderilen haber metinlerini yerel backend üzerinden işleyebilecek şekilde tasarlanmıştır.

- Backend kullanıcının kendi bilgisayarında çalıştırılabilir.
- Eğitilmiş model ağırlıkları ve yerel veri kümeleri GitHub deposuna dahil edilmemiştir.
- Dağıtımdan önce tarayıcı eklentisinin izinleri ve backend yapılandırması incelenmelidir.
- API anahtarları, erişim bilgileri, özel veri kümeleri ve kişisel bilgiler depoya eklenmemelidir.
- Ağ üzerinden veri aktarımı, kayıt tutma ve dağıtım davranışı backend ve eklenti yapılandırmasına bağlıdır.

## 🛣️ Gelecek Geliştirmeler

Projenin ilerleyen aşamalarında değerlendirilebilecek geliştirmeler:

- Daha fazla ve daha çeşitli haber kaynağı üzerinde değerlendirme.
- Görülmemiş kaynaklar ve farklı zaman dilimleri üzerinde kapsamlı testler.
- Yanıltıcı veya kasıtlı olarak değiştirilmiş metinlere karşı dayanıklılığın artırılması.
- Daha fazla haber sitesi için içerik çıkarma kuralları.
- Model optimizasyonu ve çıkarım süresinin iyileştirilmesi.
- Güvenilir harici doğrulama kaynaklarıyla isteğe bağlı entegrasyon.

## 👨‍💻 Geliştirici

**Şeyhmus**  
GitHub: [@Seyhmus-engineer](https://github.com/Seyhmus-engineer)

**Staj Projesi:** DİSKİ Genel Müdürlüğü

## 📄 Lisans

Bu depoya henüz bir açık kaynak lisansı eklenmemiştir. Lisans eklenene kadar projenin yeniden kullanımı ve dağıtımı yürürlükteki telif hakkı hükümlerine tabidir.
