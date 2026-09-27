# ============================================================
# TURKISH ELECTRA DAVRANIŞ TESTİ
#
# Amaç:
# Modelin hangi metin özelliklerinde REAL veya FAKE
# kararı verdiğini kontrollü çiftler üzerinden incelemek.
#
# Bu bir doğruluk testi değildir.
# Her test çiftinde yalnızca belirli bir özellik değiştirilir.
#
# Proje sınıfları:
# 0 -> real
# 1 -> fake
# ============================================================

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================
# PROJE YOLLARI
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "electra_turkish_article_60k_final"
)

REPORTS_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
)

OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "electra_behavior_probe_results.csv"
)


# ============================================================
# TEST ÖRNEKLERİ
# ============================================================

TEST_CASES = [
    # --------------------------------------------------------
    # 1. İlk test dosyasındaki temel kontrol
    # --------------------------------------------------------
    {
        "case_id": "anchor",
        "variant": "plausible_real",
        "changed_feature": "Temel gerçekçi haber",
        "text": (
            "Türkiye Cumhuriyet Merkez Bankası, faiz kararının "
            "Para Politikası Kurulu toplantısının ardından "
            "kamuoyuna açıklanacağını bildirdi. Toplantıda para "
            "politikası görünümü, enflasyon gelişmeleri ve ekonomik "
            "verilerin değerlendirileceği belirtildi. Kararın kurumun "
            "resmî iletişim kanalları üzerinden duyurulacağı ifade "
            "edilirken yatırımcıların açıklamayı yakından takip "
            "ettiği aktarıldı."
        ),
    },
    {
        "case_id": "anchor",
        "variant": "absurd_fake",
        "changed_feature": "Açıkça gerçek dışı iddia",
        "text": (
            "Dünyanın bütün ülkeleri aynı anda para kullanmayı "
            "bıraktığını ve herhangi bir geçiş sürecine ihtiyaç "
            "duymadan tek bir sisteme geçtiğini açıkladı. Bankaların, "
            "nakit paraların ve bütün ödeme araçlarının bir gecede "
            "kaldırıldığı öne sürüldü. Yeni sistemde herkesin tüm "
            "ürünleri hiçbir ücret ödemeden alabileceği ve uygulamanın "
            "kalıcı olacağı iddia edildi."
        ),
    },

    # --------------------------------------------------------
    # 2. Nötr dil ve sansasyonel dil
    # --------------------------------------------------------
    {
        "case_id": "tone",
        "variant": "neutral",
        "changed_feature": "Nötr haber dili",
        "text": (
            "Kuzeykent Belediyesi, kent içi ulaşımda kullanılmak "
            "üzere on iki yeni otobüsün hizmete alınacağını açıkladı. "
            "Araçların gerekli teknik kontrollerinin tamamlandığı ve "
            "ilk etapta yoğun kullanılan güzergâhlarda çalışacağı "
            "belirtildi. Belediye yetkilileri, yeni araçların sefer "
            "sıklığını artıracağını ve toplu taşımadaki yoğunluğu "
            "azaltmayı amaçladıklarını bildirdi."
        ),
    },
    {
        "case_id": "tone",
        "variant": "sensational",
        "changed_feature": "Sansasyonel ve tıklama odaklı dil",
        "text": (
            "Şehirde ulaşımı kökten değiştirecek bomba karar açıklandı. "
            "Kuzeykent Belediyesi, herkesin şaşkına döndüğü dev hamleyle "
            "on iki yeni otobüsü hizmete alıyor. Kentte ulaşımın artık "
            "eskisi gibi olmayacağı ve bütün sorunların kısa süre içinde "
            "sona ereceği öne sürüldü. Yetkililerin yaptığı sürpriz "
            "açıklama vatandaşlar arasında büyük heyecan oluşturdu."
        ),
    },

    # --------------------------------------------------------
    # 3. Sayı değişikliği
    # --------------------------------------------------------
    {
        "case_id": "number_change",
        "variant": "normal_number",
        "changed_feature": "Makul sayı",
        "text": (
            "Kuzeykent Belediyesi, kent içi ulaşımda kullanılmak "
            "üzere on iki yeni otobüsün hizmete alınacağını açıkladı. "
            "Araçların teknik kontrollerinin tamamlandığı ve yoğun "
            "güzergâhlarda çalışacağı belirtildi. Yeni otobüslerle "
            "birlikte sefer sayısının artırılması ve vatandaşların "
            "bekleme süresinin azaltılması planlanıyor."
        ),
    },
    {
        "case_id": "number_change",
        "variant": "extreme_number",
        "changed_feature": "Makul olmayan büyük sayı",
        "text": (
            "Kuzeykent Belediyesi, kent içi ulaşımda kullanılmak "
            "üzere on iki bin yeni otobüsün aynı gün hizmete "
            "alınacağını açıkladı. Araçların teknik kontrollerinin "
            "bir gecede tamamlandığı ve bütün güzergâhlarda çalışacağı "
            "belirtildi. Yeni otobüslerle birlikte şehirde bekleme "
            "süresinin tamamen ortadan kaldırılacağı ileri sürüldü."
        ),
    },

    # --------------------------------------------------------
    # 4. Yer değişikliği
    # --------------------------------------------------------
    {
        "case_id": "location_change",
        "variant": "location_a",
        "changed_feature": "Kuzeykent",
        "text": (
            "Kuzeykent Belediyesi tarafından yapılan açıklamada, "
            "şehir merkezindeki yeni kültür merkezinin gelecek ay "
            "hizmete açılacağı bildirildi. Merkezde konferans salonu, "
            "kütüphane ve çalışma alanlarının yer aldığı belirtildi. "
            "Açılış öncesinde güvenlik ve teknik kontrollerin "
            "tamamlanacağı, tesisin haftanın altı günü vatandaşların "
            "kullanımına açık olacağı ifade edildi."
        ),
    },
    {
        "case_id": "location_change",
        "variant": "location_b",
        "changed_feature": "Güneykent",
        "text": (
            "Güneykent Belediyesi tarafından yapılan açıklamada, "
            "şehir merkezindeki yeni kültür merkezinin gelecek ay "
            "hizmete açılacağı bildirildi. Merkezde konferans salonu, "
            "kütüphane ve çalışma alanlarının yer aldığı belirtildi. "
            "Açılış öncesinde güvenlik ve teknik kontrollerin "
            "tamamlanacağı, tesisin haftanın altı günü vatandaşların "
            "kullanımına açık olacağı ifade edildi."
        ),
    },

    # --------------------------------------------------------
    # 5. Olumlu ve olumsuz ifade
    # --------------------------------------------------------
    {
        "case_id": "negation",
        "variant": "approved",
        "changed_feature": "Proje onaylandı",
        "text": (
            "Kuzeykent Belediyesi Meclisi, yeni spor kompleksinin "
            "yapımına ilişkin projeyi oy çokluğuyla kabul etti. "
            "Projenin mali ve teknik incelemelerinin tamamlandığı, "
            "çalışmaların gelecek ay başlayacağı belirtildi. Tesisin "
            "kapalı spor salonu, yüzme havuzu ve açık etkinlik "
            "alanlarından oluşacağı açıklandı."
        ),
    },
    {
        "case_id": "negation",
        "variant": "rejected",
        "changed_feature": "Proje reddedildi",
        "text": (
            "Kuzeykent Belediyesi Meclisi, yeni spor kompleksinin "
            "yapımına ilişkin projeyi oy çokluğuyla kabul etmedi. "
            "Projenin mali ve teknik incelemelerinde sorunlar "
            "belirlendiği, çalışmaların gelecek ay başlamayacağı "
            "ifade edildi. Tesisin yapımına ilişkin yeni bir karar "
            "alınana kadar sürecin durdurulduğu açıklandı."
        ),
    },

    # --------------------------------------------------------
    # 6. Kesin açıklama ve iddia dili
    # --------------------------------------------------------
    {
        "case_id": "certainty",
        "variant": "official_statement",
        "changed_feature": "Resmî açıklama dili",
        "text": (
            "Kuzeykent Belediyesi, yeni öğrenci merkezinin pazartesi "
            "günü hizmete açılacağını açıkladı. Merkezde ders çalışma "
            "salonları, internet erişimi ve kütüphane hizmeti "
            "sunulacağı belirtildi. Belediyenin resmî açıklamasında "
            "tesisin hafta içi ve hafta sonu belirlenen saatlerde "
            "açık olacağı bildirildi."
        ),
    },
    {
        "case_id": "certainty",
        "variant": "social_claim",
        "changed_feature": "Sosyal medya iddiası",
        "text": (
            "Sosyal medyada yayılan mesajlarda Kuzeykent Belediyesi "
            "tarafından yeni bir öğrenci merkezinin pazartesi günü "
            "hizmete açılacağı iddia edildi. Merkezde ders çalışma "
            "salonları, internet erişimi ve kütüphane hizmeti "
            "sunulacağı öne sürüldü. Paylaşımlarda tesisin bütün gün "
            "ücretsiz kullanılabileceği ileri sürüldü."
        ),
    },

    # --------------------------------------------------------
    # 7. Kaynak ve ajans işareti
    # --------------------------------------------------------
    {
        "case_id": "source_cue",
        "variant": "with_source",
        "changed_feature": "Kaynak ve muhabir bilgisi var",
        "text": (
            "Kuzeykent'te düzenlenen bilim şenliğine çok sayıda "
            "öğrencinin katıldığı bildirildi. Etkinlikte robotik, "
            "yazılım ve enerji alanında hazırlanan projeler "
            "sergilendi. Yetkililer, şenliğin gençlerin bilimsel "
            "çalışmalara ilgisini artırmayı amaçladığını belirtti. "
            "Kuzeykent Haber Ajansı muhabiri tarafından aktarılan "
            "bilgilere göre etkinlik üç gün sürecek."
        ),
    },
    {
        "case_id": "source_cue",
        "variant": "without_source",
        "changed_feature": "Kaynak bilgisi yok",
        "text": (
            "Kuzeykent'te düzenlenen bilim şenliğine çok sayıda "
            "öğrencinin katıldığı bildirildi. Etkinlikte robotik, "
            "yazılım ve enerji alanında hazırlanan projeler "
            "sergilendi. Yetkililer, şenliğin gençlerin bilimsel "
            "çalışmalara ilgisini artırmayı amaçladığını belirtti. "
            "Etkinliğin üç gün boyunca devam edeceği açıklandı."
        ),
    },

    # --------------------------------------------------------
    # 8. Temiz ve bozuk yazım
    # --------------------------------------------------------
    {
        "case_id": "writing_quality",
        "variant": "clean",
        "changed_feature": "Temiz ve düzenli yazım",
        "text": (
            "Kuzeykent Üniversitesi tarafından düzenlenen kariyer "
            "günleri başladı. Program kapsamında farklı sektörlerden "
            "uzmanların öğrencilerle bir araya geleceği belirtildi. "
            "Etkinliklerde meslek tanıtımları, öz geçmiş hazırlama "
            "çalışmaları ve iş görüşmesi uygulamalarının yapılacağı "
            "açıklandı. Programın üç gün devam edeceği bildirildi."
        ),
    },
    {
        "case_id": "writing_quality",
        "variant": "noisy",
        "changed_feature": "Yazım ve boşluk bozuklukları",
        "text": (
            "Kuzeykent Üniversitesi tarafından düzenlenen kariyer "
            "günleri başladı.Program kapsamında farklı sektörlerden "
            "uzmanların öğrencilerle bi araya geleceği belirtildi "
            "Etkinliklerde meslek tanıtımları,öz geçmiş hazırlama "
            "çalışmaları ve iş görüşmesi uygulamalarının yapılacagı "
            "açıklandı.Programın üç gün devam edecegi bildirildi."
        ),
    },

    # --------------------------------------------------------
    # 9. Makul ve fiziksel olarak imkânsız teknoloji
    # --------------------------------------------------------
    {
        "case_id": "plausibility",
        "variant": "plausible_technology",
        "changed_feature": "Makul teknoloji açıklaması",
        "text": (
            "Kuzeykent Üniversitesindeki araştırmacılar, güneş "
            "panellerinin enerji verimliliğini artırmayı amaçlayan "
            "yeni bir kaplama geliştirdi. Laboratuvar testlerinde "
            "kaplamanın bazı koşullarda panel performansına katkı "
            "sağladığı belirtildi. Araştırmacılar, sistemin seri "
            "üretime geçmeden önce uzun süreli testlerden geçirilmesi "
            "gerektiğini ifade etti."
        ),
    },
    {
        "case_id": "plausibility",
        "variant": "impossible_technology",
        "changed_feature": "Fiziksel olarak aşırı iddia",
        "text": (
            "Kuzeykent Üniversitesindeki araştırmacılar, herhangi bir "
            "enerji kaynağı kullanmadan sınırsız elektrik üreten bir "
            "cihaz geliştirdi. Cihazın yıllarca kesintisiz çalışacağı, "
            "bakım gerektirmeyeceği ve bütün şehirlerin enerji "
            "ihtiyacını tek başına karşılayacağı öne sürüldü. Yeni "
            "sistemin fizik kurallarından etkilenmediği iddia edildi."
        ),
    },
]


# ============================================================
# MODEL DOSYALARINI KONTROL ET
# ============================================================

def validate_model_directory() -> None:
    if not MODEL_DIRECTORY.is_dir():
        raise FileNotFoundError(
            "Model klasörü bulunamadı:\n"
            f"{MODEL_DIRECTORY}"
        )

    required_files = [
        "config.json",
        "model.safetensors",
    ]

    for file_name in required_files:
        file_path = (
            MODEL_DIRECTORY
            / file_name
        )

        if not file_path.is_file():
            raise FileNotFoundError(
                "Model dosyası bulunamadı:\n"
                f"{file_path}"
            )


# ============================================================
# ANA PROGRAM
# ============================================================

def main() -> None:
    print("=" * 78)
    print("TURKISH ELECTRA DAVRANIŞ TESTİ")
    print("=" * 78)

    validate_model_directory()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Kullanılan cihaz:",
        device,
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print()
    print("Tokenizer yükleniyor...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIRECTORY),
        local_files_only=True,
        use_fast=True,
    )

    print("Model yükleniyor...")

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            str(MODEL_DIRECTORY),
            local_files_only=True,
        )
    )

    model.to(
        device
    )

    model.eval()

    print(
        "Model label yapısı:",
        model.config.id2label,
    )

    texts = [
        record["text"]
        for record in TEST_CASES
    ]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        outputs = model(
            **encoded
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

    probabilities = (
        probabilities
        .detach()
        .cpu()
    )

    results: list[
        dict[str, Any]
    ] = []

    print()
    print("=" * 78)
    print("TEKİL TAHMİN SONUÇLARI")
    print("=" * 78)

    for index, test_case in enumerate(
        TEST_CASES
    ):
        real_probability = float(
            probabilities[
                index,
                0,
            ].item()
        )

        fake_probability = float(
            probabilities[
                index,
                1,
            ].item()
        )

        predicted_id = int(
            probabilities[
                index
            ]
            .argmax()
            .item()
        )

        predicted_label = (
            "real"
            if predicted_id == 0
            else "fake"
        )

        confidence = max(
            real_probability,
            fake_probability,
        )

        result = {
            "case_id": (
                test_case["case_id"]
            ),

            "variant": (
                test_case["variant"]
            ),

            "changed_feature": (
                test_case[
                    "changed_feature"
                ]
            ),

            "prediction": (
                predicted_label
            ),

            "confidence": (
                confidence
            ),

            "real_probability": (
                real_probability
            ),

            "fake_probability": (
                fake_probability
            ),

            "text": (
                test_case["text"]
            ),
        }

        results.append(
            result
        )

        print()
        print(
            "Test:",
            (
                f"{test_case['case_id']} "
                f"/ {test_case['variant']}"
            ),
        )

        print(
            "Değişen özellik:",
            test_case[
                "changed_feature"
            ],
        )

        print(
            "Tahmin:",
            predicted_label,
        )

        print(
            "Real:",
            f"%{real_probability * 100:.2f}",
        )

        print(
            "Fake:",
            f"%{fake_probability * 100:.2f}",
        )

        print(
            "Güven:",
            f"%{confidence * 100:.2f}",
        )

    # --------------------------------------------------------
    # Çift bazlı label değişimi
    # --------------------------------------------------------

    grouped_results: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for result in results:
        grouped_results[
            result["case_id"]
        ].append(
            result
        )

    print()
    print("=" * 78)
    print("ÇİFT BAZLI KARAR DEĞİŞİMİ")
    print("=" * 78)

    for case_id, case_results in (
        grouped_results.items()
    ):
        predictions = [
            result["prediction"]
            for result in case_results
        ]

        label_changed = (
            len(set(predictions)) > 1
        )

        print()
        print(
            "Test grubu:",
            case_id,
        )

        for result in case_results:
            print(
                (
                    f"  {result['variant']:<24}"
                    f"→ {result['prediction']:<5} "
                    f"| real "
                    f"%{result['real_probability'] * 100:6.2f} "
                    f"| fake "
                    f"%{result['fake_probability'] * 100:6.2f}"
                )
            )

        print(
            "  Karar değişti mi:",
            (
                "EVET"
                if label_changed
                else "HAYIR"
            ),
        )

    # --------------------------------------------------------
    # CSV kaydet
    # --------------------------------------------------------

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    field_names = [
        "case_id",
        "variant",
        "changed_feature",
        "prediction",
        "confidence",
        "real_probability",
        "fake_probability",
        "text",
    ]

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=field_names,
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    print()
    print("=" * 78)
    print("DAVRANIŞ TESTİ TAMAMLANDI")
    print("=" * 78)

    print(
        "Sonuç dosyası:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()