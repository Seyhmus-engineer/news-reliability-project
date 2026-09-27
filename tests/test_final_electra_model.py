# ============================================================
# FINAL TURKISH ELECTRA MODELİ
# YEREL YÜKLEME VE TAHMİN TESTİ
# ============================================================

from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================
# 1. PROJE VE MODEL YOLU
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "electra_turkish_article_60k_final"
)


# ============================================================
# 2. MODEL DOSYALARINI KONTROL ET
# ============================================================

print("=" * 70)
print("MODEL KLASÖRÜ KONTROLÜ")
print("=" * 70)

print(
    "Proje yolu:",
    PROJECT_ROOT,
)

print(
    "Model yolu:",
    MODEL_DIR,
)

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Model klasörü bulunamadı:\n{MODEL_DIR}"
    )

required_files = [
    "config.json",
    "model.safetensors",
]

for required_file in required_files:

    required_path = (
        MODEL_DIR
        / required_file
    )

    if not required_path.exists():
        raise FileNotFoundError(
            f"Model dosyası bulunamadı:\n{required_path}"
        )

    print(
        required_file,
        "→ BULUNDU",
    )


# ============================================================
# 3. CİHAZ SEÇİMİ
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\n" + "=" * 70)
print("CİHAZ BİLGİSİ")
print("=" * 70)

print(
    "PyTorch:",
    torch.__version__,
)

print(
    "CUDA aktif:",
    torch.cuda.is_available(),
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

print(
    "Kullanılan cihaz:",
    device,
)


# ============================================================
# 4. TOKENIZER'I YÜKLE
# ============================================================

print("\n" + "=" * 70)
print("TOKENIZER YÜKLENİYOR")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),

    # İnternete bağlanmadan yalnızca yerel dosyaları kullan.
    local_files_only=True,

    use_fast=True,
)

print(
    "Tokenizer sınıfı:",
    tokenizer.__class__.__name__,
)

print(
    "Tokenizer başarıyla yüklendi."
)


# ============================================================
# 5. MODELİ YÜKLE
# ============================================================

print("\n" + "=" * 70)
print("ELECTRA MODELİ YÜKLENİYOR")
print("=" * 70)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        str(MODEL_DIR),

        # İnternete bağlanmadan yalnızca yerel modeli kullan.
        local_files_only=True,
    )
)

model = model.to(device)
model.eval()

print(
    "Model sınıfı:",
    model.__class__.__name__,
)

print(
    "Label eşlemesi:",
    model.config.id2label,
)

print(
    "Model başarıyla yüklendi."
)


# ============================================================
# 6. ÖRNEK HABERLER
# ============================================================

sample_news = [
    (
        "Türkiye Cumhuriyet Merkez Bankası, "
        "faiz kararının Para Politikası Kurulu "
        "toplantısının ardından açıklanacağını bildirdi."
    ),

    (
        "Dünyanın bütün ülkeleri aynı anda para "
        "kullanmayı bıraktı ve yeni sisteme geçti."
    ),
]


# ============================================================
# 7. TOKENİZASYON
# ============================================================

encoded_inputs = tokenizer(
    sample_news,

    truncation=True,
    max_length=512,

    padding=True,

    return_tensors="pt",
)

encoded_inputs = {
    key: value.to(device)
    for key, value in encoded_inputs.items()
}


# ============================================================
# 8. TAHMİN
# ============================================================

with torch.inference_mode():

    outputs = model(
        **encoded_inputs
    )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )

    predicted_ids = torch.argmax(
        probabilities,
        dim=-1,
    )


# ============================================================
# 9. SONUÇLARI YAZDIR
# ============================================================

print("\n" + "=" * 70)
print("TAHMİN SONUÇLARI")
print("=" * 70)

for index, news_text in enumerate(
    sample_news
):

    predicted_id = int(
        predicted_ids[index].item()
    )

    predicted_label = (
        model.config.id2label.get(
            predicted_id,
            str(predicted_id),
        )
    )

    real_probability = float(
        probabilities[index, 0].item()
    )

    fake_probability = float(
        probabilities[index, 1].item()
    )

    confidence = max(
        real_probability,
        fake_probability,
    )

    print(
        f"\nHaber {index + 1}:"
    )

    print(
        news_text
    )

    print(
        "Tahmin:",
        predicted_label,
    )

    print(
        "Real olasılığı:",
        f"%{real_probability * 100:.2f}",
    )

    print(
        "Fake olasılığı:",
        f"%{fake_probability * 100:.2f}",
    )

    print(
        "Güven:",
        f"%{confidence * 100:.2f}",
    )


print("\n" + "=" * 70)
print("FINAL ELECTRA YEREL TESTİ TAMAMLANDI")
print("=" * 70)