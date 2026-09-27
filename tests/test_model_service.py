# ============================================================
# MODEL SERVICE YEREL TESTİ
# ============================================================

from backend.app.services.model_service import (
    model_service,
)


def main() -> None:
    print("=" * 70)
    print("MODEL SERVİSİ TESTİ")
    print("=" * 70)

    model_service.load_model()

    print("\n" + "=" * 70)
    print("MODEL DURUMU")
    print("=" * 70)

    status = model_service.get_status()

    for key, value in status.items():
        print(
            f"{key}: {value}"
        )

    test_news = (
        "Türkiye Cumhuriyet Merkez Bankası, "
        "Para Politikası Kurulu toplantısının "
        "ardından faiz kararını kamuoyuna açıkladı."
    )

    result = model_service.predict(
        test_news
    )

    print("\n" + "=" * 70)
    print("MODEL SERVİSİ TAHMİNİ")
    print("=" * 70)

    print(
        "Tahmin:",
        result["label"],
    )

    print(
        "Label ID:",
        result["label_id"],
    )

    print(
        "Güven:",
        f"%{result['confidence'] * 100:.2f}",
    )

    print(
        "Real olasılığı:",
        f"%{result['probabilities']['real'] * 100:.2f}",
    )

    print(
        "Fake olasılığı:",
        f"%{result['probabilities']['fake'] * 100:.2f}",
    )

    print(
        "İşlenen token:",
        result["processed_token_count"],
    )

    print("\n" + "=" * 70)
    print("MODEL SERVICE TESTİ BAŞARILI")
    print("=" * 70)


if __name__ == "__main__":
    main()