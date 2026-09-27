from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "main"
    / "turkish_fake_news_main.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified"
    / "article_main_unified.csv"
)


DATASET_NAME = "article_main_trt_teyit"
TASK_NAME = "article_fake_news_binary"
TEXT_TYPE = "article_summary"


def normalize_whitespace(series: pd.Series) -> pd.Series:
    """
    Metnin anlamını değiştirmeden yalnızca gereksiz
    boşlukları ve satır aralarını düzenler.
    """
    return (
        series
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Ana haber veri seti bulunamadı: {INPUT_PATH}"
        )

    print("Ana haber verisi ortak şemaya dönüştürülüyor...")

    dataframe = pd.read_csv(INPUT_PATH)

    required_columns = {
        "title",
        "content",
        "label",
        "source",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Eksik sütunlar: {sorted(missing_columns)}"
        )

    print(f"\nHam kayıt sayısı: {len(dataframe)}")

    # Zorunlu alanları boş olan kayıtları kaldır.
    dataframe = dataframe.dropna(
        subset=[
            "title",
            "content",
            "label",
        ]
    ).copy()

    # Kaynaktan gelen metinlerin mümkün olduğunca
    # değiştirilmemiş kopyalarını sakla.
    title_raw = (
        dataframe["title"]
        .astype(str)
        .str.strip()
    )

    content_raw = (
        dataframe["content"]
        .astype(str)
        .str.strip()
    )

    # Yalnızca gereksiz boşlukları düzenle.
    title_basic = normalize_whitespace(
        dataframe["title"]
    )

    content_basic = normalize_whitespace(
        dataframe["content"]
    )

    label_standard = (
        dataframe["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    allowed_labels = {
        "fake",
        "real",
    }

    invalid_labels = set(
        label_standard.unique()
    ).difference(allowed_labels)

    if invalid_labels:
        raise ValueError(
            f"Geçersiz etiketler: {sorted(invalid_labels)}"
        )

    unified = pd.DataFrame()

    unified["record_id"] = [
        f"article_main_{index:06d}"
        for index in range(1, len(dataframe) + 1)
    ]

    unified["dataset_name"] = DATASET_NAME
    unified["task_name"] = TASK_NAME
    unified["text_type"] = TEXT_TYPE

    unified["title_raw"] = title_raw.values
    unified["content_raw"] = content_raw.values

    unified["text_raw"] = (
        title_raw
        + "\n"
        + content_raw
    ).values

    unified["text_basic"] = (
        title_basic
        + "\n"
        + content_basic
    ).values

    # Kaynak izi temizliği henüz yapılmadı.
    unified["text_destyled"] = unified["text_basic"]

    unified["label_original"] = (
        dataframe["label"]
        .astype(str)
        .values
    )

    unified["label_standard"] = (
        label_standard.values
    )

    unified["source"] = normalize_whitespace(
        dataframe["source"]
    ).values

    # Ana veri setinde bu bilgiler bulunmuyor.
    unified["url"] = pd.NA
    unified["published_date"] = pd.NA

    unified["cleaning_flags"] = (
        "whitespace_normalized"
    )

    # Yakın kopya analizi daha sonra yapılacak.
    unified["duplicate_group_id"] = pd.NA

    # Yeni split işlemi daha sonra yapılacak.
    unified["split_role"] = "unassigned"

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    unified.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nOrtak veri dosyası oluşturuldu.")
    print(f"Dosya yolu: {OUTPUT_PATH}")
    print(f"Toplam kayıt: {len(unified)}")

    print("\nEtiket dağılımı:")
    print(
        unified["label_standard"]
        .value_counts(dropna=False)
    )

    print("\nMetin türü dağılımı:")
    print(
        unified["text_type"]
        .value_counts(dropna=False)
    )

    print("\nKaynak ve etiket dağılımı:")
    print(
        pd.crosstab(
            unified["source"],
            unified["label_standard"],
        )
    )

    print("\nBoş değer sayıları:")
    print(
        unified[
            [
                "record_id",
                "title_raw",
                "content_raw",
                "text_basic",
                "label_standard",
                "source",
            ]
        ]
        .isnull()
        .sum()
    )

    print("\nİlk iki kayıt:")
    print(
        unified[
            [
                "record_id",
                "dataset_name",
                "text_type",
                "label_standard",
                "source",
            ]
        ].head(2)
    )


if __name__ == "__main__":
    main()