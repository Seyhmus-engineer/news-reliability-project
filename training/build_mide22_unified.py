from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mide22_standardized.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified"
    / "mide22_unified.csv"
)


DATASET_NAME = "mide22"
TASK_NAME = "social_media_veracity_multiclass"
TEXT_TYPE = "social_media_post"


def normalize_whitespace(series: pd.Series) -> pd.Series:
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
            f"MiDe22 dosyası bulunamadı: {INPUT_PATH}"
        )

    print("MiDe22 ortak şemaya dönüştürülüyor...")

    dataframe = pd.read_csv(INPUT_PATH)

    required_columns = {
        "tweet",
        "standard_label",
        "label",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Eksik sütunlar: {sorted(missing_columns)}"
        )

    print(f"\nGirdi kayıt sayısı: {len(dataframe)}")

    dataframe = dataframe.dropna(
        subset=[
            "tweet",
            "standard_label",
        ]
    ).copy()

    tweet_raw = (
        dataframe["tweet"]
        .astype(str)
        .str.strip()
    )

    tweet_basic = normalize_whitespace(
        dataframe["tweet"]
    )

    label_standard = (
        dataframe["standard_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    allowed_labels = {
        "fake",
        "real",
        "uncertain",
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
        f"mide22_{index:06d}"
        for index in range(1, len(dataframe) + 1)
    ]

    unified["dataset_name"] = DATASET_NAME
    unified["task_name"] = TASK_NAME
    unified["text_type"] = TEXT_TYPE

    # Tweetlerde ayrı başlık bulunmuyor.
    unified["title_raw"] = pd.NA

    unified["content_raw"] = tweet_raw.values
    unified["text_raw"] = tweet_raw.values
    unified["text_basic"] = tweet_basic.values

    # Kullanıcı adı, URL ve etiket temizliği henüz yapılmadı.
    unified["text_destyled"] = tweet_basic.values

    unified["label_original"] = (
        dataframe["label"]
        .astype(str)
        .values
    )

    unified["label_standard"] = (
        label_standard.values
    )

    # Tek tek kullanıcı kaynakları bilinmediği için
    # platform bilgisi kaynak olarak tutulur.
    unified["source"] = "Twitter"

    unified["url"] = pd.NA
    unified["published_date"] = pd.NA

    unified["cleaning_flags"] = (
        "whitespace_normalized;"
        "social_media_text;"
        "mentions_and_urls_not_removed"
    )

    unified["duplicate_group_id"] = pd.NA

    # Ana performans testi değil, alan dışı stres testidir.
    unified["split_role"] = "social_media_stress_test"

    unified["reference_content_raw"] = pd.NA

    unified["original_record_id"] = [
        str(index)
        for index in dataframe.index
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    unified.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nMiDe22 ortak veri dosyası oluşturuldu.")
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

    print("\nSplit rolü:")
    print(
        unified["split_role"]
        .value_counts(dropna=False)
    )

    print("\nZorunlu alanlardaki boş değerler:")
    print(
        unified[
            [
                "record_id",
                "content_raw",
                "text_basic",
                "label_standard",
                "source",
            ]
        ]
        .isnull()
        .sum()
    )

    print("\nOrtalama tweet uzunluğu:")
    print(
        round(
            unified["text_basic"]
            .str.len()
            .mean(),
            2,
        )
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
                "split_role",
            ]
        ].head(2)
    )


if __name__ == "__main__":
    main()