from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "satiretr_standardized.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified"
    / "satiretr_unified.csv"
)


DATASET_NAME = "satiretr"
TASK_NAME = "satire_detection_binary"
TEXT_TYPE = "full_article"


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
            f"SatireTR dosyası bulunamadı: {INPUT_PATH}"
        )

    print("SatireTR ortak şemaya dönüştürülüyor...")

    dataframe = pd.read_csv(INPUT_PATH)

    required_columns = {
        "title",
        "content",
        "standard_label",
        "source",
        "date",
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
            "title",
            "content",
            "standard_label",
        ]
    ).copy()

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

    title_basic = normalize_whitespace(
        dataframe["title"]
    )

    content_basic = normalize_whitespace(
        dataframe["content"]
    )

    label_standard = (
        dataframe["standard_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    allowed_labels = {
        "satire",
        "normal",
    }

    invalid_labels = set(
        label_standard.unique()
    ).difference(allowed_labels)

    if invalid_labels:
        raise ValueError(
            f"Geçersiz etiketler: {sorted(invalid_labels)}"
        )

    text_raw = (
        title_raw
        + "\n"
        + content_raw
    )

    text_basic = (
        title_basic
        + "\n"
        + content_basic
    )

    unified = pd.DataFrame()

    unified["record_id"] = [
        f"satiretr_{index:06d}"
        for index in range(1, len(dataframe) + 1)
    ]

    unified["dataset_name"] = DATASET_NAME
    unified["task_name"] = TASK_NAME
    unified["text_type"] = TEXT_TYPE

    unified["title_raw"] = title_raw.values
    unified["content_raw"] = content_raw.values
    unified["text_raw"] = text_raw.values
    unified["text_basic"] = text_basic.values

    # Ajans girişleri ve kaynak izleri henüz temizlenmedi.
    unified["text_destyled"] = text_basic.values

    # Bu veri setinde orijinal ve standart etiket aynıdır.
    unified["label_original"] = (
        dataframe["standard_label"]
        .astype(str)
        .values
    )

    unified["label_standard"] = label_standard.values

    unified["source"] = normalize_whitespace(
        dataframe["source"]
    ).values

    unified["url"] = pd.NA

    unified["published_date"] = normalize_whitespace(
        dataframe["date"]
    ).replace("", pd.NA).values

    unified["cleaning_flags"] = (
        "whitespace_normalized;"
        "satire_auxiliary_task;"
        "source_label_confounded;"
        "agency_patterns_not_removed"
    )

    unified["duplicate_group_id"] = pd.NA

    # Ana fake/real modelinin eğitim verisi değildir.
    unified["split_role"] = "satire_auxiliary_dataset"

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

    print("\nSatireTR ortak veri dosyası oluşturuldu.")
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

    print("\nMetin uzunluğu istatistikleri:")
    print(
        unified["text_basic"]
        .str.len()
        .describe()
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