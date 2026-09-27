from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "facturk_standardized.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified"
    / "facturk_unified.csv"
)


DATASET_NAME = "facturk"
TASK_NAME = "claim_veracity_multiclass"
TEXT_TYPE = "fact_checked_claim"


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
            f"FACTurk dosyası bulunamadı: {INPUT_PATH}"
        )

    print("FACTurk ortak şemaya dönüştürülüyor...")

    dataframe = pd.read_csv(INPUT_PATH)

    required_columns = {
        "claim",
        "title",
        "content",
        "standard_label",
        "normalised_rating",
        "organisation",
        "date_published",
        "url",
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
            "claim",
            "standard_label",
        ]
    ).copy()

    claim_raw = (
        dataframe["claim"]
        .astype(str)
        .str.strip()
    )

    claim_basic = normalize_whitespace(
        dataframe["claim"]
    )

    title_raw = normalize_whitespace(
        dataframe["title"]
    )

    report_content_raw = normalize_whitespace(
        dataframe["content"]
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
        "misleading",
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
        f"facturk_{index:06d}"
        for index in range(1, len(dataframe) + 1)
    ]

    unified["dataset_name"] = DATASET_NAME
    unified["task_name"] = TASK_NAME
    unified["text_type"] = TEXT_TYPE

    # Rapor başlığı yalnızca denetim amacıyla saklanır.
    unified["title_raw"] = title_raw.values

    # Ana model metni doğrulanan iddianın kendisidir.
    unified["content_raw"] = claim_raw.values
    unified["text_raw"] = claim_raw.values
    unified["text_basic"] = claim_basic.values

    # Kaynak izi temizliği henüz uygulanmadı.
    unified["text_destyled"] = claim_basic.values

    unified["label_original"] = (
        dataframe["normalised_rating"]
        .fillna("")
        .astype(str)
        .values
    )

    unified["label_standard"] = label_standard.values

    unified["source"] = normalize_whitespace(
        dataframe["organisation"]
    ).values

    unified["url"] = normalize_whitespace(
        dataframe["url"]
    ).replace("", pd.NA).values

    unified["published_date"] = normalize_whitespace(
        dataframe["date_published"]
    ).replace("", pd.NA).values

    unified["cleaning_flags"] = (
        "whitespace_normalized;claim_only_model_text"
    )

    unified["duplicate_group_id"] = pd.NA
    unified["split_role"] = "unassigned"

    # Doğrulama raporu saklanır fakat model girdisine eklenmez.
    unified["reference_content_raw"] = (
        report_content_raw.values
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    unified.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFACTurk ortak veri dosyası oluşturuldu.")
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

    print("\nKaynak dağılımı:")
    print(
        unified["source"]
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

    print("\nOrtalama iddia uzunluğu:")
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
            ]
        ].head(2)
    )


if __name__ == "__main__":
    main()