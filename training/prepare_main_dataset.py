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
    / "article_main_standardized.csv"
)


def clean_text(series: pd.Series) -> pd.Series:
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
            f"Ana veri seti bulunamadı: {INPUT_PATH}"
        )

    print("Ana haber veri seti hazırlanıyor...")

    dataframe = pd.read_csv(INPUT_PATH)

    print(f"\nHam kayıt sayısı: {len(dataframe)}")

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

    for column in ["title", "content", "source"]:
        dataframe[column] = clean_text(
            dataframe[column]
        )

    dataframe["standard_label"] = (
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
        dataframe["standard_label"].unique()
    ).difference(allowed_labels)

    if invalid_labels:
        raise ValueError(
            f"Geçersiz etiketler: {sorted(invalid_labels)}"
        )

    # Kısa veya boş kayıtları çıkar.
    dataframe = dataframe[
        (dataframe["title"].str.len() >= 5)
        & (dataframe["content"].str.len() >= 30)
    ].copy()

    # Tamamen aynı başlık ve içeriğe sahip tekrarları çıkar.
    dataframe = dataframe.drop_duplicates(
        subset=[
            "title",
            "content",
        ],
        keep="first",
    )

    dataframe["title_length"] = (
        dataframe["title"].str.len()
    )

    dataframe["content_length"] = (
        dataframe["content"].str.len()
    )

    dataframe = dataframe[
        [
            "title",
            "content",
            "standard_label",
            "source",
            "title_length",
            "content_length",
        ]
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nAna haber veri seti başarıyla hazırlandı.")
    print(f"Dosya yolu: {OUTPUT_PATH}")
    print(f"Temiz kayıt sayısı: {len(dataframe)}")

    print("\nEtiket dağılımı:")
    print(
        dataframe["standard_label"]
        .value_counts(dropna=False)
    )

    print("\nKaynak ve etiket dağılımı:")
    print(
        pd.crosstab(
            dataframe["source"],
            dataframe["standard_label"],
        )
    )

    print("\nİçerik uzunluğu istatistikleri:")
    print(
        dataframe["content_length"].describe()
    )


if __name__ == "__main__":
    main()