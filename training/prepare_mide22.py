from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mide22"
    / "mide22_raw.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mide22_standardized.csv"
)


LABEL_MAPPING = {
    "false": "fake",
    "true": "real",
    "other": "uncertain",
}


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
            f"MiDe22 dosyası bulunamadı: {INPUT_PATH}"
        )

    print("MiDe22 verisi hazırlanıyor...")

    dataframe = pd.read_csv(INPUT_PATH)

    print(f"\nHam kayıt sayısı: {len(dataframe)}")

    # Tweet veya etiketi boş olan kayıtları kaldır.
    dataframe = dataframe.dropna(
        subset=["tweet", "label"]
    ).copy()

    dataframe["tweet"] = clean_text(
        dataframe["tweet"]
    )

    dataframe["standard_label"] = (
        dataframe["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(LABEL_MAPPING)
    )

    missing_labels = dataframe[
        dataframe["standard_label"].isna()
    ]["label"].unique()

    if len(missing_labels) > 0:
        raise ValueError(
            f"Dönüştürülemeyen etiketler: {missing_labels}"
        )

    # Çok kısa ve anlamlı içerik taşımayan metinleri çıkar.
    dataframe = dataframe[
        dataframe["tweet"].str.len() >= 10
    ].copy()

    # Aynı tweet birden fazla kez bulunuyorsa yalnızca ilkini tut.
    dataframe = dataframe.drop_duplicates(
        subset=["tweet"],
        keep="first",
    )

    dataframe = dataframe[
        [
            "tweet",
            "standard_label",
            "label",
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

    print("\nMiDe22 verisi başarıyla hazırlandı.")
    print(f"Dosya yolu: {OUTPUT_PATH}")
    print(f"Temiz kayıt sayısı: {len(dataframe)}")

    print("\nStandart etiket dağılımı:")
    print(
        dataframe["standard_label"]
        .value_counts(dropna=False)
    )


if __name__ == "__main__":
    main()