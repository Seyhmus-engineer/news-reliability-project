from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fctr"
    / "fctr_raw.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fctr_standardized.csv"
)


LABEL_MAPPING = {
    "yanlış": "fake",
    "doğru": "real",

    "kismen yanliş": "misleading",
    "yarı doğru": "misleading",
    "çoğunlukla yanlış": "misleading",
    "çoğunlukla doğru": "misleading",

    "karma": "uncertain",
    "sonuçlandırılamadı": "uncertain",
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
            f"FCTR dosyası bulunamadı: {INPUT_PATH}"
        )

    print("FCTR verisi hazırlanıyor...")

    dataframe = pd.read_csv(
        INPUT_PATH,
        sep="\t",
        encoding="utf-8",
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

    for column in ["claim", "evidence", "summary"]:
        dataframe[column] = clean_text(
            dataframe[column]
        )

    dataframe = dataframe[
        dataframe["claim"].str.len() >= 10
    ].copy()

    dataframe = dataframe.drop_duplicates(
        subset=["claim"],
        keep="first",
    )

    selected_columns = [
        "claim_id",
        "claim",
        "evidence",
        "summary",
        "standard_label",
        "label",
        "url",
        "date",
    ]

    dataframe = dataframe[selected_columns]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFCTR verisi başarıyla hazırlandı.")
    print(f"Dosya yolu: {OUTPUT_PATH}")
    print(f"Toplam kayıt: {len(dataframe)}")

    print("\nStandart etiket dağılımı:")
    print(
        dataframe["standard_label"]
        .value_counts(dropna=False)
    )


if __name__ == "__main__":
    main()