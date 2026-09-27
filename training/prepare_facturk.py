from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "facturk"
    / "facturk_raw.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "facturk_standardized.csv"
)


LABEL_MAPPING = {
    # Kesin yanlış
    "false": "fake",
    "yanlış": "fake",
    "yanlis": "fake",
    "yanliş": "fake",
    "yanliþ": "fake",

    # Kesin doğru
    "true": "real",
    "doğru": "real",
    "dogru": "real",

    # Yanıltıcı
    "misleading": "misleading",

    # Belirsiz ve karma
    "mixed": "uncertain",
    "unknown": "uncertain",
}


def normalize_label(value) -> str:
    if pd.isna(value):
        return "uncertain"

    cleaned_value = str(value).strip().lower()

    return LABEL_MAPPING.get(
        cleaned_value,
        "uncertain",
    )


def clean_text_column(series: pd.Series) -> pd.Series:
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

    print("FACTurk verisi hazırlanıyor...")

    dataframe = pd.read_csv(INPUT_PATH)

    dataframe["standard_label"] = (
        dataframe["normalised_rating"]
        .apply(normalize_label)
    )

    for column in ["claim", "title", "content"]:
        dataframe[column] = clean_text_column(
            dataframe[column]
        )

    # İddia metni bulunmayan kayıtları çıkar.
    dataframe = dataframe[
        dataframe["claim"].str.len() >= 10
    ].copy()

    # Aynı iddianın tekrarlanan kayıtlarını çıkar.
    dataframe = dataframe.drop_duplicates(
        subset=["claim"],
        keep="first",
    )

    selected_columns = [
        "claim",
        "title",
        "content",
        "standard_label",
        "normalised_rating",
        "organisation",
        "date_published",
        "url",
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

    print("\nVeri başarıyla hazırlandı.")
    print(f"Dosya yolu: {OUTPUT_PATH}")
    print(f"Toplam kayıt: {len(dataframe)}")

    print("\nStandart etiket dağılımı:")
    print(
        dataframe["standard_label"]
        .value_counts(dropna=False)
    )

    print("\nSilinen tekrarlar ve boş kayıtlar sonrasında")
    print("veri seti ayrı bir iddia veri seti olarak saklandı.")


if __name__ == "__main__":
    main()