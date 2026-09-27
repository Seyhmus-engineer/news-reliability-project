from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SATIRICAL_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "satiretr"
    / "satirical_zaytung_raw.csv"
)

NORMAL_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "satiretr"
    / "nonsatirical_aa_raw.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "satiretr_standardized.csv"
)


def read_csv_flexible(file_path: Path) -> pd.DataFrame:
    encodings = [
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                file_path,
                sep=None,
                engine="python",
                encoding=encoding,
            )
        except Exception as error:
            last_error = error

    raise ValueError(
        f"Dosya okunamadı: {file_path}\n"
        f"Son hata: {last_error}"
    )


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def prepare_satirical_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    prepared = pd.DataFrame()

    prepared["title"] = clean_text(
        dataframe["title"]
    )

    prepared["content"] = clean_text(
        dataframe["content"]
    )

    prepared["standard_label"] = "satire"
    prepared["source"] = "Zaytung"
    prepared["date"] = clean_text(
        dataframe["timestamp"]
    )

    return prepared


def prepare_normal_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    title = clean_text(
        dataframe["title"]
    )

    subtitle = clean_text(
        dataframe["subtitle"]
    )

    body = clean_text(
        dataframe["body"]
    )

    # Haber gövdesi boşsa alt başlığı içerik olarak kullan.
    content = body.where(
        body.str.len() > 0,
        subtitle,
    )

    prepared = pd.DataFrame()

    prepared["title"] = title
    prepared["content"] = content
    prepared["standard_label"] = "normal"
    prepared["source"] = "Anadolu Ajansı"
    prepared["date"] = clean_text(
        dataframe["date"]
    )

    return prepared


def main() -> None:
    if not SATIRICAL_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Satirik veri bulunamadı: "
            f"{SATIRICAL_INPUT_PATH}"
        )

    if not NORMAL_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Normal haber verisi bulunamadı: "
            f"{NORMAL_INPUT_PATH}"
        )

    print("SatireTR verisi hazırlanıyor...")

    satirical_raw = read_csv_flexible(
        SATIRICAL_INPUT_PATH
    )

    normal_raw = read_csv_flexible(
        NORMAL_INPUT_PATH
    )

    print(f"\nHam satirik kayıt: {len(satirical_raw)}")
    print(f"Ham normal kayıt: {len(normal_raw)}")

    satirical_data = prepare_satirical_data(
        satirical_raw
    )

    normal_data = prepare_normal_data(
        normal_raw
    )

    dataframe = pd.concat(
        [
            satirical_data,
            normal_data,
        ],
        ignore_index=True,
    )

    # Çok kısa veya boş içerikleri kaldır.
    dataframe = dataframe[
        (dataframe["title"].str.len() >= 5)
        & (dataframe["content"].str.len() >= 30)
    ].copy()

    # Aynı başlık ve içeriğe sahip tekrarları kaldır.
    dataframe = dataframe.drop_duplicates(
        subset=[
            "title",
            "content",
        ],
        keep="first",
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSatireTR başarıyla hazırlandı.")
    print(f"Dosya yolu: {OUTPUT_PATH}")
    print(f"Temiz kayıt sayısı: {len(dataframe)}")

    print("\nEtiket dağılımı:")
    print(
        dataframe["standard_label"]
        .value_counts(dropna=False)
    )

    print("\nKaynak dağılımı:")
    print(
        dataframe["source"]
        .value_counts(dropna=False)
    )

    print("\nİçerik uzunluğu istatistikleri:")
    print(
        dataframe["content"]
        .str.len()
        .describe()
    )


if __name__ == "__main__":
    main()