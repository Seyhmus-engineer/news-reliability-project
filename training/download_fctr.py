from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fctr"
    / "fctr_raw.csv"
)

DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "firatcekinel/FCTR/main/data/fctr.csv"
)


def main() -> None:
    print("FCTR veri seti indiriliyor...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    urlretrieve(
        DATASET_URL,
        OUTPUT_PATH,
    )

    dataframe = pd.read_csv(
        OUTPUT_PATH,
        sep="\t",
        encoding="utf-8",
    )
    print("\nEtiket dağılımı:")
    print(
        dataframe["label"]
        .value_counts(dropna=False)
    )

    print("\nFCTR başarıyla kaydedildi.")
    print(f"Dosya yolu: {OUTPUT_PATH}")
    print(f"Toplam kayıt: {len(dataframe)}")

    print("\nSütunlar:")
    for column in dataframe.columns:
        print(f"- {column}")

    print("\nİlk üç kayıt:")
    print(dataframe.head(3))

    print("\nBoş değer sayıları:")
    print(
        dataframe
        .isnull()
        .sum()
        .sort_values(ascending=False)
    )


if __name__ == "__main__":
    main()