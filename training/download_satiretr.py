from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "satiretr"
)

SATIRICAL_PATH = (
    OUTPUT_DIRECTORY
    / "satirical_zaytung_raw.csv"
)

NON_SATIRICAL_PATH = (
    OUTPUT_DIRECTORY
    / "nonsatirical_aa_raw.csv"
)

SATIRICAL_URL = (
    "https://raw.githubusercontent.com/"
    "auotomaton/satireTR/main/"
    "TurkishSatiricalNewsDataset/"
    "satirical-zaytung.csv"
)

NON_SATIRICAL_URL = (
    "https://raw.githubusercontent.com/"
    "auotomaton/satireTR/main/"
    "TurkishSatiricalNewsDataset/"
    "nonsatirical-aa.csv"
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
        f"CSV dosyası okunamadı: {file_path}\n"
        f"Son hata: {last_error}"
    )


def print_dataset_info(
    dataset_name: str,
    dataframe: pd.DataFrame,
) -> None:
    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    print(f"Toplam kayıt: {len(dataframe)}")

    print("\nSütunlar:")
    for column in dataframe.columns:
        print(f"- {column}")

    print("\nBoş değer sayıları:")
    print(
        dataframe
        .isnull()
        .sum()
        .sort_values(ascending=False)
    )

    print("\nİlk iki kayıt:")
    print(dataframe.head(2))


def main() -> None:
    print("SatireTR veri setleri indiriliyor...")

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    urlretrieve(
        SATIRICAL_URL,
        SATIRICAL_PATH,
    )

    urlretrieve(
        NON_SATIRICAL_URL,
        NON_SATIRICAL_PATH,
    )

    satirical_dataframe = read_csv_flexible(
        SATIRICAL_PATH
    )

    non_satirical_dataframe = read_csv_flexible(
        NON_SATIRICAL_PATH
    )

    print_dataset_info(
        dataset_name="SATİRİK — Zaytung",
        dataframe=satirical_dataframe,
    )

    print_dataset_info(
        dataset_name="NORMAL HABER — Anadolu Ajansı",
        dataframe=non_satirical_dataframe,
    )

    print("\nSatireTR başarıyla kaydedildi.")
    print(f"Satirik dosya: {SATIRICAL_PATH}")
    print(f"Normal haber dosyası: {NON_SATIRICAL_PATH}")

    total_records = (
        len(satirical_dataframe)
        + len(non_satirical_dataframe)
    )

    print(f"Toplam veri: {total_records}")


if __name__ == "__main__":
    main()