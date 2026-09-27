from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "hsnakkaya/teyit_scraper/master/articles.csv"
)

EXPECTED_COLUMNS = [
    "author",
    "claim",
    "date",
    "img_link_list",
    "is_claim",
    "link_list",
    "tags",
    "text_all",
    "title",
    "url",
    "verdict",
]


def find_project_root() -> Path:
    """
    Bu dosyanın şu konumda olduğu varsayılır:
    training/data_preparation/download_teyit_scraper.py
    """
    return Path(__file__).resolve().parents[2]


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
        },
    )

    with urlopen(request, timeout=120) as response:
        content = response.read()

    if not content:
        raise RuntimeError("İndirilen dosya boş.")

    output_path.write_bytes(content)


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def load_and_validate(file_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8",
        low_memory=False,
    )

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Eksik sütunlar bulundu: "
            + ", ".join(missing_columns)
        )

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    )

    return dataframe


def print_profile(
    dataframe: pd.DataFrame,
    file_path: Path,
) -> None:
    file_size_mb = file_path.stat().st_size / 1024**2

    print("=" * 60)
    print("TEYİT SCRAPER HAM VERİ KONTROLÜ")
    print("=" * 60)

    print(f"Dosya: {file_path}")
    print(f"Dosya boyutu: {file_size_mb:.2f} MB")
    print(f"SHA-256: {calculate_sha256(file_path)}")

    print(f"\nToplam kayıt: {len(dataframe):,}")
    print(f"Sütun sayısı: {len(dataframe.columns)}")

    print("\nSütunlar:")
    for column in dataframe.columns:
        print(f"- {column}")

    print("\nİddia kaydı dağılımı:")
    print(
        dataframe["is_claim"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nHüküm dağılımı:")
    print(
        dataframe["verdict"]
        .fillna("HÜKÜM YOK")
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nTarih aralığı:")
    print("En eski:", dataframe["date"].min())
    print("En yeni:", dataframe["date"].max())

    print("\nBoş alan sayıları:")
    important_columns = [
        "claim",
        "title",
        "url",
        "verdict",
        "text_all",
    ]

    for column in important_columns:
        empty_count = (
            dataframe[column].isna()
            | dataframe[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        ).sum()

        print(f"- {column}: {empty_count:,}")

    print("\nİlk üç iddia örneği:")

    example_columns = [
        "date",
        "claim",
        "title",
        "verdict",
    ]

    examples = dataframe[
        dataframe["is_claim"].eq(True)
    ][example_columns].head(3)

    print(examples.to_string(index=False))

    print("\nHam veri başarıyla indirildi ve doğrulandı.")


def main() -> None:
    project_root = find_project_root()

    output_path = (
        project_root
        / "data"
        / "raw"
        / "teyit_scraper_2019"
        / "articles.csv"
    )

    print("Dosya indiriliyor...")
    download_file(DATA_URL, output_path)

    print("Dosya okunuyor...")
    dataframe = load_and_validate(output_path)

    print_profile(dataframe, output_path)


if __name__ == "__main__":
    main()