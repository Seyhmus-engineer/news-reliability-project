from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mide22"
    / "mide22_raw.csv"
)


def main() -> None:
    print("MiDe22 veri seti indiriliyor...")

    dataset = load_dataset(
        "ogozcelik/turkish-fake-news-detection",
        split="train",
    )

    dataframe = dataset.to_pandas()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nMiDe22 başarıyla kaydedildi.")
    print(f"Dosya yolu: {OUTPUT_PATH}")
    print(f"Toplam kayıt: {len(dataframe)}")

    print("\nSütunlar:")
    for column in dataframe.columns:
        print(f"- {column}")

    print("\nEtiket dağılımı:")
    print(
        dataframe["label"]
        .value_counts(dropna=False)
    )

    print("\nBoş değer sayıları:")
    print(
        dataframe
        .isnull()
        .sum()
    )

    print("\nİlk üç kayıt:")
    print(dataframe.head(3))


if __name__ == "__main__":
    main()