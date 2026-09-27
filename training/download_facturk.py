from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "facturk"
    / "facturk_raw.csv"
)


def main() -> None:
    print("FACTurk veri seti indiriliyor...")

    dataset = load_dataset(
        "ealtuncu/FACTurk",
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

    print("\nFACTurk başarıyla kaydedildi.")
    print(f"Dosya yolu: {OUTPUT_PATH}")
    print(f"Toplam kayıt: {len(dataframe)}")

    print("\nSütunlar:")
    for column in dataframe.columns:
        print(f"- {column}")

    if "normalised_rating" in dataframe.columns:
        print("\nNormalleştirilmiş etiket dağılımı:")
        print(
            dataframe["normalised_rating"]
            .value_counts(dropna=False)
        )

    if "organisation" in dataframe.columns:
        print("\nKuruluş dağılımı:")
        print(
            dataframe["organisation"]
            .value_counts(dropna=False)
        )

    print("\nİlk üç kayıt:")
    print(
        dataframe[
            [
                "claim",
                "normalised_rating",
                "organisation",
            ]
        ].head(3)
    )


if __name__ == "__main__":
    main()