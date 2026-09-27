from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "turkish_fake_news_main.csv"
)


def main() -> None:
    print("Veri seti indiriliyor...")

    dataset = load_dataset(
        "isakulaksiz/turkish-fake-news-detection",
        split="train",
    )

    dataframe = dataset.to_pandas()

    dataframe = dataframe.rename(
        columns={
            "description": "content",
            "status": "label",
            "Resource": "source",
        }
    )

    dataframe["label"] = dataframe["label"].map(
        {
            0: "fake",
            1: "real",
        }
    )

    dataframe = dataframe[
        [
            "title",
            "content",
            "label",
            "source",
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

    print("\nVeri seti başarıyla kaydedildi.")
    print(f"Dosya yolu: {OUTPUT_PATH}")
    print(f"Toplam kayıt: {len(dataframe)}")

    print("\nEtiket dağılımı:")
    print(dataframe["label"].value_counts())

    print("\nKaynak dağılımı:")
    print(dataframe["source"].value_counts())


if __name__ == "__main__":
    main()