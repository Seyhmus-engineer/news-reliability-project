from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "article_main_standardized.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "splits"
)


RANDOM_STATE = 42


def print_split_information(
    split_name: str,
    dataframe: pd.DataFrame,
) -> None:
    print("\n" + "=" * 50)
    print(split_name)
    print("=" * 50)

    print(f"Toplam kayıt: {len(dataframe)}")

    print("\nEtiket dağılımı:")
    print(
        dataframe["standard_label"]
        .value_counts()
    )

    print("\nEtiket yüzdeleri:")
    print(
        (
            dataframe["standard_label"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )
    )

    print("\nKaynak ve etiket dağılımı:")
    print(
        pd.crosstab(
            dataframe["source"],
            dataframe["standard_label"],
        )
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Ana veri seti bulunamadı: {INPUT_PATH}"
        )

    print("Ana haber veri seti bölünüyor...")

    dataframe = pd.read_csv(INPUT_PATH)

    required_columns = {
        "title",
        "content",
        "standard_label",
        "source",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Eksik sütunlar: {sorted(missing_columns)}"
        )

    # Önce verinin yüzde 15'ini test için ayır.
    train_validation_data, test_data = train_test_split(
        dataframe,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=dataframe["standard_label"],
    )

    # Kalan yüzde 85'in yaklaşık yüzde 17,65'i,
    # toplam verinin yüzde 15'ine karşılık gelir.
    train_data, validation_data = train_test_split(
        train_validation_data,
        test_size=0.1765,
        random_state=RANDOM_STATE,
        stratify=train_validation_data["standard_label"],
    )

    train_data = train_data.reset_index(drop=True)
    validation_data = validation_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = OUTPUT_DIRECTORY / "article_train.csv"
    validation_path = OUTPUT_DIRECTORY / "article_validation.csv"
    test_path = OUTPUT_DIRECTORY / "article_test.csv"

    train_data.to_csv(
        train_path,
        index=False,
        encoding="utf-8-sig",
    )

    validation_data.to_csv(
        validation_path,
        index=False,
        encoding="utf-8-sig",
    )

    test_data.to_csv(
        test_path,
        index=False,
        encoding="utf-8-sig",
    )

    print_split_information(
        split_name="EĞİTİM VERİSİ",
        dataframe=train_data,
    )

    print_split_information(
        split_name="DOĞRULAMA VERİSİ",
        dataframe=validation_data,
    )

    print_split_information(
        split_name="TEST VERİSİ",
        dataframe=test_data,
    )

    total_records = (
        len(train_data)
        + len(validation_data)
        + len(test_data)
    )

    print("\n" + "=" * 50)
    print("GENEL KONTROL")
    print("=" * 50)

    print(f"Orijinal kayıt sayısı: {len(dataframe)}")
    print(f"Bölünen toplam kayıt: {total_records}")

    if total_records != len(dataframe):
        raise ValueError(
            "Bölme işleminde kayıt kaybı oluştu."
        )

    print("\nDosyalar oluşturuldu:")
    print(f"- {train_path}")
    print(f"- {validation_path}")
    print(f"- {test_path}")


if __name__ == "__main__":
    main()