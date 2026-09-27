from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


MIN_WORDS = 80
MAX_WORDS = 200


def find_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_words(text: object) -> list[str]:
    if pd.isna(text):
        return []

    return re.findall(
        r"\S+",
        str(text).strip(),
    )


def truncate_to_words(
    text: str,
    target_word_count: int,
) -> str:
    words = get_words(text)

    return " ".join(
        words[:target_word_count]
    ).strip()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dosya bulunamadı: {path}"
        )

    dataframe = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_columns = [
        "pair_id",
        "text",
        "label",
        "label_id",
        "source",
        "method",
        "is_synthetic",
        "split",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Eksik sütunlar: "
            + ", ".join(missing_columns)
        )

    dataframe["text"] = (
        dataframe["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe["original_word_count"] = (
        dataframe["text"]
        .apply(lambda text: len(get_words(text)))
    )

    return dataframe


def build_length_matched_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matched_rows = []
    excluded_pairs = []

    for pair_id, group in dataframe.groupby(
        "pair_id",
        sort=False,
    ):
        real_rows = group[
            group["label"].eq("real")
        ]

        fake_rows = group[
            group["label"].eq("fake")
        ]

        if len(real_rows) != 1 or len(fake_rows) != 1:
            excluded_pairs.append(
                {
                    "pair_id": pair_id,
                    "reason": "pair_structure_error",
                    "real_count": len(real_rows),
                    "fake_count": len(fake_rows),
                }
            )
            continue

        real_row = real_rows.iloc[0]
        fake_row = fake_rows.iloc[0]

        real_word_count = int(
            real_row["original_word_count"]
        )

        fake_word_count = int(
            fake_row["original_word_count"]
        )

        pair_minimum = min(
            real_word_count,
            fake_word_count,
        )

        if pair_minimum < MIN_WORDS:
            excluded_pairs.append(
                {
                    "pair_id": pair_id,
                    "reason": "pair_under_minimum",
                    "real_count": real_word_count,
                    "fake_count": fake_word_count,
                }
            )
            continue

        target_word_count = min(
            pair_minimum,
            MAX_WORDS,
        )

        for _, row in group.iterrows():
            output_row = row.to_dict()

            output_row["text"] = truncate_to_words(
                row["text"],
                target_word_count,
            )

            output_row["matched_word_count"] = (
                target_word_count
            )

            output_row["was_truncated"] = (
                int(row["original_word_count"])
                > target_word_count
            )

            matched_rows.append(output_row)

    matched_dataframe = pd.DataFrame(
        matched_rows
    )

    excluded_dataframe = pd.DataFrame(
        excluded_pairs
    )

    return matched_dataframe, excluded_dataframe


def validate_dataset(
    dataframe: pd.DataFrame,
) -> None:
    print("\n" + "=" * 65)
    print("UZUNLUK EŞİTLENMİŞ VERİ KONTROLÜ")
    print("=" * 65)

    print(
        "Toplam kayıt:",
        f"{len(dataframe):,}",
    )

    print(
        "Benzersiz çift:",
        f"{dataframe['pair_id'].nunique():,}",
    )

    print("\nEtiket dağılımı:")

    print(
        dataframe["label"]
        .value_counts()
        .to_string()
    )

    print("\nSplit dağılımı:")

    print(
        pd.crosstab(
            dataframe["split"],
            dataframe["label"],
        ).to_string()
    )

    print("\nYeni kelime uzunlukları:")

    print(
        dataframe.groupby("label")[
            "matched_word_count"
        ]
        .describe()[
            [
                "count",
                "mean",
                "50%",
                "min",
                "max",
            ]
        ]
        .round(2)
        .to_string()
    )

    pair_word_counts = (
        dataframe.groupby("pair_id")[
            "matched_word_count"
        ]
        .nunique()
    )

    unequal_pairs = pair_word_counts[
        pair_word_counts.ne(1)
    ]

    print(
        "\nUzunluğu eşit olmayan çift:",
        len(unequal_pairs),
    )

    pair_sizes = dataframe.groupby(
        "pair_id"
    ).size()

    invalid_pair_sizes = pair_sizes[
        pair_sizes.ne(2)
    ]

    print(
        "İki kayıttan oluşmayan çift:",
        len(invalid_pair_sizes),
    )

    train_ids = set(
        dataframe.loc[
            dataframe["split"].eq("train"),
            "pair_id",
        ]
    )

    validation_ids = set(
        dataframe.loc[
            dataframe["split"].eq("validation"),
            "pair_id",
        ]
    )

    test_ids = set(
        dataframe.loc[
            dataframe["split"].eq("test"),
            "pair_id",
        ]
    )

    split_leakage = (
        bool(train_ids & validation_ids)
        or bool(train_ids & test_ids)
        or bool(validation_ids & test_ids)
    )

    print(
        "Split sızıntısı:",
        split_leakage,
    )

    if len(unequal_pairs):
        raise RuntimeError(
            "Bazı gerçek-sentetik çiftlerin "
            "kelime uzunluğu eşit değil."
        )

    if len(invalid_pair_sizes):
        raise RuntimeError(
            "Bazı çiftlerde kayıt eksik."
        )

    if split_leakage:
        raise RuntimeError(
            "Splitler arasında pair_id sızıntısı var."
        )


def save_files(
    dataframe: pd.DataFrame,
    excluded_dataframe: pd.DataFrame,
    output_directory: Path,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_path = (
        output_directory
        / "article_length_matched_full.csv"
    )

    train_path = (
        output_directory
        / "article_length_matched_train.csv"
    )

    validation_path = (
        output_directory
        / "article_length_matched_validation.csv"
    )

    test_path = (
        output_directory
        / "article_length_matched_test.csv"
    )

    excluded_path = (
        output_directory
        / "excluded_pairs.csv"
    )

    dataframe.to_csv(
        full_path,
        index=False,
        encoding="utf-8-sig",
    )

    dataframe[
        dataframe["split"].eq("train")
    ].to_csv(
        train_path,
        index=False,
        encoding="utf-8-sig",
    )

    dataframe[
        dataframe["split"].eq("validation")
    ].to_csv(
        validation_path,
        index=False,
        encoding="utf-8-sig",
    )

    dataframe[
        dataframe["split"].eq("test")
    ].to_csv(
        test_path,
        index=False,
        encoding="utf-8-sig",
    )

    excluded_dataframe.to_csv(
        excluded_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nKaydedilen dosyalar:")
    print("-", full_path)
    print("-", train_path)
    print("-", validation_path)
    print("-", test_path)
    print("-", excluded_path)


def main() -> None:
    project_root = find_project_root()

    input_path = (
        project_root
        / "data"
        / "training"
        / "article_60k"
        / "article_pairs_full_60k.csv"
    )

    output_directory = (
        project_root
        / "data"
        / "training"
        / "article_60k_length_matched"
    )

    print("60 binlik veri okunuyor...")

    dataframe = load_dataset(
        input_path
    )

    print(
        "Başlangıç kayıt sayısı:",
        f"{len(dataframe):,}",
    )

    (
        matched_dataframe,
        excluded_dataframe,
    ) = build_length_matched_dataset(
        dataframe
    )

    print(
        "Uzunluk eşitleme sonrası kayıt:",
        f"{len(matched_dataframe):,}",
    )

    print(
        "Dışlanan çift:",
        f"{len(excluded_dataframe):,}",
    )

    validate_dataset(
        matched_dataframe
    )

    save_files(
        matched_dataframe,
        excluded_dataframe,
        output_directory,
    )


if __name__ == "__main__":
    main()