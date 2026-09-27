from pathlib import Path
import re
import unicodedata

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

UNIFIED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)

DATASET_FILES = [
    "article_main_unified.csv",
    "facturk_unified.csv",
    "fctr_unified.csv",
    "mide22_unified.csv",
    "satiretr_unified.csv",
]


def normalize_text(text: object) -> str:
    """
    Metnin anlamını değiştirmeden karşılaştırmaya uygun
    bir anahtar oluşturur.
    """
    if pd.isna(text):
        return ""

    normalized = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    normalized = normalized.lower()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def main() -> None:
    print("Veri setleri arası tekrarlar denetleniyor...")

    loaded_dataframes = []

    for file_name in DATASET_FILES:
        file_path = UNIFIED_DIRECTORY / file_name

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dosya bulunamadı: {file_path}"
            )

        dataframe = pd.read_csv(file_path)

        required_columns = {
            "record_id",
            "dataset_name",
            "text_basic",
            "label_standard",
            "source",
        }

        missing_columns = required_columns.difference(
            dataframe.columns
        )

        if missing_columns:
            raise ValueError(
                f"{file_name} dosyasında eksik sütunlar: "
                f"{sorted(missing_columns)}"
            )

        loaded_dataframes.append(
            dataframe[
                [
                    "record_id",
                    "dataset_name",
                    "task_name",
                    "text_type",
                    "text_basic",
                    "label_standard",
                    "source",
                ]
            ].copy()
        )

    combined = pd.concat(
        loaded_dataframes,
        ignore_index=True,
    )

    combined["normalized_text"] = (
        combined["text_basic"]
        .apply(normalize_text)
    )

    empty_normalized = (
        combined["normalized_text"]
        .eq("")
        .sum()
    )

    if empty_normalized > 0:
        raise ValueError(
            f"Normalize edildikten sonra boş kalan "
            f"{empty_normalized} metin bulundu."
        )

    group_sizes = (
        combined
        .groupby("normalized_text")
        .size()
        .rename("duplicate_count")
    )

    duplicated = combined.merge(
        group_sizes,
        on="normalized_text",
        how="left",
    )

    duplicated = duplicated[
        duplicated["duplicate_count"] > 1
    ].copy()

    if not duplicated.empty:
        dataset_counts = (
            duplicated
            .groupby("normalized_text")["dataset_name"]
            .nunique()
            .rename("dataset_count")
        )

        label_counts = (
            duplicated
            .groupby("normalized_text")["label_standard"]
            .nunique()
            .rename("label_count")
        )

        duplicated = duplicated.merge(
            dataset_counts,
            on="normalized_text",
            how="left",
        )

        duplicated = duplicated.merge(
            label_counts,
            on="normalized_text",
            how="left",
        )
    else:
        duplicated["dataset_count"] = pd.Series(
            dtype="int64"
        )

        duplicated["label_count"] = pd.Series(
            dtype="int64"
        )

    cross_dataset_duplicates = duplicated[
        duplicated["dataset_count"] > 1
    ].copy()

    conflicting_labels = cross_dataset_duplicates[
        cross_dataset_duplicates["label_count"] > 1
    ].copy()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    duplicates_path = (
        OUTPUT_DIRECTORY
        / "cross_dataset_exact_duplicates.csv"
    )

    conflicts_path = (
        OUTPUT_DIRECTORY
        / "cross_dataset_label_conflicts.csv"
    )

    cross_dataset_duplicates.to_csv(
        duplicates_path,
        index=False,
        encoding="utf-8-sig",
    )

    conflicting_labels.to_csv(
        conflicts_path,
        index=False,
        encoding="utf-8-sig",
    )

    duplicate_group_count = (
        cross_dataset_duplicates["normalized_text"]
        .nunique()
        if not cross_dataset_duplicates.empty
        else 0
    )

    conflict_group_count = (
        conflicting_labels["normalized_text"]
        .nunique()
        if not conflicting_labels.empty
        else 0
    )

    print("\n" + "=" * 60)
    print("GENEL SONUÇ")
    print("=" * 60)

    print(f"Toplam kayıt: {len(combined)}")
    print(
        "Dosyalar arası birebir tekrar grubu: "
        f"{duplicate_group_count}"
    )
    print(
        "Tekrar gruplarındaki toplam kayıt: "
        f"{len(cross_dataset_duplicates)}"
    )
    print(
        "Farklı etiket taşıyan tekrar grubu: "
        f"{conflict_group_count}"
    )

    if not cross_dataset_duplicates.empty:
        print("\nTekrarların veri seti eşleşmeleri:")

        dataset_pairs = (
            cross_dataset_duplicates
            .groupby("normalized_text")["dataset_name"]
            .apply(
                lambda values: " + ".join(
                    sorted(set(values))
                )
            )
            .value_counts()
        )

        print(dataset_pairs)

        print("\nİlk tekrar örnekleri:")

        print(
            cross_dataset_duplicates[
                [
                    "record_id",
                    "dataset_name",
                    "label_standard",
                    "source",
                    "text_basic",
                ]
            ].head(10)
        )

    if conflict_group_count > 0:
        print("\nUYARI:")
        print(
            "Aynı metnin farklı veri setlerinde farklı "
            "etiketlerle bulunduğu kayıtlar tespit edildi."
        )

    print("\nRapor dosyaları:")
    print(f"- {duplicates_path}")
    print(f"- {conflicts_path}")


if __name__ == "__main__":
    main()