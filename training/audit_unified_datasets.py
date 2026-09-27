from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

UNIFIED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified"
)


DATASET_FILES = {
    "article_main": "article_main_unified.csv",
    "facturk": "facturk_unified.csv",
    "fctr": "fctr_unified.csv",
    "mide22": "mide22_unified.csv",
    "satiretr": "satiretr_unified.csv",
}


REQUIRED_COLUMNS = {
    "record_id",
    "dataset_name",
    "task_name",
    "text_type",
    "title_raw",
    "content_raw",
    "text_raw",
    "text_basic",
    "text_destyled",
    "label_original",
    "label_standard",
    "source",
    "url",
    "published_date",
    "cleaning_flags",
    "duplicate_group_id",
    "split_role",
}


def audit_dataset(
    dataset_key: str,
    file_name: str,
) -> pd.DataFrame:
    file_path = UNIFIED_DIRECTORY / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dosya bulunamadı: {file_path}"
        )

    dataframe = pd.read_csv(file_path)

    missing_columns = REQUIRED_COLUMNS.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{dataset_key} dosyasında eksik sütunlar: "
            f"{sorted(missing_columns)}"
        )

    print("\n" + "=" * 65)
    print(dataset_key.upper())
    print("=" * 65)

    print(f"Dosya: {file_path.name}")
    print(f"Toplam kayıt: {len(dataframe)}")

    print("\nGörev dağılımı:")
    print(
        dataframe["task_name"]
        .value_counts(dropna=False)
    )

    print("\nMetin türü dağılımı:")
    print(
        dataframe["text_type"]
        .value_counts(dropna=False)
    )

    print("\nEtiket dağılımı:")
    print(
        dataframe["label_standard"]
        .value_counts(dropna=False)
    )

    print("\nSplit rolü:")
    print(
        dataframe["split_role"]
        .value_counts(dropna=False)
    )

    print("\nZorunlu alanlardaki boş değerler:")
    print(
        dataframe[
            [
                "record_id",
                "dataset_name",
                "task_name",
                "text_type",
                "content_raw",
                "text_basic",
                "text_destyled",
                "label_standard",
                "source",
                "cleaning_flags",
                "split_role",
            ]
        ]
        .isnull()
        .sum()
    )

    duplicated_ids = (
        dataframe["record_id"]
        .duplicated()
        .sum()
    )

    duplicated_texts = (
        dataframe["text_basic"]
        .duplicated()
        .sum()
    )

    print(f"\nDosya içi tekrar eden record_id: {duplicated_ids}")
    print(f"Dosya içi birebir tekrar eden metin: {duplicated_texts}")

    empty_text_count = (
        dataframe["text_basic"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"Boş metin sayısı: {empty_text_count}")

    return dataframe


def main() -> None:
    print("Ortak veri dosyaları denetleniyor...")

    loaded_datasets = []

    for dataset_key, file_name in DATASET_FILES.items():
        dataframe = audit_dataset(
            dataset_key=dataset_key,
            file_name=file_name,
        )

        loaded_datasets.append(dataframe)

    combined = pd.concat(
        loaded_datasets,
        ignore_index=True,
        sort=False,
    )

    print("\n" + "=" * 65)
    print("GENEL DENETİM")
    print("=" * 65)

    print(f"Toplam birleşik kayıt: {len(combined)}")

    duplicate_record_ids = (
        combined["record_id"]
        .duplicated()
        .sum()
    )

    print(
        "Veri setleri arasında çakışan record_id: "
        f"{duplicate_record_ids}"
    )

    print("\nVeri seti kayıt sayıları:")
    print(
        combined["dataset_name"]
        .value_counts(dropna=False)
    )

    print("\nGörev kayıt sayıları:")
    print(
        combined["task_name"]
        .value_counts(dropna=False)
    )

    print("\nMetin türü kayıt sayıları:")
    print(
        combined["text_type"]
        .value_counts(dropna=False)
    )

    print("\nTüm etiketlerin dağılımı:")
    print(
        combined["label_standard"]
        .value_counts(dropna=False)
    )

    print("\nKaynak sayısı:")
    print(
        combined["source"]
        .nunique(dropna=True)
    )

    if duplicate_record_ids > 0:
        raise ValueError(
            "Veri setleri arasında record_id çakışması bulundu."
        )

    print("\nOrtak şema denetimi başarıyla tamamlandı.")


if __name__ == "__main__":
    main()