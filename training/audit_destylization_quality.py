from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "destyled"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)


DATASET_FILES = {
    "article_main": "article_main_destyled.csv",
    "facturk": "facturk_training_candidates_destyled.csv",
    "fctr": "fctr_destyled.csv",
    "mide22": "mide22_destyled.csv",
    "satiretr": "satiretr_destyled.csv",
}


SAMPLES_PER_DATASET = 15
RANDOM_STATE = 42


def validate_columns(
    dataframe: pd.DataFrame,
    file_name: str,
) -> None:
    required_columns = {
        "record_id",
        "dataset_name",
        "source",
        "label_standard",
        "text_basic",
        "text_masked",
        "text_destyled",
        "destyle_change_count",
        "text_length_before",
        "text_length_after",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{file_name} dosyasında eksik sütunlar: "
            f"{sorted(missing_columns)}"
        )


def contains_control_character(text: object) -> bool:
    if pd.isna(text):
        return False

    return bool(
        re.search(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F]",
            str(text),
        )
    )


def calculate_quality_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    output["length_ratio"] = (
        output["text_length_after"]
        .div(
            output["text_length_before"]
            .replace(0, pd.NA)
        )
    )

    output["is_changed"] = (
        output["destyle_change_count"] > 0
    )

    output["masked_equals_basic"] = (
        output["text_masked"]
        .fillna("")
        .eq(
            output["text_basic"]
            .fillna("")
        )
    )

    output["destyled_equals_basic"] = (
        output["text_destyled"]
        .fillna("")
        .eq(
            output["text_basic"]
            .fillna("")
        )
    )

    output["destyled_equals_masked"] = (
        output["text_destyled"]
        .fillna("")
        .eq(
            output["text_masked"]
            .fillna("")
        )
    )

    output["contains_control_character"] = (
        output["text_destyled"]
        .apply(contains_control_character)
    )

    output["suspicious_length_change"] = (
        (output["length_ratio"] < 0.60)
        | (output["length_ratio"] > 1.40)
    )

    output["empty_destyled"] = (
        output["text_destyled"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    output["url_placeholder_count"] = (
        output["text_destyled"]
        .fillna("")
        .astype(str)
        .str.count(r"<URL>")
    )

    output["user_placeholder_count"] = (
        output["text_destyled"]
        .fillna("")
        .astype(str)
        .str.count(r"<USER>")
    )

    output["source_placeholder_count"] = (
        output["text_destyled"]
        .fillna("")
        .astype(str)
        .str.count(r"<SOURCE>")
    )

    return output


def select_examples(
    dataframe: pd.DataFrame,
    dataset_key: str,
) -> pd.DataFrame:
    changed = dataframe[
        dataframe["is_changed"]
    ].copy()

    if changed.empty:
        return pd.DataFrame()

    sample_size = min(
        SAMPLES_PER_DATASET,
        len(changed),
    )

    examples = changed.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    ).copy()

    examples["analysis_dataset"] = dataset_key

    selected_columns = [
        "analysis_dataset",
        "record_id",
        "dataset_name",
        "source",
        "label_standard",
        "destyle_change_count",
        "text_length_before",
        "text_length_after",
        "length_ratio",
        "text_basic",
        "text_masked",
        "text_destyled",
    ]

    return examples[selected_columns]


def main() -> None:
    print("Kontrollü temizlik kalite denetimi başlatılıyor...")

    all_examples = []
    summary_rows = []
    suspicious_rows = []

    for dataset_key, file_name in DATASET_FILES.items():
        file_path = INPUT_DIRECTORY / file_name

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dosya bulunamadı: {file_path}"
            )

        dataframe = pd.read_csv(file_path)

        validate_columns(
            dataframe=dataframe,
            file_name=file_name,
        )

        dataframe = calculate_quality_columns(
            dataframe
        )

        examples = select_examples(
            dataframe=dataframe,
            dataset_key=dataset_key,
        )

        if not examples.empty:
            all_examples.append(examples)

        suspicious = dataframe[
            dataframe["suspicious_length_change"]
            | dataframe["contains_control_character"]
            | dataframe["empty_destyled"]
        ].copy()

        if not suspicious.empty:
            suspicious["analysis_dataset"] = (
                dataset_key
            )

            suspicious_rows.append(
                suspicious[
                    [
                        "analysis_dataset",
                        "record_id",
                        "source",
                        "label_standard",
                        "destyle_change_count",
                        "text_length_before",
                        "text_length_after",
                        "length_ratio",
                        "suspicious_length_change",
                        "contains_control_character",
                        "empty_destyled",
                        "text_basic",
                        "text_destyled",
                    ]
                ]
            )

        changed_count = int(
            dataframe["is_changed"].sum()
        )

        summary_rows.append(
            {
                "dataset": dataset_key,
                "total_records": len(dataframe),
                "changed_records": changed_count,
                "unchanged_records": (
                    len(dataframe) - changed_count
                ),
                "empty_destyled": int(
                    dataframe["empty_destyled"].sum()
                ),
                "control_character_records": int(
                    dataframe[
                        "contains_control_character"
                    ].sum()
                ),
                "suspicious_length_records": int(
                    dataframe[
                        "suspicious_length_change"
                    ].sum()
                ),
                "average_length_ratio": round(
                    dataframe["length_ratio"].mean(),
                    4,
                ),
                "url_placeholders": int(
                    dataframe[
                        "url_placeholder_count"
                    ].sum()
                ),
                "user_placeholders": int(
                    dataframe[
                        "user_placeholder_count"
                    ].sum()
                ),
                "source_placeholders": int(
                    dataframe[
                        "source_placeholder_count"
                    ].sum()
                ),
            }
        )

    examples_report = (
        pd.concat(
            all_examples,
            ignore_index=True,
        )
        if all_examples
        else pd.DataFrame()
    )

    suspicious_report = (
        pd.concat(
            suspicious_rows,
            ignore_index=True,
        )
        if suspicious_rows
        else pd.DataFrame()
    )

    summary_report = pd.DataFrame(
        summary_rows
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    examples_path = (
        OUTPUT_DIRECTORY
        / "destylization_quality_examples.csv"
    )

    suspicious_path = (
        OUTPUT_DIRECTORY
        / "destylization_suspicious_records.csv"
    )

    summary_path = (
        OUTPUT_DIRECTORY
        / "destylization_quality_summary.csv"
    )

    examples_report.to_csv(
        examples_path,
        index=False,
        encoding="utf-8-sig",
    )

    suspicious_report.to_csv(
        suspicious_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_report.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("KALİTE DENETİMİ ÖZETİ")
    print("=" * 70)

    print(
        summary_report.to_string(
            index=False
        )
    )

    print("\nToplam örnek sayısı:")
    print(len(examples_report))

    print("\nŞüpheli kayıt sayısı:")
    print(len(suspicious_report))

    print("\nOluşturulan raporlar:")
    print(f"- {examples_path}")
    print(f"- {suspicious_path}")
    print(f"- {summary_path}")

    if summary_report["empty_destyled"].sum() > 0:
        raise ValueError(
            "Temizlikten sonra boş kalan metin bulundu."
        )

    if (
        summary_report[
            "control_character_records"
        ].sum() > 0
    ):
        print(
            "\nUYARI: Kontrol karakteri içeren "
            "metinler bulundu."
        )

    print(
        "\nKalite denetimi tamamlandı. "
        "Hiçbir veri değiştirilmedi."
    )


if __name__ == "__main__":
    main()