from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACTURK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified"
    / "facturk_unified.csv"
)

DUPLICATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "cross_dataset_exact_duplicates.csv"
)

CONFLICTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "cross_dataset_label_conflicts.csv"
)

FLAGGED_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified"
    / "facturk_unified_flagged.csv"
)

TRAINING_CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified"
    / "facturk_training_candidates.csv"
)

EXCLUSION_LIST_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "facturk_fctr_exclusion_list.csv"
)


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    file_name: str,
) -> None:
    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{file_name} dosyasında eksik sütunlar: "
            f"{sorted(missing_columns)}"
        )


def main() -> None:
    print("FACTurk eğitim dışlama listesi hazırlanıyor...")

    for file_path in [
        FACTURK_PATH,
        DUPLICATES_PATH,
        CONFLICTS_PATH,
    ]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Gerekli dosya bulunamadı: {file_path}"
            )

    facturk = pd.read_csv(FACTURK_PATH)
    duplicates = pd.read_csv(DUPLICATES_PATH)
    conflicts = pd.read_csv(CONFLICTS_PATH)

    validate_columns(
        dataframe=facturk,
        required_columns={
            "record_id",
            "dataset_name",
            "text_basic",
            "label_standard",
            "source",
            "cleaning_flags",
            "split_role",
        },
        file_name=FACTURK_PATH.name,
    )

    validate_columns(
        dataframe=duplicates,
        required_columns={
            "record_id",
            "dataset_name",
            "normalized_text",
            "label_standard",
            "source",
        },
        file_name=DUPLICATES_PATH.name,
    )

    validate_columns(
        dataframe=conflicts,
        required_columns={
            "record_id",
            "dataset_name",
            "normalized_text",
            "label_standard",
        },
        file_name=CONFLICTS_PATH.name,
    )

    print(f"\nFACTurk toplam kayıt: {len(facturk)}")

    facturk_duplicate_rows = duplicates[
        duplicates["dataset_name"].eq("facturk")
    ].copy()

    fctr_duplicate_rows = duplicates[
        duplicates["dataset_name"].eq("fctr")
    ].copy()

    if facturk_duplicate_rows.empty:
        raise ValueError(
            "FACTurk ile FCTR arasında birebir tekrar bulunamadı."
        )

    # Aynı normalize edilmiş metne ait FACTurk ve FCTR
    # kayıtlarını tek satırda eşleştir.
    exclusion_list = facturk_duplicate_rows[
        [
            "normalized_text",
            "record_id",
            "label_standard",
            "source",
            "text_basic",
        ]
    ].merge(
        fctr_duplicate_rows[
            [
                "normalized_text",
                "record_id",
                "label_standard",
                "source",
            ]
        ],
        on="normalized_text",
        how="inner",
        suffixes=("_facturk", "_fctr"),
        validate="one_to_one",
    )

    conflict_keys = set(
        conflicts["normalized_text"]
        .dropna()
        .astype(str)
    )

    exclusion_list["annotation_conflict"] = (
        exclusion_list["normalized_text"]
        .astype(str)
        .isin(conflict_keys)
    )

    exclusion_list["exclusion_reason"] = (
        "exact_duplicate_with_fctr"
    )

    exclusion_list.loc[
        exclusion_list["annotation_conflict"],
        "exclusion_reason",
    ] = (
        "exact_duplicate_with_fctr;"
        "fake_misleading_annotation_conflict"
    )

    exclusion_record_ids = set(
        exclusion_list["record_id_facturk"]
        .astype(str)
    )

    # Ana FACTurk dosyasını silmeden işaretle.
    facturk_flagged = facturk.copy()

    facturk_flagged["exclude_from_training"] = (
        facturk_flagged["record_id"]
        .astype(str)
        .isin(exclusion_record_ids)
    )

    conflict_record_ids = set(
        exclusion_list.loc[
            exclusion_list["annotation_conflict"],
            "record_id_facturk",
        ].astype(str)
    )

    facturk_flagged["annotation_conflict"] = (
        facturk_flagged["record_id"]
        .astype(str)
        .isin(conflict_record_ids)
    )

    facturk_flagged["exclusion_reason"] = ""

    facturk_flagged.loc[
        facturk_flagged["exclude_from_training"],
        "exclusion_reason",
    ] = "exact_duplicate_with_fctr"

    facturk_flagged.loc[
        facturk_flagged["annotation_conflict"],
        "exclusion_reason",
    ] = (
        "exact_duplicate_with_fctr;"
        "fake_misleading_annotation_conflict"
    )

    # FCTR ile örtüşmeyen FACTurk kayıtları eğitim adayı olur.
    training_candidates = facturk_flagged[
        ~facturk_flagged["exclude_from_training"]
    ].copy()

    training_candidates["split_role"] = (
        "training_candidate"
    )

    training_candidates["cleaning_flags"] = (
        training_candidates["cleaning_flags"]
        .fillna("")
        .astype(str)
        + ";fctr_exact_overlap_checked"
    )

    FLAGGED_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXCLUSION_LIST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    facturk_flagged.to_csv(
        FLAGGED_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    training_candidates.to_csv(
        TRAINING_CANDIDATES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    exclusion_list.to_csv(
        EXCLUSION_LIST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    excluded_count = int(
        facturk_flagged["exclude_from_training"].sum()
    )

    conflict_count = int(
        facturk_flagged["annotation_conflict"].sum()
    )

    print("\n" + "=" * 60)
    print("SONUÇ")
    print("=" * 60)

    print(f"FACTurk toplam kayıt: {len(facturk_flagged)}")
    print(f"Eğitimden dışlanan kayıt: {excluded_count}")
    print(f"Etiket uyuşmazlığı bulunan kayıt: {conflict_count}")
    print(
        "FACTurk eğitim adayı: "
        f"{len(training_candidates)}"
    )

    print("\nDışlanan kayıtlardaki etiket eşleşmeleri:")
    print(
        exclusion_list[
            [
                "label_standard_facturk",
                "label_standard_fctr",
            ]
        ]
        .value_counts()
    )

    print("\nOluşturulan dosyalar:")
    print(f"- {FLAGGED_OUTPUT_PATH}")
    print(f"- {TRAINING_CANDIDATES_PATH}")
    print(f"- {EXCLUSION_LIST_PATH}")

    expected_total = (
        len(training_candidates)
        + excluded_count
    )

    if expected_total != len(facturk_flagged):
        raise ValueError(
            "Eğitim adayı ve dışlanan kayıt sayıları "
            "FACTurk toplamıyla eşleşmiyor."
        )

    if training_candidates[
        "record_id"
    ].astype(str).isin(exclusion_record_ids).any():
        raise ValueError(
            "Dışlanan bir kayıt eğitim adaylarında kaldı."
        )

    print("\nDışlama işlemi başarıyla tamamlandı.")


if __name__ == "__main__":
    main()