from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kemik_42k_standardized.csv"
)

PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

ANALYSIS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)


# Bütün kayıtların seçim kararlarını içeren ana çıktı.
FLAGGED_OUTPUT_PATH = (
    PROCESSED_DIRECTORY
    / "kemik_42k_final_flagged.csv"
)

# Exact tekrarları tekilleştirilmiş bütün kayıtlar.
DEDUPLICATED_OUTPUT_PATH = (
    PROCESSED_DIRECTORY
    / "kemik_42k_deduplicated.csv"
)

# Model eğitiminde kullanılacak dengeli gerçek haber havuzu.
BALANCED_OUTPUT_PATH = (
    PROCESSED_DIRECTORY
    / "kemik_42k_balanced_real_pool.csv"
)

# Analiz ve rapor dosyaları.
GLOBAL_SUMMARY_PATH = (
    ANALYSIS_DIRECTORY
    / "kemik_42k_final_preparation_summary.csv"
)

CATEGORY_SUMMARY_PATH = (
    ANALYSIS_DIRECTORY
    / "kemik_42k_balanced_category_summary.csv"
)

EXCLUDED_RECORDS_PATH = (
    ANALYSIS_DIRECTORY
    / "kemik_42k_excluded_records.csv"
)


# Bir kategorinin eğitim havuzuna verebileceği
# en fazla kayıt sayısı.
MAX_RECORDS_PER_CATEGORY = 1000

# Aynı kod her çalıştığında aynı örneklerin seçilmesini sağlar.
RANDOM_STATE = 42

# review_required durumundaki kayıtlar final eğitim
# havuzuna alınmayacak fakat dosyalardan silinmeyecek.
EXCLUDE_REVIEW_REQUIRED = True


def validate_columns(dataframe: pd.DataFrame) -> None:
    required_columns = {
        "record_id",
        "dataset_name",
        "category_original",
        "title_raw",
        "text_basic",
        "text_destyled",
        "label_standard",
        "source",
        "quality_status",
        "word_count",
        "exact_text_hash",
        "exact_duplicate_group_size",
        "is_exact_duplicate",
        "duplicate_group_id",
        "split_role",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Girdi dosyasında eksik sütunlar var: "
            f"{sorted(missing_columns)}"
        )


def prepare_basic_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    output["record_id"] = (
        output["record_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    output["category_original"] = (
        output["category_original"]
        .fillna("unknown_category")
        .astype(str)
        .str.strip()
    )

    output["source"] = (
        output["source"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )

    output["quality_status"] = (
        output["quality_status"]
        .fillna("review_required")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    output["exact_text_hash"] = (
        output["exact_text_hash"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    output["word_count"] = (
        pd.to_numeric(
            output["word_count"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    empty_hash_count = int(
        output["exact_text_hash"]
        .eq("")
        .sum()
    )

    if empty_hash_count > 0:
        raise ValueError(
            f"{empty_hash_count} kayıtta "
            "exact_text_hash boş."
        )

    duplicate_record_id_count = int(
        output["record_id"]
        .duplicated()
        .sum()
    )

    if duplicate_record_id_count > 0:
        raise ValueError(
            f"{duplicate_record_id_count} adet "
            "tekrarlanan record_id bulundu."
        )

    return output


def choose_exact_duplicate_representatives(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    # Aynı haber farklı kategorilerde bulunuyorsa,
    # daha az veriye sahip kategoriyi korumaya öncelik verir.
    category_sizes = (
        output["category_original"]
        .value_counts()
    )

    output["_category_size_priority"] = (
        output["category_original"]
        .map(category_sizes)
        .fillna(len(output))
        .astype(int)
    )

    # quality_candidate kayıtları önce gelir.
    output["_quality_priority"] = (
        output["quality_status"]
        .ne("quality_candidate")
        .astype(int)
    )

    # Kaynağı bilinen kayıtlar unknown kayıtlardan önce gelir.
    output["_source_priority"] = (
        output["source"]
        .str.lower()
        .eq("unknown")
        .astype(int)
    )

    # Her exact hash kümesinde temsilci seçimi:
    # 1. Kaliteli kayıt
    # 2. Daha az temsil edilen kategori
    # 3. Kaynağı bilinen kayıt
    # 4. Daha uzun metin
    # 5. record_id sırası
    output = output.sort_values(
        by=[
            "exact_text_hash",
            "_quality_priority",
            "_category_size_priority",
            "_source_priority",
            "word_count",
            "record_id",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            False,
            True,
        ],
    ).copy()

    output["dedup_rank"] = (
        output
        .groupby(
            "exact_text_hash",
            sort=False,
        )
        .cumcount()
        + 1
    )

    output["dedup_group_size"] = (
        output
        .groupby(
            "exact_text_hash",
            sort=False,
        )["record_id"]
        .transform("size")
    )

    output["dedup_keep"] = (
        output["dedup_rank"] == 1
    )

    output["deduplication_action"] = (
        "keep_unique"
    )

    representative_mask = (
        output["dedup_keep"]
        & output["dedup_group_size"].gt(1)
    )

    excluded_duplicate_mask = (
        ~output["dedup_keep"]
    )

    output.loc[
        representative_mask,
        "deduplication_action",
    ] = "keep_exact_group_representative"

    output.loc[
        excluded_duplicate_mask,
        "deduplication_action",
    ] = "exclude_exact_duplicate"

    return output


def mark_quality_eligibility(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    output["low_information_excluded"] = False

    if EXCLUDE_REVIEW_REQUIRED:
        output["low_information_excluded"] = (
            output["quality_status"]
            .ne("quality_candidate")
        )

    output["eligible_for_balancing"] = (
        output["dedup_keep"]
        & ~output["low_information_excluded"]
    )

    return output


def select_balanced_records(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    output["selected_for_balanced_pool"] = False
    output["balanced_selection_order"] = pd.NA

    eligible = output[
        output["eligible_for_balancing"]
    ].copy()

    categories = sorted(
        eligible["category_original"]
        .unique()
    )

    selection_order = 1

    for category_index, category in enumerate(
        categories
    ):
        category_data = eligible[
            eligible["category_original"].eq(
                category
            )
        ].copy()

        # Örnekleme öncesinde sıra sabitlenir.
        category_data = (
            category_data
            .sort_values("record_id")
        )

        if (
            len(category_data)
            <= MAX_RECORDS_PER_CATEGORY
        ):
            selected_indexes = (
                category_data.index
            )
        else:
            selected_indexes = (
                category_data.sample(
                    n=MAX_RECORDS_PER_CATEGORY,
                    random_state=(
                        RANDOM_STATE
                        + category_index
                    ),
                ).index
            )

        output.loc[
            selected_indexes,
            "selected_for_balanced_pool",
        ] = True

        selected_indexes_sorted = sorted(
            selected_indexes
        )

        for row_index in selected_indexes_sorted:
            output.loc[
                row_index,
                "balanced_selection_order",
            ] = selection_order

            selection_order += 1

    return output


def assign_final_actions(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    output["final_selection_action"] = (
        "not_evaluated"
    )

    output["final_exclusion_reason"] = ""

    exact_duplicate_mask = (
        ~output["dedup_keep"]
    )

    low_information_mask = (
        output["dedup_keep"]
        & output["low_information_excluded"]
    )

    category_cap_mask = (
        output["eligible_for_balancing"]
        & ~output["selected_for_balanced_pool"]
    )

    selected_mask = (
        output["selected_for_balanced_pool"]
    )

    output.loc[
        exact_duplicate_mask,
        "final_selection_action",
    ] = "excluded"

    output.loc[
        exact_duplicate_mask,
        "final_exclusion_reason",
    ] = "exact_duplicate_non_representative"

    output.loc[
        low_information_mask,
        "final_selection_action",
    ] = "excluded"

    output.loc[
        low_information_mask,
        "final_exclusion_reason",
    ] = "review_required_low_information"

    output.loc[
        category_cap_mask,
        "final_selection_action",
    ] = "excluded"

    output.loc[
        category_cap_mask,
        "final_exclusion_reason",
    ] = "category_balance_cap"

    output.loc[
        selected_mask,
        "final_selection_action",
    ] = "selected"

    output.loc[
        selected_mask,
        "final_exclusion_reason",
    ] = ""

    output["final_split_role"] = (
        "analysis_only"
    )

    output.loc[
        selected_mask,
        "final_split_role",
    ] = "real_training_candidate_balanced"

    return output


def create_category_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows = []

    for category, group in dataframe.groupby(
        "category_original",
        sort=True,
    ):
        deduplicated_mask = (
            group["dedup_keep"]
        )

        quality_candidate_mask = (
            group["eligible_for_balancing"]
        )

        selected_mask = (
            group["selected_for_balanced_pool"]
        )

        duplicate_excluded_mask = (
            group["final_exclusion_reason"]
            .eq(
                "exact_duplicate_non_representative"
            )
        )

        low_information_mask = (
            group["final_exclusion_reason"]
            .eq(
                "review_required_low_information"
            )
        )

        category_cap_mask = (
            group["final_exclusion_reason"]
            .eq("category_balance_cap")
        )

        selected_group = group[
            selected_mask
        ]

        summary_rows.append(
            {
                "category_original": category,
                "original_records": len(group),
                "deduplicated_records": int(
                    deduplicated_mask.sum()
                ),
                "quality_candidate_records": int(
                    quality_candidate_mask.sum()
                ),
                "selected_records": int(
                    selected_mask.sum()
                ),
                "excluded_exact_duplicates": int(
                    duplicate_excluded_mask.sum()
                ),
                "excluded_low_information": int(
                    low_information_mask.sum()
                ),
                "excluded_category_balance": int(
                    category_cap_mask.sum()
                ),
                "selected_average_word_count": (
                    round(
                        selected_group[
                            "word_count"
                        ].mean(),
                        2,
                    )
                    if not selected_group.empty
                    else 0.0
                ),
                "selected_known_source_records": int(
                    (
                        selected_group["source"]
                        .str.lower()
                        .ne("unknown")
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


def create_global_summary(
    dataframe: pd.DataFrame,
    category_summary: pd.DataFrame,
) -> pd.DataFrame:
    original_count = len(dataframe)

    deduplicated_count = int(
        dataframe["dedup_keep"].sum()
    )

    exact_duplicate_excluded = int(
        (
            dataframe["final_exclusion_reason"]
            == "exact_duplicate_non_representative"
        ).sum()
    )

    low_information_excluded = int(
        (
            dataframe["final_exclusion_reason"]
            == "review_required_low_information"
        ).sum()
    )

    category_balance_excluded = int(
        (
            dataframe["final_exclusion_reason"]
            == "category_balance_cap"
        ).sum()
    )

    selected_count = int(
        dataframe[
            "selected_for_balanced_pool"
        ].sum()
    )

    selected_data = dataframe[
        dataframe["selected_for_balanced_pool"]
    ]

    duplicate_hashes_in_final = int(
        selected_data[
            "exact_text_hash"
        ].duplicated().sum()
    )

    summary = {
        "original_records": original_count,
        "deduplicated_records": deduplicated_count,
        "excluded_exact_duplicates": (
            exact_duplicate_excluded
        ),
        "excluded_low_information": (
            low_information_excluded
        ),
        "excluded_category_balance": (
            category_balance_excluded
        ),
        "balanced_pool_records": selected_count,
        "balanced_pool_categories": int(
            selected_data[
                "category_original"
            ].nunique()
        ),
        "maximum_records_per_category": (
            MAX_RECORDS_PER_CATEGORY
        ),
        "balanced_pool_duplicate_hashes": (
            duplicate_hashes_in_final
        ),
        "balanced_pool_average_word_count": (
            round(
                selected_data[
                    "word_count"
                ].mean(),
                2,
            )
        ),
        "balanced_pool_minimum_word_count": (
            int(
                selected_data[
                    "word_count"
                ].min()
            )
        ),
        "balanced_pool_maximum_word_count": (
            int(
                selected_data[
                    "word_count"
                ].max()
            )
        ),
        "smallest_category_selected_count": (
            int(
                category_summary[
                    "selected_records"
                ].min()
            )
        ),
        "largest_category_selected_count": (
            int(
                category_summary[
                    "selected_records"
                ].max()
            )
        ),
    }

    return pd.DataFrame(
        [summary]
    )


def validate_final_outputs(
    dataframe: pd.DataFrame,
) -> None:
    selected = dataframe[
        dataframe["selected_for_balanced_pool"]
    ]

    if selected.empty:
        raise ValueError(
            "Dengeli eğitim havuzu boş oluştu."
        )

    duplicate_hash_count = int(
        selected["exact_text_hash"]
        .duplicated()
        .sum()
    )

    if duplicate_hash_count > 0:
        raise ValueError(
            "Final eğitim havuzunda "
            f"{duplicate_hash_count} exact tekrar var."
        )

    review_required_count = int(
        selected["quality_status"]
        .ne("quality_candidate")
        .sum()
    )

    if review_required_count > 0:
        raise ValueError(
            "Final eğitim havuzunda "
            f"{review_required_count} inceleme "
            "gerektiren kayıt var."
        )

    maximum_category_count = int(
        selected["category_original"]
        .value_counts()
        .max()
    )

    if (
        maximum_category_count
        > MAX_RECORDS_PER_CATEGORY
    ):
        raise ValueError(
            "Bir kategori belirlenen üst sınırı geçti."
        )

    invalid_label_count = int(
        selected["label_standard"]
        .ne("real")
        .sum()
    )

    if invalid_label_count > 0:
        raise ValueError(
            "Final gerçek haber havuzunda "
            "real dışında etiket bulundu."
        )


def main() -> None:
    print(
        "42 Bin Haber final eğitim havuzu "
        "hazırlanıyor..."
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Girdi dosyası bulunamadı: "
            f"{INPUT_PATH}"
        )

    dataframe = pd.read_csv(
        INPUT_PATH
    )

    validate_columns(
        dataframe
    )

    print(
        f"\nBaşlangıç kayıt sayısı: "
        f"{len(dataframe)}"
    )

    dataframe = prepare_basic_columns(
        dataframe
    )

    print(
        "Exact tekrar temsilcileri seçiliyor..."
    )

    dataframe = (
        choose_exact_duplicate_representatives(
            dataframe
        )
    )

    print(
        "Düşük bilgi kayıtları "
        "işaretleniyor..."
    )

    dataframe = mark_quality_eligibility(
        dataframe
    )

    print(
        "Kategoriler dengeleniyor..."
    )

    dataframe = select_balanced_records(
        dataframe
    )

    dataframe = assign_final_actions(
        dataframe
    )

    category_summary = (
        create_category_summary(
            dataframe
        )
    )

    global_summary = (
        create_global_summary(
            dataframe=dataframe,
            category_summary=category_summary,
        )
    )

    validate_final_outputs(
        dataframe
    )

    # Geçici öncelik sütunları final çıktıya alınmaz.
    temporary_columns = [
        "_category_size_priority",
        "_quality_priority",
        "_source_priority",
    ]

    dataframe.drop(
        columns=[
            column
            for column in temporary_columns
            if column in dataframe.columns
        ],
        inplace=True,
    )

    # Bütün seçim kararlarını içeren dosya.
    flagged_output = (
        dataframe
        .sort_values("record_id")
        .reset_index(drop=True)
    )

    # Exact tekrarları tekilleştirilmiş veri.
    deduplicated_output = (
        dataframe[
            dataframe["dedup_keep"]
        ]
        .sort_values(
            [
                "category_original",
                "record_id",
            ]
        )
        .reset_index(drop=True)
    )

    # Model eğitiminde kullanılacak dengeli havuz.
    balanced_output = (
        dataframe[
            dataframe[
                "selected_for_balanced_pool"
            ]
        ]
        .sort_values(
            [
                "category_original",
                "balanced_selection_order",
            ]
        )
        .reset_index(drop=True)
    )

    excluded_output = (
        dataframe[
            ~dataframe[
                "selected_for_balanced_pool"
            ]
        ][
            [
                "record_id",
                "category_original",
                "source",
                "quality_status",
                "word_count",
                "dedup_group_size",
                "dedup_rank",
                "deduplication_action",
                "final_exclusion_reason",
                "duplicate_group_id",
                "title_raw",
                "text_basic",
            ]
        ]
        .sort_values(
            [
                "final_exclusion_reason",
                "category_original",
                "record_id",
            ]
        )
        .reset_index(drop=True)
    )

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANALYSIS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    flagged_output.to_csv(
        FLAGGED_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    deduplicated_output.to_csv(
        DEDUPLICATED_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    balanced_output.to_csv(
        BALANCED_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    global_summary.to_csv(
        GLOBAL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    category_summary.to_csv(
        CATEGORY_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    excluded_output.to_csv(
        EXCLUDED_RECORDS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print("GENEL SONUÇ")
    print("=" * 72)

    print(
        global_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 72)
    print("KATEGORİ DAĞILIMI")
    print("=" * 72)

    print(
        category_summary[
            [
                "category_original",
                "original_records",
                "deduplicated_records",
                "quality_candidate_records",
                "selected_records",
                "excluded_exact_duplicates",
                "excluded_low_information",
                "excluded_category_balance",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nOluşturulan dosyalar:")

    print(
        f"- {FLAGGED_OUTPUT_PATH}"
    )

    print(
        f"- {DEDUPLICATED_OUTPUT_PATH}"
    )

    print(
        f"- {BALANCED_OUTPUT_PATH}"
    )

    print(
        f"- {GLOBAL_SUMMARY_PATH}"
    )

    print(
        f"- {CATEGORY_SUMMARY_PATH}"
    )

    print(
        f"- {EXCLUDED_RECORDS_PATH}"
    )

    print(
        "\nOrijinal standartlaştırılmış dosya "
        "değiştirilmedi."
    )

    print(
        "\nDengeli gerçek haber eğitim havuzu "
        "başarıyla oluşturuldu."
    )


if __name__ == "__main__":
    main()