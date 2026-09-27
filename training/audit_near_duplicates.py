from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "quality_flagged"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)


DATASET_FILES = {
    "article_main": (
        "article_main_quality_flagged.csv"
    ),
    "facturk": (
        "facturk_training_candidates_quality_flagged.csv"
    ),
    "fctr": (
        "fctr_quality_flagged.csv"
    ),
    "mide22": (
        "mide22_quality_flagged.csv"
    ),
    "satiretr": (
        "satiretr_quality_flagged.csv"
    ),
}


# Birbiriyle karşılaştırılması anlamlı olan
# kritik veri seti çiftleri.
CROSS_DATASET_PAIRS = [
    ("facturk", "fctr"),
    ("facturk", "mide22"),
    ("fctr", "mide22"),
    ("article_main", "satiretr"),
]


SIMILARITY_THRESHOLD = 0.85
MAX_NEIGHBORS = 8
MINIMUM_LENGTH_RATIO = 0.55
MAX_FEATURES = 75000


PLACEHOLDER_PATTERN = re.compile(
    r"<(?:URL|USER|EMAIL|SOURCE|ENGAGEMENT)>",
    flags=re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"\d+(?:[.,:/-]\d+)*"
)

PUNCTUATION_PATTERN = re.compile(
    r"[^0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+"
)


def normalize_for_duplicate_analysis(
    text: object,
) -> str:
    if pd.isna(text):
        return ""

    normalized = unicodedata.normalize(
        "NFKC",
        str(text),
    ).lower()

    # Görünmeyen biçimlendirme karakterlerini kaldır.
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )

    # Kaynak, URL ve kullanıcı göstergeleri benzerliği
    # yapay biçimde yükseltmesin.
    normalized = PLACEHOLDER_PATTERN.sub(
        " ",
        normalized,
    )

    # Aynı şablonun yalnızca tarih veya sayı değişmiş
    # sürümlerini yakalamaya yardımcı olur.
    normalized = NUMBER_PATTERN.sub(
        " numtoken ",
        normalized,
    )

    normalized = PUNCTUATION_PATTERN.sub(
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def parse_boolean_series(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "evet",
            }
        )
    )


def create_text_preview(
    text: object,
    limit: int = 300,
) -> str:
    if pd.isna(text):
        return ""

    preview = re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()

    if len(preview) > limit:
        return preview[:limit] + "..."

    return preview


def calculate_length_ratio(
    first_text: str,
    second_text: str,
) -> float:
    first_length = len(first_text)
    second_length = len(second_text)

    maximum_length = max(
        first_length,
        second_length,
    )

    if maximum_length == 0:
        return 0.0

    return min(
        first_length,
        second_length,
    ) / maximum_length


def similarity_band(
    similarity: float,
) -> str:
    if similarity >= 0.95:
        return "very_high_95_plus"

    if similarity >= 0.90:
        return "very_high_90_95"

    return "high_85_90"


def load_datasets() -> dict[str, pd.DataFrame]:
    loaded = {}

    for dataset_key, file_name in DATASET_FILES.items():
        file_path = (
            INPUT_DIRECTORY
            / file_name
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dosya bulunamadı: {file_path}"
            )

        dataframe = pd.read_csv(
            file_path
        )

        required_columns = {
            "record_id",
            "dataset_name",
            "source",
            "label_standard",
            "text_destyled",
            "low_information",
        }

        missing_columns = required_columns.difference(
            dataframe.columns
        )

        if missing_columns:
            raise ValueError(
                f"{file_name} dosyasında eksik sütunlar: "
                f"{sorted(missing_columns)}"
            )

        low_information_mask = (
            parse_boolean_series(
                dataframe["low_information"]
            )
        )

        quality_candidates = dataframe[
            ~low_information_mask
        ].copy()

        quality_candidates[
            "analysis_dataset"
        ] = dataset_key

        quality_candidates[
            "comparison_text"
        ] = quality_candidates[
            "text_destyled"
        ].apply(
            normalize_for_duplicate_analysis
        )

        quality_candidates = quality_candidates[
            quality_candidates[
                "comparison_text"
            ].str.len() > 0
        ].copy()

        quality_candidates.reset_index(
            drop=True,
            inplace=True,
        )

        loaded[dataset_key] = (
            quality_candidates
        )

        print(
            f"{dataset_key}: "
            f"{len(dataframe)} toplam, "
            f"{low_information_mask.sum()} düşük bilgi dışarıda, "
            f"{len(quality_candidates)} analiz adayı"
        )

    return loaded


def create_exact_duplicate_report(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    combined = pd.concat(
        datasets.values(),
        ignore_index=True,
    )

    duplicate_mask = combined[
        "comparison_text"
    ].duplicated(
        keep=False
    )

    duplicates = combined[
        duplicate_mask
    ].copy()

    if duplicates.empty:
        return pd.DataFrame()

    duplicates = duplicates.sort_values(
        by=[
            "comparison_text",
            "analysis_dataset",
            "record_id",
        ]
    ).reset_index(drop=True)

    group_numbers = (
        duplicates
        .groupby(
            "comparison_text",
            sort=True,
        )
        .ngroup()
        + 1
    )

    duplicates["exact_group_id"] = [
        f"postclean_exact_{number:05d}"
        for number in group_numbers
    ]

    group_sizes = (
        duplicates
        .groupby("exact_group_id")[
            "record_id"
        ]
        .transform("size")
    )

    dataset_counts = (
        duplicates
        .groupby("exact_group_id")[
            "analysis_dataset"
        ]
        .transform("nunique")
    )

    label_counts = (
        duplicates
        .groupby("exact_group_id")[
            "label_standard"
        ]
        .transform("nunique")
    )

    duplicates["group_size"] = (
        group_sizes
    )

    duplicates[
        "group_dataset_count"
    ] = dataset_counts

    duplicates[
        "group_label_count"
    ] = label_counts

    duplicates[
        "has_label_conflict"
    ] = (
        duplicates["group_label_count"] > 1
    )

    duplicates["text_preview"] = (
        duplicates["text_destyled"]
        .apply(create_text_preview)
    )

    selected_columns = [
        "exact_group_id",
        "group_size",
        "group_dataset_count",
        "group_label_count",
        "has_label_conflict",
        "analysis_dataset",
        "record_id",
        "source",
        "label_standard",
        "text_preview",
        "comparison_text",
    ]

    return duplicates[
        selected_columns
    ]


def create_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        lowercase=False,
        norm="l2",
        dtype=np.float32,
    )


def create_pair_row(
    left_row: pd.Series,
    right_row: pd.Series,
    similarity: float,
    pair_scope: str,
) -> dict:
    length_ratio = calculate_length_ratio(
        left_row["comparison_text"],
        right_row["comparison_text"],
    )

    return {
        "pair_scope": pair_scope,
        "dataset_left": (
            left_row["analysis_dataset"]
        ),
        "dataset_right": (
            right_row["analysis_dataset"]
        ),
        "record_id_left": (
            left_row["record_id"]
        ),
        "record_id_right": (
            right_row["record_id"]
        ),
        "source_left": (
            left_row["source"]
        ),
        "source_right": (
            right_row["source"]
        ),
        "label_left": (
            left_row["label_standard"]
        ),
        "label_right": (
            right_row["label_standard"]
        ),
        "label_match": (
            left_row["label_standard"]
            == right_row["label_standard"]
        ),
        "similarity": round(
            similarity,
            4,
        ),
        "similarity_band": (
            similarity_band(similarity)
        ),
        "length_ratio": round(
            length_ratio,
            4,
        ),
        "text_left": create_text_preview(
            left_row["text_destyled"]
        ),
        "text_right": create_text_preview(
            right_row["text_destyled"]
        ),
    }


def find_within_dataset_pairs(
    dataset_key: str,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if len(dataframe) < 2:
        return pd.DataFrame()

    vectorizer = create_vectorizer()

    try:
        matrix = vectorizer.fit_transform(
            dataframe["comparison_text"]
        )
    except ValueError:
        return pd.DataFrame()

    neighbor_count = min(
        MAX_NEIGHBORS,
        len(dataframe),
    )

    model = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    model.fit(matrix)

    distances, indexes = model.kneighbors(
        matrix,
        return_distance=True,
    )

    pair_rows = []
    seen_pairs = set()

    for left_index in range(len(dataframe)):
        for distance, right_index in zip(
            distances[left_index],
            indexes[left_index],
        ):
            right_index = int(right_index)

            if left_index == right_index:
                continue

            pair_key = tuple(
                sorted(
                    (
                        left_index,
                        right_index,
                    )
                )
            )

            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)

            similarity = 1.0 - float(distance)

            if similarity < SIMILARITY_THRESHOLD:
                continue

            left_row = dataframe.iloc[
                left_index
            ]

            right_row = dataframe.iloc[
                right_index
            ]

            # Birebir normalize edilmiş tekrarlar ayrı
            # exact duplicate raporunda tutuluyor.
            if (
                left_row["comparison_text"]
                == right_row["comparison_text"]
            ):
                continue

            length_ratio = calculate_length_ratio(
                left_row["comparison_text"],
                right_row["comparison_text"],
            )

            if (
                length_ratio
                < MINIMUM_LENGTH_RATIO
            ):
                continue

            pair_rows.append(
                create_pair_row(
                    left_row=left_row,
                    right_row=right_row,
                    similarity=similarity,
                    pair_scope=(
                        f"within_{dataset_key}"
                    ),
                )
            )

    return pd.DataFrame(pair_rows)


def find_cross_dataset_pairs(
    left_key: str,
    right_key: str,
    left_dataframe: pd.DataFrame,
    right_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if (
        left_dataframe.empty
        or right_dataframe.empty
    ):
        return pd.DataFrame()

    combined_texts = pd.concat(
        [
            left_dataframe[
                "comparison_text"
            ],
            right_dataframe[
                "comparison_text"
            ],
        ],
        ignore_index=True,
    )

    vectorizer = create_vectorizer()

    try:
        vectorizer.fit(
            combined_texts
        )
    except ValueError:
        return pd.DataFrame()

    left_matrix = vectorizer.transform(
        left_dataframe["comparison_text"]
    )

    right_matrix = vectorizer.transform(
        right_dataframe["comparison_text"]
    )

    neighbor_count = min(
        MAX_NEIGHBORS,
        len(right_dataframe),
    )

    model = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    model.fit(
        right_matrix
    )

    distances, indexes = model.kneighbors(
        left_matrix,
        return_distance=True,
    )

    pair_rows = []
    seen_pairs = set()

    for left_index in range(
        len(left_dataframe)
    ):
        for distance, right_index in zip(
            distances[left_index],
            indexes[left_index],
        ):
            right_index = int(right_index)

            pair_key = (
                left_index,
                right_index,
            )

            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)

            similarity = 1.0 - float(distance)

            if similarity < SIMILARITY_THRESHOLD:
                continue

            left_row = left_dataframe.iloc[
                left_index
            ]

            right_row = right_dataframe.iloc[
                right_index
            ]

            if (
                left_row["comparison_text"]
                == right_row["comparison_text"]
            ):
                continue

            length_ratio = calculate_length_ratio(
                left_row["comparison_text"],
                right_row["comparison_text"],
            )

            if (
                length_ratio
                < MINIMUM_LENGTH_RATIO
            ):
                continue

            pair_rows.append(
                create_pair_row(
                    left_row=left_row,
                    right_row=right_row,
                    similarity=similarity,
                    pair_scope=(
                        f"cross_{left_key}_{right_key}"
                    ),
                )
            )

    return pd.DataFrame(pair_rows)


def create_summary(
    exact_report: pd.DataFrame,
    within_report: pd.DataFrame,
    cross_report: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows = []

    if not exact_report.empty:
        exact_groups = (
            exact_report[
                [
                    "exact_group_id",
                    "group_dataset_count",
                    "has_label_conflict",
                ]
            ]
            .drop_duplicates()
        )

        summary_rows.append(
            {
                "report_type": (
                    "post_cleaning_exact"
                ),
                "scope": "all_datasets",
                "pair_or_group_count": len(
                    exact_groups
                ),
                "record_count": len(
                    exact_report
                ),
                "label_conflict_count": int(
                    exact_groups[
                        "has_label_conflict"
                    ].sum()
                ),
            }
        )

    for report_name, report in [
        ("within_near_duplicate", within_report),
        ("cross_near_duplicate", cross_report),
    ]:
        if report.empty:
            continue

        for scope, group in report.groupby(
            "pair_scope"
        ):
            summary_rows.append(
                {
                    "report_type": report_name,
                    "scope": scope,
                    "pair_or_group_count": len(
                        group
                    ),
                    "record_count": (
                        group[
                            [
                                "record_id_left",
                                "record_id_right",
                            ]
                        ]
                        .stack()
                        .nunique()
                    ),
                    "label_conflict_count": int(
                        (~group["label_match"]).sum()
                    ),
                }
            )

    return pd.DataFrame(
        summary_rows
    )


def main() -> None:
    print(
        "Temizlik sonrası exact ve yakın kopya "
        "analizi başlatılıyor..."
    )

    print("\nVeri setleri yükleniyor:")

    datasets = load_datasets()

    print("\n" + "=" * 65)
    print("TEMİZLİK SONRASI EXACT DUPLICATE")
    print("=" * 65)

    exact_report = (
        create_exact_duplicate_report(
            datasets
        )
    )

    if exact_report.empty:
        print(
            "Normalize edilmiş birebir tekrar bulunmadı."
        )
    else:
        print(
            "Exact duplicate grup sayısı: "
            f"{exact_report['exact_group_id'].nunique()}"
        )

        print(
            "Exact duplicate kayıt sayısı: "
            f"{len(exact_report)}"
        )

        conflict_count = int(
            exact_report[
                [
                    "exact_group_id",
                    "has_label_conflict",
                ]
            ]
            .drop_duplicates()
            ["has_label_conflict"]
            .sum()
        )

        print(
            "Etiket çatışmalı grup sayısı: "
            f"{conflict_count}"
        )

    within_reports = []

    print("\n" + "=" * 65)
    print("VERİ SETİ İÇİ YAKIN KOPYALAR")
    print("=" * 65)

    for dataset_key, dataframe in datasets.items():
        print(
            f"{dataset_key} analiz ediliyor..."
        )

        report = find_within_dataset_pairs(
            dataset_key=dataset_key,
            dataframe=dataframe,
        )

        if not report.empty:
            within_reports.append(report)

        print(
            f"Bulunan aday çift: {len(report)}"
        )

    within_report = (
        pd.concat(
            within_reports,
            ignore_index=True,
        )
        if within_reports
        else pd.DataFrame()
    )

    cross_reports = []

    print("\n" + "=" * 65)
    print("VERİ SETLERİ ARASI YAKIN KOPYALAR")
    print("=" * 65)

    for left_key, right_key in CROSS_DATASET_PAIRS:
        print(
            f"{left_key} ↔ {right_key} analiz ediliyor..."
        )

        report = find_cross_dataset_pairs(
            left_key=left_key,
            right_key=right_key,
            left_dataframe=datasets[left_key],
            right_dataframe=datasets[right_key],
        )

        if not report.empty:
            cross_reports.append(report)

        print(
            f"Bulunan aday çift: {len(report)}"
        )

    cross_report = (
        pd.concat(
            cross_reports,
            ignore_index=True,
        )
        if cross_reports
        else pd.DataFrame()
    )

    summary_report = create_summary(
        exact_report=exact_report,
        within_report=within_report,
        cross_report=cross_report,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    exact_path = (
        OUTPUT_DIRECTORY
        / "post_cleaning_exact_duplicates.csv"
    )

    within_path = (
        OUTPUT_DIRECTORY
        / "within_dataset_near_duplicates.csv"
    )

    cross_path = (
        OUTPUT_DIRECTORY
        / "cross_dataset_near_duplicates.csv"
    )

    summary_path = (
        OUTPUT_DIRECTORY
        / "near_duplicate_summary.csv"
    )

    exact_report.to_csv(
        exact_path,
        index=False,
        encoding="utf-8-sig",
    )

    within_report.to_csv(
        within_path,
        index=False,
        encoding="utf-8-sig",
    )

    cross_report.to_csv(
        cross_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_report.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 65)
    print("GENEL ÖZET")
    print("=" * 65)

    if summary_report.empty:
        print(
            "Exact veya yakın kopya adayı bulunmadı."
        )
    else:
        print(
            summary_report.to_string(
                index=False
            )
        )

    print("\nOluşturulan raporlar:")
    print(f"- {exact_path}")
    print(f"- {within_path}")
    print(f"- {cross_path}")
    print(f"- {summary_path}")

    print(
        "\nHiçbir kayıt silinmedi veya değiştirilmedi."
    )


if __name__ == "__main__":
    main()