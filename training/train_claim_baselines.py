from pathlib import Path
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACTURK_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "quality_flagged"
    / "facturk_training_candidates_quality_flagged.csv"
)

FCTR_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "quality_flagged"
    / "fctr_quality_flagged.csv"
)

CROSS_NEAR_DUPLICATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "cross_dataset_near_duplicates.csv"
)

WITHIN_NEAR_DUPLICATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "within_dataset_near_duplicates.csv"
)

POST_CLEANING_EXACT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "post_cleaning_exact_duplicates.csv"
)


MODEL_INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "model_input"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "baselines"
)

ANALYSIS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)


FACTURK_FLAGGED_PATH = (
    MODEL_INPUT_DIRECTORY
    / "claim_binary_facturk_flagged.csv"
)

FACTURK_TRAIN_PATH = (
    MODEL_INPUT_DIRECTORY
    / "claim_binary_facturk_train.csv"
)

FACTURK_VALIDATION_PATH = (
    MODEL_INPUT_DIRECTORY
    / "claim_binary_facturk_validation.csv"
)

FCTR_EXTERNAL_TEST_PATH = (
    MODEL_INPUT_DIRECTORY
    / "claim_binary_fctr_external_test.csv"
)

OVERLAP_EXCLUSION_PATH = (
    ANALYSIS_DIRECTORY
    / "claim_binary_facturk_fctr_overlap_exclusions.csv"
)

GROUP_CONFLICT_PATH = (
    ANALYSIS_DIRECTORY
    / "claim_binary_facturk_group_conflicts.csv"
)

PREPARATION_SUMMARY_PATH = (
    ANALYSIS_DIRECTORY
    / "claim_binary_preparation_summary.csv"
)

SPLIT_SUMMARY_PATH = (
    ANALYSIS_DIRECTORY
    / "claim_binary_split_summary.csv"
)

METRICS_PATH = (
    ANALYSIS_DIRECTORY
    / "claim_binary_metrics.csv"
)

EXTERNAL_PREDICTIONS_PATH = (
    ANALYSIS_DIRECTORY
    / "claim_binary_fctr_predictions.csv"
)

LOGISTIC_MODEL_PATH = (
    MODEL_DIRECTORY
    / "claim_tfidf_logistic.joblib"
)

SVM_MODEL_PATH = (
    MODEL_DIRECTORY
    / "claim_tfidf_linear_svm.joblib"
)


TEXT_COLUMN = "text_destyled"

RANDOM_STATE = 42
VALIDATION_RATIO = 0.15
TFIDF_MAX_FEATURES = 120_000


class UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {
            item: item
            for item in items
        }

        self.rank = {
            item: 0
            for item in items
        }

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]

    def union(
        self,
        first_item: str,
        second_item: str,
    ) -> None:
        first_root = self.find(first_item)
        second_root = self.find(second_item)

        if first_root == second_root:
            return

        if (
            self.rank[first_root]
            < self.rank[second_root]
        ):
            self.parent[first_root] = (
                second_root
            )

        elif (
            self.rank[first_root]
            > self.rank[second_root]
        ):
            self.parent[second_root] = (
                first_root
            )

        else:
            self.parent[second_root] = (
                first_root
            )

            self.rank[first_root] += 1


def safe_read_csv(
    file_path: Path,
) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            file_path
        )

    except EmptyDataError:
        return pd.DataFrame()


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


def normalize_exact_text(
    text: object,
) -> str:
    if pd.isna(text):
        return ""

    normalized = unicodedata.normalize(
        "NFKC",
        str(text),
    ).casefold()

    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )

    # Sayılar korunur. Sadece boşluk ve noktalama
    # farklılıkları azaltılır.
    normalized = re.sub(
        r"[^0-9a-zçğıöşüâîû]+",
        " ",
        normalized,
    )

    return re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()


def map_binary_label(
    label: object,
) -> str | None:
    normalized = (
        ""
        if pd.isna(label)
        else str(label).strip().lower()
    )

    if normalized in {
        "fake",
        "misleading",
    }:
        return "risky"

    if normalized == "real":
        return "real"

    return None


def load_dataset(
    file_path: Path,
    dataset_key: str,
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dosya bulunamadı: {file_path}"
        )

    dataframe = pd.read_csv(
        file_path
    )

    required_columns = {
        "record_id",
        "label_standard",
        TEXT_COLUMN,
        "source",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{file_path.name} dosyasında eksik "
            f"sütunlar var: {sorted(missing_columns)}"
        )

    output = dataframe.copy()

    output["record_id"] = (
        output["record_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    output[TEXT_COLUMN] = (
        output[TEXT_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    output["source"] = (
        output["source"]
        .fillna("unknown")
        .astype(str)
    )

    output["analysis_dataset"] = (
        dataset_key
    )

    output["binary_label"] = (
        output["label_standard"]
        .apply(map_binary_label)
    )

    if "low_information" in output.columns:
        output["low_information_parsed"] = (
            parse_boolean_series(
                output["low_information"]
            )
        )
    else:
        output["low_information_parsed"] = False

    output["exact_normalized_text"] = (
        output[TEXT_COLUMN]
        .apply(normalize_exact_text)
    )

    duplicate_record_ids = int(
        output["record_id"]
        .duplicated()
        .sum()
    )

    if duplicate_record_ids > 0:
        raise ValueError(
            f"{dataset_key} içinde "
            f"{duplicate_record_ids} tekrarlanan "
            "record_id bulundu."
        )

    return output


def get_facturk_fctr_overlap_ids() -> tuple[
    set[str],
    set[str],
    pd.DataFrame,
]:
    exact_report = safe_read_csv(
        POST_CLEANING_EXACT_PATH
    )

    near_report = safe_read_csv(
        CROSS_NEAR_DUPLICATE_PATH
    )

    exact_facturk_ids = set()
    near_facturk_ids = set()
    report_rows = []

    if (
        not exact_report.empty
        and {
            "exact_group_id",
            "analysis_dataset",
            "record_id",
        }.issubset(exact_report.columns)
    ):
        for group_id, group in exact_report.groupby(
            "exact_group_id"
        ):
            datasets = set(
                group["analysis_dataset"]
                .astype(str)
            )

            if not {
                "facturk",
                "fctr",
            }.issubset(datasets):
                continue

            facturk_rows = group[
                group["analysis_dataset"]
                .eq("facturk")
            ]

            for _, row in facturk_rows.iterrows():
                record_id = str(
                    row["record_id"]
                )

                exact_facturk_ids.add(
                    record_id
                )

                report_rows.append(
                    {
                        "record_id": record_id,
                        "overlap_type": (
                            "post_cleaning_exact"
                        ),
                        "matched_dataset": "fctr",
                        "matched_record_id": "",
                        "similarity": 1.0,
                        "label_facturk": row.get(
                            "label_standard",
                            "",
                        ),
                        "label_fctr": "",
                        "exact_group_id": group_id,
                    }
                )

    if (
        not near_report.empty
        and {
            "dataset_left",
            "dataset_right",
            "record_id_left",
            "record_id_right",
            "similarity",
        }.issubset(near_report.columns)
    ):
        for _, row in near_report.iterrows():
            left_dataset = str(
                row["dataset_left"]
            )

            right_dataset = str(
                row["dataset_right"]
            )

            if (
                left_dataset == "facturk"
                and right_dataset == "fctr"
            ):
                facturk_record_id = str(
                    row["record_id_left"]
                )

                fctr_record_id = str(
                    row["record_id_right"]
                )

                facturk_label = row.get(
                    "label_left",
                    "",
                )

                fctr_label = row.get(
                    "label_right",
                    "",
                )

            elif (
                left_dataset == "fctr"
                and right_dataset == "facturk"
            ):
                facturk_record_id = str(
                    row["record_id_right"]
                )

                fctr_record_id = str(
                    row["record_id_left"]
                )

                facturk_label = row.get(
                    "label_right",
                    "",
                )

                fctr_label = row.get(
                    "label_left",
                    "",
                )

            else:
                continue

            near_facturk_ids.add(
                facturk_record_id
            )

            report_rows.append(
                {
                    "record_id": facturk_record_id,
                    "overlap_type": (
                        "near_duplicate"
                    ),
                    "matched_dataset": "fctr",
                    "matched_record_id": (
                        fctr_record_id
                    ),
                    "similarity": row.get(
                        "similarity",
                        0.0,
                    ),
                    "label_facturk": (
                        facturk_label
                    ),
                    "label_fctr": fctr_label,
                    "exact_group_id": "",
                }
            )

    overlap_report = pd.DataFrame(
        report_rows
    )

    return (
        exact_facturk_ids,
        near_facturk_ids,
        overlap_report,
    )


def mark_internal_exact_duplicates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    output["internal_exact_rank"] = (
        output
        .sort_values("record_id")
        .groupby(
            "exact_normalized_text",
            sort=False,
        )
        .cumcount()
        + 1
    )

    output["internal_exact_group_size"] = (
        output
        .groupby(
            "exact_normalized_text",
            sort=False,
        )["record_id"]
        .transform("size")
    )

    output["internal_exact_keep"] = (
        output["internal_exact_rank"] == 1
    )

    return output


def prepare_facturk(
    dataframe: pd.DataFrame,
    exact_overlap_ids: set[str],
    near_overlap_ids: set[str],
) -> pd.DataFrame:
    output = dataframe.copy()

    output["exclude_low_information"] = (
        output["low_information_parsed"]
    )

    output["exclude_uncertain_label"] = (
        output["binary_label"].isna()
    )

    output["exclude_fctr_exact_overlap"] = (
        output["record_id"].isin(
            exact_overlap_ids
        )
    )

    output["exclude_fctr_near_overlap"] = (
        output["record_id"].isin(
            near_overlap_ids
        )
    )

    preliminary_candidate_mask = ~(
        output[
            [
                "exclude_low_information",
                "exclude_uncertain_label",
                "exclude_fctr_exact_overlap",
                "exclude_fctr_near_overlap",
            ]
        ].any(axis=1)
    )

    candidate_part = output[
        preliminary_candidate_mask
    ].copy()

    candidate_part = (
        mark_internal_exact_duplicates(
            candidate_part
        )
    )

    output["internal_exact_keep"] = True
    output["internal_exact_rank"] = 1
    output["internal_exact_group_size"] = 1

    output.loc[
        candidate_part.index,
        "internal_exact_keep",
    ] = candidate_part[
        "internal_exact_keep"
    ]

    output.loc[
        candidate_part.index,
        "internal_exact_rank",
    ] = candidate_part[
        "internal_exact_rank"
    ]

    output.loc[
        candidate_part.index,
        "internal_exact_group_size",
    ] = candidate_part[
        "internal_exact_group_size"
    ]

    output[
        "exclude_internal_exact_duplicate"
    ] = (
        preliminary_candidate_mask
        & ~output["internal_exact_keep"]
    )

    output["pre_group_candidate"] = ~(
        output[
            [
                "exclude_low_information",
                "exclude_uncertain_label",
                "exclude_fctr_exact_overlap",
                "exclude_fctr_near_overlap",
                "exclude_internal_exact_duplicate",
            ]
        ].any(axis=1)
    )

    return output


def add_within_facturk_near_groups(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    candidates = output[
        output["pre_group_candidate"]
    ].copy()

    record_ids = (
        candidates["record_id"]
        .astype(str)
        .tolist()
    )

    valid_ids = set(
        record_ids
    )

    union_find = UnionFind(
        record_ids
    )

    near_report = safe_read_csv(
        WITHIN_NEAR_DUPLICATE_PATH
    )

    union_count = 0

    if (
        not near_report.empty
        and {
            "dataset_left",
            "dataset_right",
            "record_id_left",
            "record_id_right",
        }.issubset(near_report.columns)
    ):
        facturk_pairs = near_report[
            near_report["dataset_left"]
            .eq("facturk")
            & near_report["dataset_right"]
            .eq("facturk")
        ]

        for _, row in facturk_pairs.iterrows():
            left_id = str(
                row["record_id_left"]
            )

            right_id = str(
                row["record_id_right"]
            )

            if (
                left_id not in valid_ids
                or right_id not in valid_ids
            ):
                continue

            union_find.union(
                left_id,
                right_id,
            )

            union_count += 1

    root_to_group = {}
    next_group_number = 1
    record_to_group = {}

    for record_id in record_ids:
        root = union_find.find(
            record_id
        )

        if root not in root_to_group:
            root_to_group[root] = (
                f"facturk_claim_group_"
                f"{next_group_number:05d}"
            )

            next_group_number += 1

        record_to_group[record_id] = (
            root_to_group[root]
        )

    output["split_group_id"] = ""

    output.loc[
        candidates.index,
        "split_group_id",
    ] = (
        candidates["record_id"]
        .map(record_to_group)
    )

    print(
        "FACTurk veri seti içi yakın "
        f"kopya bağlantısı: {union_count}"
    )

    print(
        "FACTurk split grup sayısı: "
        f"{len(set(record_to_group.values()))}"
    )

    return output


def remove_mixed_binary_groups(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = dataframe.copy()

    candidates = output[
        output["pre_group_candidate"]
    ].copy()

    label_counts = (
        candidates
        .groupby("split_group_id")[
            "binary_label"
        ]
        .nunique()
    )

    conflict_group_ids = set(
        label_counts[
            label_counts > 1
        ].index
    )

    output[
        "exclude_mixed_binary_group"
    ] = (
        output["split_group_id"]
        .isin(conflict_group_ids)
    )

    conflicts = output[
        output[
            "exclude_mixed_binary_group"
        ]
    ].copy()

    output["final_training_candidate"] = (
        output["pre_group_candidate"]
        & ~output[
            "exclude_mixed_binary_group"
        ]
    )

    return output, conflicts


def assign_facturk_splits(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    candidates = output[
        output["final_training_candidate"]
    ].copy()

    group_table = (
        candidates
        .groupby("split_group_id")
        .agg(
            group_label=(
                "binary_label",
                "first",
            ),
            group_size=(
                "record_id",
                "size",
            ),
        )
        .reset_index()
    )

    train_groups, validation_groups = (
        train_test_split(
            group_table,
            test_size=VALIDATION_RATIO,
            random_state=RANDOM_STATE,
            stratify=group_table[
                "group_label"
            ],
        )
    )

    group_to_split = {}

    for group_id in train_groups[
        "split_group_id"
    ]:
        group_to_split[group_id] = "train"

    for group_id in validation_groups[
        "split_group_id"
    ]:
        group_to_split[group_id] = (
            "validation"
        )

    output["final_split"] = "excluded"

    candidate_indexes = output[
        "final_training_candidate"
    ]

    output.loc[
        candidate_indexes,
        "final_split",
    ] = (
        output.loc[
            candidate_indexes,
            "split_group_id",
        ]
        .map(group_to_split)
    )

    if (
        output.loc[
            candidate_indexes,
            "final_split",
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "Bazı FACTurk kayıtları split alamadı."
        )

    group_split_counts = (
        output[
            output[
                "final_training_candidate"
            ]
        ]
        .groupby("split_group_id")[
            "final_split"
        ]
        .nunique()
    )

    leaking_group_count = int(
        (
            group_split_counts > 1
        ).sum()
    )

    if leaking_group_count > 0:
        raise ValueError(
            f"{leaking_group_count} yakın kopya "
            "grubu birden fazla splitte bulundu."
        )

    return output


def assign_exclusion_reasons(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    reason_columns = [
        (
            "exclude_low_information",
            "low_information",
        ),
        (
            "exclude_uncertain_label",
            "uncertain_label",
        ),
        (
            "exclude_fctr_exact_overlap",
            "fctr_exact_overlap",
        ),
        (
            "exclude_fctr_near_overlap",
            "fctr_near_overlap",
        ),
        (
            "exclude_internal_exact_duplicate",
            "internal_exact_duplicate",
        ),
        (
            "exclude_mixed_binary_group",
            "mixed_binary_near_duplicate_group",
        ),
    ]

    reasons = []

    for _, row in output.iterrows():
        row_reasons = []

        for column_name, reason_name in (
            reason_columns
        ):
            if bool(row.get(column_name, False)):
                row_reasons.append(
                    reason_name
                )

        reasons.append(
            ";".join(row_reasons)
        )

    output["exclusion_reasons"] = reasons

    return output


def prepare_fctr(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    output["exclude_low_information"] = (
        output["low_information_parsed"]
    )

    output["exclude_uncertain_label"] = (
        output["binary_label"].isna()
    )

    preliminary_candidate_mask = ~(
        output[
            [
                "exclude_low_information",
                "exclude_uncertain_label",
            ]
        ].any(axis=1)
    )

    candidate_part = output[
        preliminary_candidate_mask
    ].copy()

    candidate_part = (
        mark_internal_exact_duplicates(
            candidate_part
        )
    )

    output["internal_exact_keep"] = True
    output["internal_exact_rank"] = 1
    output["internal_exact_group_size"] = 1

    output.loc[
        candidate_part.index,
        "internal_exact_keep",
    ] = candidate_part[
        "internal_exact_keep"
    ]

    output.loc[
        candidate_part.index,
        "internal_exact_rank",
    ] = candidate_part[
        "internal_exact_rank"
    ]

    output.loc[
        candidate_part.index,
        "internal_exact_group_size",
    ] = candidate_part[
        "internal_exact_group_size"
    ]

    output[
        "exclude_internal_exact_duplicate"
    ] = (
        preliminary_candidate_mask
        & ~output["internal_exact_keep"]
    )

    output["external_test_candidate"] = ~(
        output[
            [
                "exclude_low_information",
                "exclude_uncertain_label",
                "exclude_internal_exact_duplicate",
            ]
        ].any(axis=1)
    )

    output["final_split"] = (
        "excluded"
    )

    output.loc[
        output["external_test_candidate"],
        "final_split",
    ] = "external_test"

    return output


def create_pipeline(
    model_name: str,
) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        strip_accents=None,
    )

    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="liblinear",
        )

    elif model_name == "linear_svm":
        classifier = LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(
            f"Bilinmeyen model: {model_name}"
        )

    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )


def calculate_risky_scores(
    pipeline: Pipeline,
    texts: pd.Series,
) -> np.ndarray:
    classifier = pipeline.named_steps[
        "classifier"
    ]

    classes = list(
        classifier.classes_
    )

    if hasattr(
        pipeline,
        "predict_proba",
    ):
        probabilities = (
            pipeline.predict_proba(
                texts
            )
        )

        risky_index = classes.index(
            "risky"
        )

        return probabilities[
            :,
            risky_index,
        ]

    decision_scores = (
        pipeline.decision_function(
            texts
        )
    )

    if len(classes) != 2:
        raise ValueError(
            "Bu kod ikili sınıflandırma "
            "için hazırlanmıştır."
        )

    if classes[1] == "risky":
        return decision_scores

    return -decision_scores


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    dataframe: pd.DataFrame,
    split_name: str,
    dataset_name: str,
) -> tuple[dict, pd.DataFrame]:
    true_labels = dataframe[
        "binary_label"
    ]

    predicted_labels = pipeline.predict(
        dataframe[TEXT_COLUMN]
    )

    risky_scores = (
        calculate_risky_scores(
            pipeline=pipeline,
            texts=dataframe[TEXT_COLUMN],
        )
    )

    binary_true = (
        true_labels
        .eq("risky")
        .astype(int)
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[
            "risky",
            "real",
        ],
    )

    try:
        roc_auc = roc_auc_score(
            binary_true,
            risky_scores,
        )

    except ValueError:
        roc_auc = np.nan

    metrics = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "split": split_name,
        "record_count": len(dataframe),
        "risky_count": int(
            true_labels.eq("risky").sum()
        ),
        "real_count": int(
            true_labels.eq("real").sum()
        ),
        "accuracy": round(
            accuracy_score(
                true_labels,
                predicted_labels,
            ),
            4,
        ),
        "precision_risky": round(
            precision_score(
                true_labels,
                predicted_labels,
                pos_label="risky",
                zero_division=0,
            ),
            4,
        ),
        "recall_risky": round(
            recall_score(
                true_labels,
                predicted_labels,
                pos_label="risky",
                zero_division=0,
            ),
            4,
        ),
        "f1_risky": round(
            f1_score(
                true_labels,
                predicted_labels,
                pos_label="risky",
                zero_division=0,
            ),
            4,
        ),
        "macro_f1": round(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
        "roc_auc_risky": (
            round(float(roc_auc), 4)
            if not np.isnan(roc_auc)
            else np.nan
        ),
        "actual_risky_predicted_risky": int(
            matrix[0, 0]
        ),
        "actual_risky_predicted_real": int(
            matrix[0, 1]
        ),
        "actual_real_predicted_risky": int(
            matrix[1, 0]
        ),
        "actual_real_predicted_real": int(
            matrix[1, 1]
        ),
    }

    predictions = dataframe[
        [
            "record_id",
            "source",
            "label_standard",
            "binary_label",
            TEXT_COLUMN,
        ]
    ].copy()

    predictions[
        f"{model_name}_prediction"
    ] = predicted_labels

    predictions[
        f"{model_name}_risky_score"
    ] = risky_scores

    return metrics, predictions


def create_preparation_summary(
    facturk: pd.DataFrame,
    fctr: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows = []

    summary_rows.append(
        {
            "dataset": "facturk",
            "original_records": len(
                facturk
            ),
            "low_information_excluded": int(
                facturk[
                    "exclude_low_information"
                ].sum()
            ),
            "uncertain_label_excluded": int(
                facturk[
                    "exclude_uncertain_label"
                ].sum()
            ),
            "fctr_exact_overlap_excluded": int(
                facturk[
                    "exclude_fctr_exact_overlap"
                ].sum()
            ),
            "fctr_near_overlap_excluded": int(
                facturk[
                    "exclude_fctr_near_overlap"
                ].sum()
            ),
            "internal_exact_excluded": int(
                facturk[
                    "exclude_internal_exact_duplicate"
                ].sum()
            ),
            "mixed_binary_group_excluded": int(
                facturk[
                    "exclude_mixed_binary_group"
                ].sum()
            ),
            "final_train_records": int(
                facturk[
                    "final_split"
                ].eq("train").sum()
            ),
            "final_validation_records": int(
                facturk[
                    "final_split"
                ].eq("validation").sum()
            ),
            "external_test_records": 0,
        }
    )

    summary_rows.append(
        {
            "dataset": "fctr",
            "original_records": len(
                fctr
            ),
            "low_information_excluded": int(
                fctr[
                    "exclude_low_information"
                ].sum()
            ),
            "uncertain_label_excluded": int(
                fctr[
                    "exclude_uncertain_label"
                ].sum()
            ),
            "fctr_exact_overlap_excluded": 0,
            "fctr_near_overlap_excluded": 0,
            "internal_exact_excluded": int(
                fctr[
                    "exclude_internal_exact_duplicate"
                ].sum()
            ),
            "mixed_binary_group_excluded": 0,
            "final_train_records": 0,
            "final_validation_records": 0,
            "external_test_records": int(
                fctr[
                    "external_test_candidate"
                ].sum()
            ),
        }
    )

    return pd.DataFrame(
        summary_rows
    )


def create_split_summary(
    facturk: pd.DataFrame,
    fctr_external: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for split_name in [
        "train",
        "validation",
    ]:
        split_data = facturk[
            facturk["final_split"]
            .eq(split_name)
        ]

        for label_name in [
            "risky",
            "real",
        ]:
            label_data = split_data[
                split_data["binary_label"]
                .eq(label_name)
            ]

            rows.append(
                {
                    "dataset": "facturk",
                    "split": split_name,
                    "binary_label": label_name,
                    "record_count": len(
                        label_data
                    ),
                    "group_count": int(
                        label_data[
                            "split_group_id"
                        ].nunique()
                    ),
                    "average_text_length": round(
                        label_data[
                            TEXT_COLUMN
                        ].str.len().mean(),
                        2,
                    ),
                }
            )

    for label_name in [
        "risky",
        "real",
    ]:
        label_data = fctr_external[
            fctr_external["binary_label"]
            .eq(label_name)
        ]

        rows.append(
            {
                "dataset": "fctr",
                "split": "external_test",
                "binary_label": label_name,
                "record_count": len(
                    label_data
                ),
                "group_count": len(
                    label_data
                ),
                "average_text_length": round(
                    label_data[
                        TEXT_COLUMN
                    ].str.len().mean(),
                    2,
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    print(
        "FACTurk-FCTR kısa iddia baseline "
        "hazırlığı başlatılıyor..."
    )

    MODEL_INPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANALYSIS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    facturk = load_dataset(
        FACTURK_INPUT_PATH,
        "facturk",
    )

    fctr = load_dataset(
        FCTR_INPUT_PATH,
        "fctr",
    )

    print(
        f"\nFACTurk başlangıç: {len(facturk)}"
    )

    print(
        f"FCTR başlangıç: {len(fctr)}"
    )

    (
        exact_overlap_ids,
        near_overlap_ids,
        overlap_report,
    ) = get_facturk_fctr_overlap_ids()

    print(
        "\nFACTurk-FCTR overlap:"
    )

    print(
        "Post-cleaning exact FACTurk kaydı: "
        f"{len(exact_overlap_ids)}"
    )

    print(
        "Yakın kopya FACTurk kaydı: "
        f"{len(near_overlap_ids)}"
    )

    facturk = prepare_facturk(
        dataframe=facturk,
        exact_overlap_ids=exact_overlap_ids,
        near_overlap_ids=near_overlap_ids,
    )

    facturk = (
        add_within_facturk_near_groups(
            facturk
        )
    )

    facturk, group_conflicts = (
        remove_mixed_binary_groups(
            facturk
        )
    )

    facturk = assign_facturk_splits(
        facturk
    )

    facturk = assign_exclusion_reasons(
        facturk
    )

    fctr = prepare_fctr(
        fctr
    )

    fctr_external = fctr[
        fctr["external_test_candidate"]
    ].copy()

    facturk_train = facturk[
        facturk["final_split"]
        .eq("train")
    ].copy()

    facturk_validation = facturk[
        facturk["final_split"]
        .eq("validation")
    ].copy()

    if facturk_train.empty:
        raise ValueError(
            "FACTurk eğitim verisi boş oluştu."
        )

    if facturk_validation.empty:
        raise ValueError(
            "FACTurk validation verisi boş oluştu."
        )

    if fctr_external.empty:
        raise ValueError(
            "FCTR dış test verisi boş oluştu."
        )

    preparation_summary = (
        create_preparation_summary(
            facturk=facturk,
            fctr=fctr,
        )
    )

    split_summary = create_split_summary(
        facturk=facturk,
        fctr_external=fctr_external,
    )

    facturk.to_csv(
        FACTURK_FLAGGED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    facturk_train.to_csv(
        FACTURK_TRAIN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    facturk_validation.to_csv(
        FACTURK_VALIDATION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    fctr_external.to_csv(
        FCTR_EXTERNAL_TEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    overlap_report.to_csv(
        OVERLAP_EXCLUSION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    group_conflicts.to_csv(
        GROUP_CONFLICT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    preparation_summary.to_csv(
        PREPARATION_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    split_summary.to_csv(
        SPLIT_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 74)
    print("VERİ HAZIRLIK ÖZETİ")
    print("=" * 74)

    print(
        preparation_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 74)
    print("SPLIT ÖZETİ")
    print("=" * 74)

    print(
        split_summary.to_string(
            index=False
        )
    )

    print("\nModeller eğitiliyor...")

    model_definitions = {
        "logistic_regression": (
            LOGISTIC_MODEL_PATH
        ),
        "linear_svm": (
            SVM_MODEL_PATH
        ),
    }

    metric_rows = []
    external_prediction_reports = []

    for model_name, model_path in (
        model_definitions.items()
    ):
        print("\n" + "-" * 74)

        print(
            f"Model: {model_name}"
        )

        pipeline = create_pipeline(
            model_name
        )

        pipeline.fit(
            facturk_train[TEXT_COLUMN],
            facturk_train["binary_label"],
        )

        joblib.dump(
            pipeline,
            model_path,
        )

        validation_metrics, _ = (
            evaluate_model(
                model_name=model_name,
                pipeline=pipeline,
                dataframe=facturk_validation,
                split_name="validation",
                dataset_name="facturk",
            )
        )

        external_metrics, predictions = (
            evaluate_model(
                model_name=model_name,
                pipeline=pipeline,
                dataframe=fctr_external,
                split_name="external_test",
                dataset_name="fctr",
            )
        )

        metric_rows.extend(
            [
                validation_metrics,
                external_metrics,
            ]
        )

        external_prediction_reports.append(
            predictions
        )

        print(
            "FACTurk validation: "
            f"accuracy="
            f"{validation_metrics['accuracy']}, "
            f"macro_f1="
            f"{validation_metrics['macro_f1']}, "
            f"risky_recall="
            f"{validation_metrics['recall_risky']}"
        )

        print(
            "FCTR external test: "
            f"accuracy="
            f"{external_metrics['accuracy']}, "
            f"macro_f1="
            f"{external_metrics['macro_f1']}, "
            f"risky_recall="
            f"{external_metrics['recall_risky']}"
        )

    metrics_report = pd.DataFrame(
        metric_rows
    )

    metrics_report.to_csv(
        METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    combined_predictions = (
        external_prediction_reports[0]
    )

    for additional_report in (
        external_prediction_reports[1:]
    ):
        prediction_columns = [
            column
            for column
            in additional_report.columns
            if (
                column.endswith(
                    "_prediction"
                )
                or column.endswith(
                    "_risky_score"
                )
            )
        ]

        combined_predictions = (
            combined_predictions.merge(
                additional_report[
                    ["record_id"]
                    + prediction_columns
                ],
                on="record_id",
                how="left",
            )
        )

    combined_predictions.to_csv(
        EXTERNAL_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 74)
    print("MODEL SONUÇLARI")
    print("=" * 74)

    print(
        metrics_report.to_string(
            index=False
        )
    )

    print("\nKaydedilen modeller:")

    print(f"- {LOGISTIC_MODEL_PATH}")
    print(f"- {SVM_MODEL_PATH}")

    print("\nModel veri dosyaları:")

    print(f"- {FACTURK_FLAGGED_PATH}")
    print(f"- {FACTURK_TRAIN_PATH}")
    print(f"- {FACTURK_VALIDATION_PATH}")
    print(f"- {FCTR_EXTERNAL_TEST_PATH}")

    print("\nAnaliz raporları:")

    print(f"- {OVERLAP_EXCLUSION_PATH}")
    print(f"- {GROUP_CONFLICT_PATH}")
    print(f"- {PREPARATION_SUMMARY_PATH}")
    print(f"- {SPLIT_SUMMARY_PATH}")
    print(f"- {METRICS_PATH}")
    print(f"- {EXTERNAL_PREDICTIONS_PATH}")

    print(
        "\nKısa iddia baseline eğitimi "
        "başarıyla tamamlandı."
    )


if __name__ == "__main__":
    main()