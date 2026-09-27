from pathlib import Path
from collections import defaultdict

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

ARTICLE_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "quality_flagged"
    / "article_main_quality_flagged.csv"
)

NEAR_DUPLICATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "within_dataset_near_duplicates.csv"
)

EXACT_DUPLICATE_PATH = (
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


ALL_DATA_PATH = (
    MODEL_INPUT_DIRECTORY
    / "article_baseline_all.csv"
)

TRAIN_PATH = (
    MODEL_INPUT_DIRECTORY
    / "article_baseline_train.csv"
)

VALIDATION_PATH = (
    MODEL_INPUT_DIRECTORY
    / "article_baseline_validation.csv"
)

TEST_PATH = (
    MODEL_INPUT_DIRECTORY
    / "article_baseline_test.csv"
)

GROUP_CONFLICT_PATH = (
    ANALYSIS_DIRECTORY
    / "article_baseline_group_conflicts.csv"
)

SPLIT_SUMMARY_PATH = (
    ANALYSIS_DIRECTORY
    / "article_baseline_split_summary.csv"
)

METRICS_PATH = (
    ANALYSIS_DIRECTORY
    / "article_baseline_metrics.csv"
)

PREDICTIONS_PATH = (
    ANALYSIS_DIRECTORY
    / "article_baseline_test_predictions.csv"
)

LOGISTIC_MODEL_PATH = (
    MODEL_DIRECTORY
    / "article_tfidf_logistic.joblib"
)

SVM_MODEL_PATH = (
    MODEL_DIRECTORY
    / "article_tfidf_linear_svm.joblib"
)


RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

TEXT_COLUMN = "text_destyled"

TFIDF_MAX_FEATURES = 100_000


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


def load_article_data() -> pd.DataFrame:
    if not ARTICLE_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Article Main dosyası bulunamadı: "
            f"{ARTICLE_INPUT_PATH}"
        )

    dataframe = pd.read_csv(
        ARTICLE_INPUT_PATH
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
            "Article Main dosyasında eksik "
            f"sütunlar var: {sorted(missing_columns)}"
        )

    if "low_information" in dataframe.columns:
        low_information_mask = (
            parse_boolean_series(
                dataframe["low_information"]
            )
        )

        dataframe = dataframe[
            ~low_information_mask
        ].copy()

    dataframe = dataframe[
        dataframe["label_standard"].isin(
            {
                "fake",
                "real",
            }
        )
    ].copy()

    dataframe[TEXT_COLUMN] = (
        dataframe[TEXT_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe[
        dataframe[TEXT_COLUMN].str.len() > 0
    ].copy()

    duplicate_record_ids = int(
        dataframe["record_id"]
        .duplicated()
        .sum()
    )

    if duplicate_record_ids > 0:
        raise ValueError(
            f"{duplicate_record_ids} tekrarlanan "
            "record_id bulundu."
        )

    dataframe.reset_index(
        drop=True,
        inplace=True,
    )

    return dataframe


def add_exact_duplicate_groups(
    union_find: UnionFind,
    valid_record_ids: set[str],
) -> int:
    exact_report = safe_read_csv(
        EXACT_DUPLICATE_PATH
    )

    if exact_report.empty:
        return 0

    required_columns = {
        "exact_group_id",
        "analysis_dataset",
        "record_id",
    }

    if not required_columns.issubset(
        exact_report.columns
    ):
        return 0

    article_exact = exact_report[
        exact_report["analysis_dataset"]
        .eq("article_main")
    ].copy()

    union_count = 0

    for _, group in article_exact.groupby(
        "exact_group_id"
    ):
        record_ids = [
            str(record_id)
            for record_id in group["record_id"]
            if str(record_id) in valid_record_ids
        ]

        if len(record_ids) < 2:
            continue

        first_record_id = record_ids[0]

        for other_record_id in record_ids[1:]:
            union_find.union(
                first_record_id,
                other_record_id,
            )

            union_count += 1

    return union_count


def add_near_duplicate_groups(
    union_find: UnionFind,
    valid_record_ids: set[str],
) -> int:
    near_report = safe_read_csv(
        NEAR_DUPLICATE_PATH
    )

    if near_report.empty:
        return 0

    required_columns = {
        "dataset_left",
        "dataset_right",
        "record_id_left",
        "record_id_right",
    }

    if not required_columns.issubset(
        near_report.columns
    ):
        return 0

    article_pairs = near_report[
        near_report["dataset_left"]
        .eq("article_main")
        & near_report["dataset_right"]
        .eq("article_main")
    ].copy()

    union_count = 0

    for _, row in article_pairs.iterrows():
        left_record_id = str(
            row["record_id_left"]
        )

        right_record_id = str(
            row["record_id_right"]
        )

        if (
            left_record_id not in valid_record_ids
            or right_record_id not in valid_record_ids
        ):
            continue

        union_find.union(
            left_record_id,
            right_record_id,
        )

        union_count += 1

    return union_count


def assign_duplicate_groups(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    record_ids = (
        output["record_id"]
        .astype(str)
        .tolist()
    )

    valid_record_ids = set(
        record_ids
    )

    union_find = UnionFind(
        record_ids
    )

    exact_union_count = (
        add_exact_duplicate_groups(
            union_find=union_find,
            valid_record_ids=valid_record_ids,
        )
    )

    near_union_count = (
        add_near_duplicate_groups(
            union_find=union_find,
            valid_record_ids=valid_record_ids,
        )
    )

    root_to_group_number = {}
    next_group_number = 1

    group_ids = []

    for record_id in record_ids:
        root = union_find.find(
            record_id
        )

        if root not in root_to_group_number:
            root_to_group_number[root] = (
                next_group_number
            )

            next_group_number += 1

        group_number = (
            root_to_group_number[root]
        )

        group_ids.append(
            f"article_group_{group_number:05d}"
        )

    output["split_group_id"] = group_ids

    print(
        f"Exact bağlantı sayısı: "
        f"{exact_union_count}"
    )

    print(
        f"Yakın kopya bağlantı sayısı: "
        f"{near_union_count}"
    )

    print(
        "Toplam split grubu: "
        f"{output['split_group_id'].nunique()}"
    )

    return output


def remove_mixed_label_groups(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_label_counts = (
        dataframe
        .groupby("split_group_id")[
            "label_standard"
        ]
        .nunique()
    )

    conflict_group_ids = (
        group_label_counts[
            group_label_counts > 1
        ].index
    )

    conflicts = dataframe[
        dataframe["split_group_id"]
        .isin(conflict_group_ids)
    ].copy()

    clean_data = dataframe[
        ~dataframe["split_group_id"]
        .isin(conflict_group_ids)
    ].copy()

    return clean_data, conflicts


def create_group_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    group_table = (
        dataframe
        .groupby("split_group_id")
        .agg(
            group_label=(
                "label_standard",
                "first",
            ),
            group_size=(
                "record_id",
                "size",
            ),
        )
        .reset_index()
    )

    return group_table


def create_group_safe_splits(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    group_table = create_group_table(
        dataframe
    )

    train_groups, temporary_groups = (
        train_test_split(
            group_table,
            test_size=(
                VALIDATION_RATIO
                + TEST_RATIO
            ),
            random_state=RANDOM_STATE,
            stratify=group_table[
                "group_label"
            ],
        )
    )

    relative_test_ratio = (
        TEST_RATIO
        / (
            VALIDATION_RATIO
            + TEST_RATIO
        )
    )

    validation_groups, test_groups = (
        train_test_split(
            temporary_groups,
            test_size=relative_test_ratio,
            random_state=RANDOM_STATE,
            stratify=temporary_groups[
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

    for group_id in test_groups[
        "split_group_id"
    ]:
        group_to_split[group_id] = "test"

    output = dataframe.copy()

    output["final_split"] = (
        output["split_group_id"]
        .map(group_to_split)
    )

    if output["final_split"].isna().any():
        raise ValueError(
            "Bazı kayıtlar split alamadı."
        )

    return output


def validate_split_isolation(
    dataframe: pd.DataFrame,
) -> None:
    group_split_counts = (
        dataframe
        .groupby("split_group_id")[
            "final_split"
        ]
        .nunique()
    )

    leaking_groups = int(
        (
            group_split_counts > 1
        ).sum()
    )

    if leaking_groups > 0:
        raise ValueError(
            f"{leaking_groups} yakın kopya grubu "
            "birden fazla splitte bulundu."
        )

    duplicated_record_ids = int(
        dataframe["record_id"]
        .duplicated()
        .sum()
    )

    if duplicated_record_ids > 0:
        raise ValueError(
            "Split verisinde tekrarlanan "
            "record_id bulundu."
        )


def create_split_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        dataframe
        .groupby(
            [
                "final_split",
                "label_standard",
            ]
        )
        .agg(
            record_count=(
                "record_id",
                "size",
            ),
            group_count=(
                "split_group_id",
                "nunique",
            ),
            average_text_length=(
                TEXT_COLUMN,
                lambda values: round(
                    values.str.len().mean(),
                    2,
                ),
            ),
        )
        .reset_index()
    )

    return summary


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
            max_iter=2000,
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


def calculate_fake_scores(
    pipeline: Pipeline,
    texts: pd.Series,
) -> np.ndarray:
    classifier = pipeline.named_steps[
        "classifier"
    ]

    classes = list(
        classifier.classes_
    )

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(
            texts
        )

        fake_index = classes.index(
            "fake"
        )

        return probabilities[
            :,
            fake_index,
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

    # Binary decision_function pozitif değerleri
    # classes_[1] sınıfına aittir.
    if classes[1] == "fake":
        return decision_scores

    return -decision_scores


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    dataframe: pd.DataFrame,
    split_name: str,
) -> tuple[dict, pd.DataFrame]:
    split_data = dataframe[
        dataframe["final_split"].eq(
            split_name
        )
    ].copy()

    true_labels = split_data[
        "label_standard"
    ]

    predicted_labels = pipeline.predict(
        split_data[TEXT_COLUMN]
    )

    fake_scores = calculate_fake_scores(
        pipeline=pipeline,
        texts=split_data[TEXT_COLUMN],
    )

    binary_true = (
        true_labels
        .eq("fake")
        .astype(int)
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[
            "fake",
            "real",
        ],
    )

    try:
        roc_auc = roc_auc_score(
            binary_true,
            fake_scores,
        )
    except ValueError:
        roc_auc = np.nan

    metrics = {
        "model_name": model_name,
        "split": split_name,
        "record_count": len(split_data),
        "accuracy": round(
            accuracy_score(
                true_labels,
                predicted_labels,
            ),
            4,
        ),
        "precision_fake": round(
            precision_score(
                true_labels,
                predicted_labels,
                pos_label="fake",
                zero_division=0,
            ),
            4,
        ),
        "recall_fake": round(
            recall_score(
                true_labels,
                predicted_labels,
                pos_label="fake",
                zero_division=0,
            ),
            4,
        ),
        "f1_fake": round(
            f1_score(
                true_labels,
                predicted_labels,
                pos_label="fake",
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
        "roc_auc_fake": (
            round(float(roc_auc), 4)
            if not np.isnan(roc_auc)
            else np.nan
        ),
        "actual_fake_predicted_fake": int(
            matrix[0, 0]
        ),
        "actual_fake_predicted_real": int(
            matrix[0, 1]
        ),
        "actual_real_predicted_fake": int(
            matrix[1, 0]
        ),
        "actual_real_predicted_real": int(
            matrix[1, 1]
        ),
    }

    predictions = split_data[
        [
            "record_id",
            "source",
            "label_standard",
            "final_split",
            "split_group_id",
            TEXT_COLUMN,
        ]
    ].copy()

    predictions[
        f"{model_name}_prediction"
    ] = predicted_labels

    predictions[
        f"{model_name}_fake_score"
    ] = fake_scores

    return metrics, predictions


def main() -> None:
    print(
        "Article Main final baseline verisi "
        "hazırlanıyor..."
    )

    dataframe = load_article_data()

    print(
        f"\nBaşlangıç kayıt sayısı: "
        f"{len(dataframe)}"
    )

    print("\nYakın kopya grupları oluşturuluyor...")

    dataframe = assign_duplicate_groups(
        dataframe
    )

    dataframe, conflicts = (
        remove_mixed_label_groups(
            dataframe
        )
    )

    conflicts.to_csv(
        GROUP_CONFLICT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Farklı etiket içeren kopya "
        f"gruplarındaki kayıt: {len(conflicts)}"
    )

    dataframe = create_group_safe_splits(
        dataframe
    )

    validate_split_isolation(
        dataframe
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

    dataframe.to_csv(
        ALL_DATA_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    train_data = dataframe[
        dataframe["final_split"].eq(
            "train"
        )
    ].copy()

    validation_data = dataframe[
        dataframe["final_split"].eq(
            "validation"
        )
    ].copy()

    test_data = dataframe[
        dataframe["final_split"].eq(
            "test"
        )
    ].copy()

    train_data.to_csv(
        TRAIN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    validation_data.to_csv(
        VALIDATION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    test_data.to_csv(
        TEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    split_summary = create_split_summary(
        dataframe
    )

    split_summary.to_csv(
        SPLIT_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print("SPLIT ÖZETİ")
    print("=" * 72)

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
    test_prediction_reports = []

    for model_name, model_path in (
        model_definitions.items()
    ):
        print("\n" + "-" * 72)

        print(
            f"Model: {model_name}"
        )

        pipeline = create_pipeline(
            model_name
        )

        pipeline.fit(
            train_data[TEXT_COLUMN],
            train_data["label_standard"],
        )

        joblib.dump(
            pipeline,
            model_path,
        )

        for split_name in (
            "validation",
            "test",
        ):
            metrics, predictions = (
                evaluate_model(
                    model_name=model_name,
                    pipeline=pipeline,
                    dataframe=dataframe,
                    split_name=split_name,
                )
            )

            metric_rows.append(
                metrics
            )

            print(
                f"{split_name}: "
                f"accuracy={metrics['accuracy']}, "
                f"macro_f1={metrics['macro_f1']}, "
                f"fake_recall={metrics['recall_fake']}"
            )

            if split_name == "test":
                test_prediction_reports.append(
                    predictions
                )

    metrics_report = pd.DataFrame(
        metric_rows
    )

    metrics_report.to_csv(
        METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if test_prediction_reports:
        combined_predictions = (
            test_prediction_reports[0]
        )

        for additional_predictions in (
            test_prediction_reports[1:]
        ):
            prediction_columns = [
                column
                for column in (
                    additional_predictions.columns
                )
                if (
                    column.endswith(
                        "_prediction"
                    )
                    or column.endswith(
                        "_fake_score"
                    )
                )
            ]

            combined_predictions = (
                combined_predictions.merge(
                    additional_predictions[
                        ["record_id"]
                        + prediction_columns
                    ],
                    on="record_id",
                    how="left",
                )
            )

        combined_predictions.to_csv(
            PREDICTIONS_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    print("\n" + "=" * 72)
    print("MODEL SONUÇLARI")
    print("=" * 72)

    print(
        metrics_report.to_string(
            index=False
        )
    )

    print("\nOluşturulan veri dosyaları:")

    print(f"- {ALL_DATA_PATH}")
    print(f"- {TRAIN_PATH}")
    print(f"- {VALIDATION_PATH}")
    print(f"- {TEST_PATH}")

    print("\nKaydedilen modeller:")

    print(f"- {LOGISTIC_MODEL_PATH}")
    print(f"- {SVM_MODEL_PATH}")

    print("\nAnaliz raporları:")

    print(f"- {SPLIT_SUMMARY_PATH}")
    print(f"- {METRICS_PATH}")
    print(f"- {PREDICTIONS_PATH}")
    print(f"- {GROUP_CONFLICT_PATH}")

    print(
        "\nİlk baseline eğitimi başarıyla "
        "tamamlandı."
    )


if __name__ == "__main__":
    main()