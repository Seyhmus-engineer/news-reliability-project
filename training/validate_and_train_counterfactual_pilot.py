from pathlib import Path
from difflib import SequenceMatcher
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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

PILOT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_pilot"
    / "trnews_counterfactual_pilot_pairs.csv"
)

ARTICLE_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_input"
    / "article_baseline_test.csv"
)

MODEL_INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "model_input"
)

ANALYSIS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "pilot"
)


QUALITY_AUDIT_PATH = (
    ANALYSIS_DIRECTORY
    / "counterfactual_pilot_quality_audit.csv"
)

ACCEPTED_PAIRS_PATH = (
    MODEL_INPUT_DIRECTORY
    / "counterfactual_pilot_accepted_pairs.csv"
)

TRAIN_PATH = (
    MODEL_INPUT_DIRECTORY
    / "counterfactual_pilot_train.csv"
)

VALIDATION_PATH = (
    MODEL_INPUT_DIRECTORY
    / "counterfactual_pilot_validation.csv"
)

METRICS_PATH = (
    ANALYSIS_DIRECTORY
    / "counterfactual_pilot_metrics.csv"
)

VALIDATION_PREDICTIONS_PATH = (
    ANALYSIS_DIRECTORY
    / "counterfactual_pilot_validation_predictions.csv"
)

ARTICLE_TEST_PREDICTIONS_PATH = (
    ANALYSIS_DIRECTORY
    / "counterfactual_pilot_article_test_predictions.csv"
)

LOGISTIC_MODEL_PATH = (
    MODEL_DIRECTORY
    / "counterfactual_pilot_tfidf_logistic.joblib"
)

SVM_MODEL_PATH = (
    MODEL_DIRECTORY
    / "counterfactual_pilot_tfidf_linear_svm.joblib"
)


RANDOM_STATE = 42
VALIDATION_RATIO = 0.20
MAX_FEATURES = 100_000

MINIMUM_LENGTH_RATIO = 0.95
MINIMUM_TOKEN_SIMILARITY = 0.90
MAXIMUM_CHANGED_TOKEN_COUNT = 8


TOKEN_PATTERN = re.compile(
    r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+"
)

FACT_NORMALIZATION_PATTERN = re.compile(
    r"[^0-9a-zçğıöşüâîû]+"
)


def parse_boolean(value: object) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "evet",
    }


def normalize_fact(text: object) -> str:
    normalized = str(text).casefold()

    normalized = (
        FACT_NORMALIZATION_PATTERN
        .sub(
            " ",
            normalized,
        )
        .strip()
    )

    return re.sub(
        r"\s+",
        " ",
        normalized,
    )


def tokenize(text: object) -> list[str]:
    return [
        token.casefold()
        for token in TOKEN_PATTERN.findall(
            str(text)
        )
    ]


def calculate_token_difference(
    real_text: str,
    fake_text: str,
) -> tuple[float, int]:
    real_tokens = tokenize(real_text)
    fake_tokens = tokenize(fake_text)

    matcher = SequenceMatcher(
        None,
        real_tokens,
        fake_tokens,
        autojunk=False,
    )

    similarity = matcher.ratio()

    changed_token_count = 0

    for (
        operation,
        real_start,
        real_end,
        fake_start,
        fake_end,
    ) in matcher.get_opcodes():

        if operation == "equal":
            continue

        changed_token_count += max(
            real_end - real_start,
            fake_end - fake_start,
        )

    return similarity, changed_token_count


def calculate_length_ratio(
    first_text: str,
    second_text: str,
) -> float:
    maximum_length = max(
        len(first_text),
        len(second_text),
    )

    if maximum_length == 0:
        return 0.0

    return (
        min(
            len(first_text),
            len(second_text),
        )
        / maximum_length
    )


def audit_pilot_pairs(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str]]:
    required_columns = {
        "record_id",
        "pair_id",
        "split_group_id",
        "parent_record_id",
        "label_standard",
        "text_basic",
        "is_synthetic",
        "edit_type",
        "original_fact",
        "modified_fact",
    }

    missing_columns = (
        required_columns.difference(
            dataframe.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Pilot dosyasında eksik sütunlar: "
            f"{sorted(missing_columns)}"
        )

    audit_rows = []
    accepted_pair_ids = set()

    for pair_id, group in dataframe.groupby(
        "pair_id",
        sort=False,
    ):
        reasons = []

        if len(group) != 2:
            reasons.append(
                "pair_record_count_not_two"
            )

        real_rows = group[
            group["label_standard"].eq(
                "real"
            )
        ]

        fake_rows = group[
            group["label_standard"].eq(
                "fake"
            )
        ]

        if len(real_rows) != 1:
            reasons.append(
                "real_record_count_not_one"
            )

        if len(fake_rows) != 1:
            reasons.append(
                "fake_record_count_not_one"
            )

        if reasons:
            audit_rows.append(
                {
                    "pair_id": pair_id,
                    "quality_status": "rejected",
                    "quality_reasons": ";".join(
                        reasons
                    ),
                }
            )

            continue

        real_row = real_rows.iloc[0]
        fake_row = fake_rows.iloc[0]

        real_text = str(
            real_row["text_basic"]
        ).strip()

        fake_text = str(
            fake_row["text_basic"]
        ).strip()

        if not real_text:
            reasons.append(
                "empty_real_text"
            )

        if not fake_text:
            reasons.append(
                "empty_fake_text"
            )

        if real_text == fake_text:
            reasons.append(
                "real_and_fake_identical"
            )

        if not parse_boolean(
            fake_row["is_synthetic"]
        ):
            reasons.append(
                "fake_not_marked_synthetic"
            )

        if parse_boolean(
            real_row["is_synthetic"]
        ):
            reasons.append(
                "real_marked_synthetic"
            )

        if (
            str(fake_row["parent_record_id"])
            != str(real_row["record_id"])
        ):
            reasons.append(
                "parent_record_id_mismatch"
            )

        if (
            str(real_row["split_group_id"])
            != str(fake_row["split_group_id"])
        ):
            reasons.append(
                "split_group_mismatch"
            )

        original_fact = normalize_fact(
            fake_row["original_fact"]
        )

        modified_fact = normalize_fact(
            fake_row["modified_fact"]
        )

        normalized_real = normalize_fact(
            real_text
        )

        normalized_fake = normalize_fact(
            fake_text
        )

        if not original_fact:
            reasons.append(
                "missing_original_fact"
            )

        elif original_fact not in normalized_real:
            reasons.append(
                "original_fact_not_found_in_real"
            )

        if not modified_fact:
            reasons.append(
                "missing_modified_fact"
            )

        elif modified_fact not in normalized_fake:
            reasons.append(
                "modified_fact_not_found_in_fake"
            )

        if (
            original_fact
            and modified_fact
            and original_fact == modified_fact
        ):
            reasons.append(
                "original_and_modified_fact_same"
            )

        length_ratio = calculate_length_ratio(
            real_text,
            fake_text,
        )

        (
            token_similarity,
            changed_token_count,
        ) = calculate_token_difference(
            real_text,
            fake_text,
        )

        review_reasons = []

        if (
            length_ratio
            < MINIMUM_LENGTH_RATIO
        ):
            review_reasons.append(
                "low_length_ratio"
            )

        if (
            token_similarity
            < MINIMUM_TOKEN_SIMILARITY
        ):
            review_reasons.append(
                "low_token_similarity"
            )

        if (
            changed_token_count
            > MAXIMUM_CHANGED_TOKEN_COUNT
        ):
            review_reasons.append(
                "too_many_changed_tokens"
            )

        structural_reasons = [
            reason
            for reason in reasons
            if reason not in {
                "original_fact_not_found_in_real",
                "modified_fact_not_found_in_fake",
            }
        ]

        if structural_reasons:
            quality_status = "rejected"

        elif reasons or review_reasons:
            quality_status = (
                "review_required"
            )

        else:
            quality_status = (
                "accepted_auto"
            )

            accepted_pair_ids.add(
                pair_id
            )

        all_reasons = (
            reasons
            + review_reasons
        )

        audit_rows.append(
            {
                "pair_id": pair_id,
                "real_record_id": (
                    real_row["record_id"]
                ),
                "fake_record_id": (
                    fake_row["record_id"]
                ),
                "source": (
                    real_row.get(
                        "source",
                        "",
                    )
                ),
                "edit_type": (
                    fake_row["edit_type"]
                ),
                "original_fact": (
                    fake_row["original_fact"]
                ),
                "modified_fact": (
                    fake_row["modified_fact"]
                ),
                "length_ratio": round(
                    length_ratio,
                    5,
                ),
                "token_similarity": round(
                    token_similarity,
                    5,
                ),
                "changed_token_count": (
                    changed_token_count
                ),
                "quality_status": (
                    quality_status
                ),
                "quality_reasons": (
                    ";".join(
                        all_reasons
                    )
                ),
            }
        )

    return (
        pd.DataFrame(audit_rows),
        accepted_pair_ids,
    )


def create_group_safe_split(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    pair_ids = sorted(
        dataframe["pair_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    train_pair_ids, validation_pair_ids = (
        train_test_split(
            pair_ids,
            test_size=VALIDATION_RATIO,
            random_state=RANDOM_STATE,
        )
    )

    pair_to_split = {
        pair_id: "train"
        for pair_id in train_pair_ids
    }

    pair_to_split.update(
        {
            pair_id: "validation"
            for pair_id
            in validation_pair_ids
        }
    )

    output = dataframe.copy()

    output["final_split"] = (
        output["pair_id"]
        .map(pair_to_split)
    )

    group_split_counts = (
        output
        .groupby("pair_id")[
            "final_split"
        ]
        .nunique()
    )

    leaking_pairs = int(
        (
            group_split_counts > 1
        ).sum()
    )

    if leaking_pairs > 0:
        raise ValueError(
            f"{leaking_pairs} çift birden "
            "fazla splitte bulundu."
        )

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
        max_features=MAX_FEATURES,
        sublinear_tf=True,
    )

    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2_000,
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

    if hasattr(
        pipeline,
        "predict_proba",
    ):
        probabilities = (
            pipeline.predict_proba(
                texts
            )
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

    if classes[1] == "fake":
        return decision_scores

    return -decision_scores


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    dataframe: pd.DataFrame,
    dataset_name: str,
    split_name: str,
    text_column: str,
) -> tuple[dict, pd.DataFrame]:
    true_labels = dataframe[
        "label_standard"
    ]

    predictions = pipeline.predict(
        dataframe[text_column]
    )

    fake_scores = calculate_fake_scores(
        pipeline,
        dataframe[text_column],
    )

    binary_true = (
        true_labels.eq("fake")
        .astype(int)
    )

    matrix = confusion_matrix(
        true_labels,
        predictions,
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
        "dataset_name": dataset_name,
        "split": split_name,
        "record_count": len(
            dataframe
        ),
        "accuracy": round(
            accuracy_score(
                true_labels,
                predictions,
            ),
            4,
        ),
        "balanced_accuracy": round(
            balanced_accuracy_score(
                true_labels,
                predictions,
            ),
            4,
        ),
        "precision_fake": round(
            precision_score(
                true_labels,
                predictions,
                pos_label="fake",
                zero_division=0,
            ),
            4,
        ),
        "recall_fake": round(
            recall_score(
                true_labels,
                predictions,
                pos_label="fake",
                zero_division=0,
            ),
            4,
        ),
        "f1_fake": round(
            f1_score(
                true_labels,
                predictions,
                pos_label="fake",
                zero_division=0,
            ),
            4,
        ),
        "macro_f1": round(
            f1_score(
                true_labels,
                predictions,
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

    prediction_report = dataframe.copy()

    prediction_report[
        f"{model_name}_prediction"
    ] = predictions

    prediction_report[
        f"{model_name}_fake_score"
    ] = fake_scores

    return metrics, prediction_report


def calculate_pair_ranking_accuracy(
    prediction_report: pd.DataFrame,
    model_name: str,
) -> float:
    score_column = (
        f"{model_name}_fake_score"
    )

    correctly_ranked = 0
    total_pairs = 0

    for _, group in prediction_report.groupby(
        "pair_id"
    ):
        fake_rows = group[
            group["label_standard"].eq(
                "fake"
            )
        ]

        real_rows = group[
            group["label_standard"].eq(
                "real"
            )
        ]

        if (
            len(fake_rows) != 1
            or len(real_rows) != 1
        ):
            continue

        total_pairs += 1

        fake_score = float(
            fake_rows.iloc[0][
                score_column
            ]
        )

        real_score = float(
            real_rows.iloc[0][
                score_column
            ]
        )

        if fake_score > real_score:
            correctly_ranked += 1

    if total_pairs == 0:
        return np.nan

    return round(
        correctly_ranked
        / total_pairs,
        4,
    )


def load_article_test() -> tuple[
    pd.DataFrame,
    str,
]:
    if not ARTICLE_TEST_PATH.exists():
        raise FileNotFoundError(
            "Article Main test dosyası "
            f"bulunamadı: {ARTICLE_TEST_PATH}"
        )

    dataframe = pd.read_csv(
        ARTICLE_TEST_PATH
    )

    text_column = next(
        (
            column
            for column in [
                "text_destyled",
                "text_basic",
                "text_raw",
            ]
            if column in dataframe.columns
        ),
        None,
    )

    if text_column is None:
        raise ValueError(
            "Article Main test dosyasında "
            "metin sütunu bulunamadı."
        )

    dataframe = dataframe[
        dataframe["label_standard"].isin(
            [
                "fake",
                "real",
            ]
        )
    ].copy()

    dataframe[text_column] = (
        dataframe[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe[
        dataframe[text_column].str.len() > 0
    ].copy()

    return dataframe, text_column


def merge_prediction_reports(
    reports: list[pd.DataFrame],
    base_columns: list[str],
) -> pd.DataFrame:
    combined = reports[0][
        base_columns
        + [
            column
            for column in reports[0].columns
            if (
                column.endswith(
                    "_prediction"
                )
                or column.endswith(
                    "_fake_score"
                )
            )
        ]
    ].copy()

    for report in reports[1:]:
        prediction_columns = [
            column
            for column in report.columns
            if (
                column.endswith(
                    "_prediction"
                )
                or column.endswith(
                    "_fake_score"
                )
            )
        ]

        combined = combined.merge(
            report[
                ["record_id"]
                + prediction_columns
            ],
            on="record_id",
            how="left",
        )

    return combined


def main() -> None:
    if not PILOT_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Pilot veri bulunamadı: "
            f"{PILOT_INPUT_PATH}"
        )

    MODEL_INPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANALYSIS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Karşı-olgusal pilot kalite "
        "kontrolü başlatılıyor..."
    )

    pilot_dataframe = pd.read_csv(
        PILOT_INPUT_PATH,
        dtype=str,
        keep_default_na=False,
    )

    (
        audit_dataframe,
        accepted_pair_ids,
    ) = audit_pilot_pairs(
        pilot_dataframe
    )

    audit_dataframe.to_csv(
        QUALITY_AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    accepted_dataframe = (
        pilot_dataframe[
            pilot_dataframe["pair_id"]
            .isin(accepted_pair_ids)
        ]
        .copy()
    )

    if accepted_dataframe.empty:
        raise ValueError(
            "Otomatik kalite kontrolünden "
            "geçen çift bulunamadı."
        )

    accepted_dataframe = (
        create_group_safe_split(
            accepted_dataframe
        )
    )

    train_dataframe = (
        accepted_dataframe[
            accepted_dataframe[
                "final_split"
            ].eq("train")
        ]
        .copy()
    )

    validation_dataframe = (
        accepted_dataframe[
            accepted_dataframe[
                "final_split"
            ].eq("validation")
        ]
        .copy()
    )

    accepted_dataframe.to_csv(
        ACCEPTED_PAIRS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    train_dataframe.to_csv(
        TRAIN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    validation_dataframe.to_csv(
        VALIDATION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    article_test, article_text_column = (
        load_article_test()
    )

    print("\nKalite kontrol dağılımı:")

    print(
        audit_dataframe[
            "quality_status"
        ].value_counts()
    )

    print(
        "\nEğitim kayıtları:"
        f" {len(train_dataframe):,}"
    )

    print(
        "Validation kayıtları:"
        f" {len(validation_dataframe):,}"
    )

    model_paths = {
        "logistic_regression": (
            LOGISTIC_MODEL_PATH
        ),
        "linear_svm": (
            SVM_MODEL_PATH
        ),
    }

    metric_rows = []
    validation_reports = []
    article_reports = []

    for model_name, model_path in (
        model_paths.items()
    ):
        print("\n" + "-" * 76)
        print(f"Model: {model_name}")

        pipeline = create_pipeline(
            model_name
        )

        pipeline.fit(
            train_dataframe[
                "text_basic"
            ],
            train_dataframe[
                "label_standard"
            ],
        )

        joblib.dump(
            pipeline,
            model_path,
        )

        (
            validation_metrics,
            validation_predictions,
        ) = evaluate_model(
            model_name=model_name,
            pipeline=pipeline,
            dataframe=(
                validation_dataframe
            ),
            dataset_name=(
                "counterfactual_pilot"
            ),
            split_name="validation",
            text_column="text_basic",
        )

        pair_ranking_accuracy = (
            calculate_pair_ranking_accuracy(
                prediction_report=(
                    validation_predictions
                ),
                model_name=model_name,
            )
        )

        validation_metrics[
            "pair_ranking_accuracy"
        ] = pair_ranking_accuracy

        (
            article_metrics,
            article_predictions,
        ) = evaluate_model(
            model_name=model_name,
            pipeline=pipeline,
            dataframe=article_test,
            dataset_name=(
                "article_main"
            ),
            split_name=(
                "human_external_test"
            ),
            text_column=(
                article_text_column
            ),
        )

        article_metrics[
            "pair_ranking_accuracy"
        ] = np.nan

        metric_rows.extend(
            [
                validation_metrics,
                article_metrics,
            ]
        )

        validation_reports.append(
            validation_predictions
        )

        article_reports.append(
            article_predictions
        )

        print(
            "Pilot validation: "
            f"accuracy="
            f"{validation_metrics['accuracy']}, "
            f"macro_f1="
            f"{validation_metrics['macro_f1']}, "
            f"pair_ranking="
            f"{pair_ranking_accuracy}"
        )

        print(
            "Article Main dış test: "
            f"accuracy="
            f"{article_metrics['accuracy']}, "
            f"macro_f1="
            f"{article_metrics['macro_f1']}, "
            f"fake_recall="
            f"{article_metrics['recall_fake']}"
        )

    metrics_dataframe = pd.DataFrame(
        metric_rows
    )

    metrics_dataframe.to_csv(
        METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    validation_base_columns = [
        "record_id",
        "pair_id",
        "source",
        "label_standard",
        "edit_type",
        "text_basic",
    ]

    combined_validation = (
        merge_prediction_reports(
            reports=validation_reports,
            base_columns=(
                validation_base_columns
            ),
        )
    )

    combined_validation.to_csv(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    article_base_columns = [
        column
        for column in [
            "record_id",
            "source",
            "label_standard",
            article_text_column,
        ]
        if column in article_test.columns
    ]

    combined_article = (
        merge_prediction_reports(
            reports=article_reports,
            base_columns=(
                article_base_columns
            ),
        )
    )

    combined_article.to_csv(
        ARTICLE_TEST_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 76)
    print("PİLOT MODEL SONUÇLARI")
    print("=" * 76)

    print(
        metrics_dataframe.to_string(
            index=False
        )
    )

    print("\nOluşturulan dosyalar:")

    print(f"- {QUALITY_AUDIT_PATH}")
    print(f"- {ACCEPTED_PAIRS_PATH}")
    print(f"- {TRAIN_PATH}")
    print(f"- {VALIDATION_PATH}")
    print(f"- {METRICS_PATH}")

    print("\nKaydedilen pilot modeller:")

    print(f"- {LOGISTIC_MODEL_PATH}")
    print(f"- {SVM_MODEL_PATH}")


if __name__ == "__main__":
    main()