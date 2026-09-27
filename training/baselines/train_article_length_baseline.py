from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def find_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_data(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    # Uzunluk eşitlenmiş dosyalarda yeni uzunluğu kullan
    if "matched_word_count" in dataframe.columns:
        dataframe["word_count"] = dataframe[
            "matched_word_count"
        ]

    required_columns = [
        "word_count",
        "label_id",
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

    dataframe["word_count"] = pd.to_numeric(
        dataframe["word_count"],
        errors="coerce",
    )

    dataframe["label_id"] = pd.to_numeric(
        dataframe["label_id"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            "word_count",
            "label_id",
        ]
    ).copy()

    dataframe["label_id"] = (
        dataframe["label_id"].astype(int)
    )

    return dataframe

def evaluate(
    model: LogisticRegression,
    dataframe: pd.DataFrame,
    split_name: str,
) -> None:
    features = dataframe[
        ["word_count"]
    ]

    true_labels = dataframe["label_id"]

    predictions = model.predict(features)

    probabilities = model.predict_proba(
        features
    )[:, 1]

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
    )

    roc_auc = roc_auc_score(
        true_labels,
        probabilities,
    )

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1],
    )

    print("\n" + "=" * 60)
    print(f"{split_name.upper()} UZUNLUK BASELINE")
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")

    print("\nConfusion matrix:")
    print(matrix)

    print("\nSınıflandırma raporu:")

    print(
        classification_report(
            true_labels,
            predictions,
            labels=[0, 1],
            target_names=["real", "fake"],
            digits=4,
            zero_division=0,
        )
    )


def main() -> None:
    project_root = find_project_root()

    data_directory = (
            project_root
            / "data"
            / "training"
            / "article_60k_length_matched"
    )

    train_data = load_data(
        data_directory
        / "article_length_matched_train.csv"
    )

    validation_data = load_data(
        data_directory
        / "article_length_matched_validation.csv"
    )

    test_data = load_data(
        data_directory
        / "article_length_matched_test.csv"
    )

    print("Eğitim uzunluk özeti:")

    print(
        train_data.groupby("label_id")[
            "word_count"
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

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        train_data[["word_count"]],
        train_data["label_id"],
    )

    print("\nModel katsayısı:", model.coef_[0][0])
    print("Model sabiti:", model.intercept_[0])

    evaluate(
        model,
        validation_data,
        "validation",
    )

    evaluate(
        model,
        test_data,
        "test",
    )


if __name__ == "__main__":
    main()