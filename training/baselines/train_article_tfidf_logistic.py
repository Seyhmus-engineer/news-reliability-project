from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


RANDOM_STATE = 42


def find_project_root() -> Path:
    """
    Dosyanın konumu:

    training/baselines/train_article_tfidf_logistic.py
    """
    return Path(__file__).resolve().parents[2]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {path}"
        )

    dataframe = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_columns = [
        "pair_id",
        "text",
        "label",
        "label_id",
        "method",
        "split",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Eksik sütunlar bulundu: "
            + ", ".join(missing_columns)
        )

    dataframe["text"] = (
        dataframe["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_count = dataframe["text"].eq("").sum()

    if empty_count:
        raise ValueError(
            f"{path.name} içinde {empty_count} boş metin var."
        )

    dataframe["label_id"] = (
        dataframe["label_id"].astype(int)
    )

    return dataframe


def evaluate_model(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    dataframe: pd.DataFrame,
    split_name: str,
    output_directory: Path,
) -> dict:
    print("\n" + "=" * 70)
    print(f"{split_name.upper()} SONUÇLARI")
    print("=" * 70)

    features = vectorizer.transform(
        dataframe["text"]
    )

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

    fake_precision = precision_score(
        true_labels,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    fake_recall = recall_score(
        true_labels,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    fake_f1 = f1_score(
        true_labels,
        predictions,
        pos_label=1,
        zero_division=0,
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

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Fake precision: {fake_precision:.4f}")
    print(f"Fake recall:    {fake_recall:.4f}")
    print(f"Fake F1:        {fake_f1:.4f}")
    print(f"ROC-AUC:        {roc_auc:.4f}")

    print("\nConfusion matrix:")
    print("Satırlar: gerçek etiket")
    print("Sütunlar: tahmin")
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

    result_dataframe = dataframe.copy()

    result_dataframe["prediction_id"] = predictions

    result_dataframe["prediction"] = (
        result_dataframe["prediction_id"]
        .map(
            {
                0: "real",
                1: "fake",
            }
        )
    )

    result_dataframe["fake_probability"] = (
        probabilities
    )

    result_dataframe["correct"] = (
        result_dataframe["label_id"]
        .eq(result_dataframe["prediction_id"])
    )

    predictions_path = (
        output_directory
        / f"{split_name}_predictions.csv"
    )

    result_dataframe.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    mistakes = result_dataframe[
        ~result_dataframe["correct"]
    ].copy()

    mistakes_path = (
        output_directory
        / f"{split_name}_mistakes.csv"
    )

    mistakes.to_csv(
        mistakes_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSentetik üretim yöntemlerine göre fake recall:")

    fake_rows = result_dataframe[
        result_dataframe["label_id"].eq(1)
    ].copy()

    method_results = []

    for method, group in fake_rows.groupby("method"):
        method_recall = recall_score(
            group["label_id"],
            group["prediction_id"],
            pos_label=1,
            zero_division=0,
        )

        method_results.append(
            {
                "method": method,
                "count": len(group),
                "fake_recall": method_recall,
            }
        )

    method_results_dataframe = pd.DataFrame(
        method_results
    ).sort_values(
        "fake_recall",
        ascending=False,
    )

    print(
        method_results_dataframe
        .to_string(index=False)
    )

    method_results_dataframe.to_csv(
        output_directory
        / f"{split_name}_method_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "split": split_name,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "fake_precision": float(fake_precision),
        "fake_recall": float(fake_recall),
        "fake_f1": float(fake_f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": matrix.tolist(),
        "mistake_count": int(len(mistakes)),
    }


def print_dataset_summary(
    name: str,
    dataframe: pd.DataFrame,
) -> None:
    print(f"\n{name}:")

    print("Kayıt:", f"{len(dataframe):,}")

    print(
        dataframe["label"]
        .value_counts()
        .to_string()
    )


def main() -> None:
    project_root = find_project_root()

    data_directory = (
        project_root
        / "data"
        / "training"
        / "article_60k"
    )

    train_path = (
        data_directory
        / "article_train_48k.csv"
    )

    validation_path = (
        data_directory
        / "article_validation_6k.csv"
    )

    test_path = (
        data_directory
        / "article_test_6k.csv"
    )

    output_directory = (
        project_root
        / "models"
        / "baselines"
        / "article_tfidf_logistic"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Veriler okunuyor...")

    train_data = load_dataset(train_path)
    validation_data = load_dataset(validation_path)
    test_data = load_dataset(test_path)

    print_dataset_summary(
        "Eğitim verisi",
        train_data,
    )

    print_dataset_summary(
        "Doğrulama verisi",
        validation_data,
    )

    print_dataset_summary(
        "Test verisi",
        test_data,
    )

    print("\nTF-IDF özellikleri hazırlanıyor...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        max_features=100_000,
        sublinear_tf=True,
        norm="l2",
        dtype="float32",
    )

    train_features = vectorizer.fit_transform(
        train_data["text"]
    )

    print(
        "Eğitim özellik matrisi:",
        train_features.shape,
    )

    print(
        "Kullanılan TF-IDF özellik sayısı:",
        len(vectorizer.vocabulary_),
    )

    print("\nLogistic Regression eğitiliyor...")

    model = LogisticRegression(
        C=2.0,
        solver="liblinear",
        max_iter=1_000,
        random_state=RANDOM_STATE,
    )

    model.fit(
        train_features,
        train_data["label_id"],
    )

    print("Model eğitimi tamamlandı.")

    validation_results = evaluate_model(
        model=model,
        vectorizer=vectorizer,
        dataframe=validation_data,
        split_name="validation",
        output_directory=output_directory,
    )

    test_results = evaluate_model(
        model=model,
        vectorizer=vectorizer,
        dataframe=test_data,
        split_name="test",
        output_directory=output_directory,
    )

    joblib.dump(
        vectorizer,
        output_directory
        / "tfidf_vectorizer.joblib",
    )

    joblib.dump(
        model,
        output_directory
        / "logistic_regression.joblib",
    )

    metrics = {
        "model_name": "TF-IDF Logistic Regression",
        "train_size": len(train_data),
        "validation_size": len(validation_data),
        "test_size": len(test_data),
        "tfidf_features": len(
            vectorizer.vocabulary_
        ),
        "validation": validation_results,
        "test": test_results,
    }

    with (
        output_directory
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("EĞİTİM TAMAMLANDI")
    print("=" * 70)

    print("Model ve sonuçlar kaydedildi:")

    for filename in [
        "tfidf_vectorizer.joblib",
        "logistic_regression.joblib",
        "metrics.json",
        "validation_predictions.csv",
        "validation_mistakes.csv",
        "validation_method_results.csv",
        "test_predictions.csv",
        "test_mistakes.csv",
        "test_method_results.csv",
    ]:
        print(
            "-",
            output_directory / filename,
        )


if __name__ == "__main__":
    main()