from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC


RANDOM_STATE = 42


def find_project_root() -> Path:
    """
    Bu dosyanın şu konumda olduğu varsayılır:

    training/baselines/train_article_tfidf_svm_length_matched.py
    """
    return Path(__file__).resolve().parents[2]


def load_dataset(path: Path) -> pd.DataFrame:
    """
    CSV dosyasını okur ve eğitim için gerekli kontrolleri yapar.
    """
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
        "matched_word_count",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path.name} dosyasında eksik sütunlar bulundu: "
            + ", ".join(missing_columns)
        )

    dataframe["text"] = (
        dataframe["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_text_count = dataframe["text"].eq("").sum()

    if empty_text_count:
        raise ValueError(
            f"{path.name} dosyasında "
            f"{empty_text_count} boş metin bulundu."
        )

    dataframe["label_id"] = pd.to_numeric(
        dataframe["label_id"],
        errors="coerce",
    )

    dataframe["matched_word_count"] = pd.to_numeric(
        dataframe["matched_word_count"],
        errors="coerce",
    )

    invalid_label_count = (
        dataframe["label_id"].isna().sum()
    )

    invalid_length_count = (
        dataframe["matched_word_count"].isna().sum()
    )

    if invalid_label_count:
        raise ValueError(
            f"{path.name} dosyasında "
            f"{invalid_label_count} geçersiz label_id bulundu."
        )

    if invalid_length_count:
        raise ValueError(
            f"{path.name} dosyasında "
            f"{invalid_length_count} geçersiz matched_word_count bulundu."
        )

    dataframe["label_id"] = (
        dataframe["label_id"].astype(int)
    )

    valid_labels = set(
        dataframe["label_id"].unique()
    )

    if not valid_labels.issubset({0, 1}):
        raise ValueError(
            f"Beklenmeyen label_id değerleri: {valid_labels}"
        )

    return dataframe


def print_dataset_summary(
    name: str,
    dataframe: pd.DataFrame,
) -> None:
    """
    Veri setinin temel bilgilerini ekrana yazdırır.
    """
    print("\n" + "=" * 65)
    print(name.upper())
    print("=" * 65)

    print("Kayıt sayısı:", f"{len(dataframe):,}")

    print(
        "Benzersiz pair_id:",
        f"{dataframe['pair_id'].nunique():,}",
    )

    print("\nEtiket dağılımı:")

    print(
        dataframe["label"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nKelime uzunluğu özeti:")

    print(
        dataframe.groupby("label")[
            "matched_word_count"
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

    fake_data = dataframe[
        dataframe["label_id"].eq(1)
    ]

    print("\nSentetik yöntem dağılımı:")

    print(
        fake_data["method"]
        .value_counts(dropna=False)
        .to_string()
    )


def check_split_leakage(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """
    Aynı pair_id değerinin farklı splitlerde bulunup bulunmadığını kontrol eder.
    """
    train_ids = set(train_data["pair_id"])
    validation_ids = set(validation_data["pair_id"])
    test_ids = set(test_data["pair_id"])

    train_validation_overlap = (
        train_ids & validation_ids
    )

    train_test_overlap = (
        train_ids & test_ids
    )

    validation_test_overlap = (
        validation_ids & test_ids
    )

    print("\n" + "=" * 65)
    print("SPLIT SIZINTISI KONTROLÜ")
    print("=" * 65)

    print(
        "Train-validation ortak pair_id:",
        len(train_validation_overlap),
    )

    print(
        "Train-test ortak pair_id:",
        len(train_test_overlap),
    )

    print(
        "Validation-test ortak pair_id:",
        len(validation_test_overlap),
    )

    if (
        train_validation_overlap
        or train_test_overlap
        or validation_test_overlap
    ):
        raise RuntimeError(
            "Train, validation ve test arasında "
            "pair_id sızıntısı bulundu."
        )

    print("Split sızıntısı bulunmadı.")


def evaluate_by_method(
    result_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sahte haberleri üretim yöntemlerine göre ayrı ayrı değerlendirir.
    """
    fake_rows = result_dataframe[
        result_dataframe["label_id"].eq(1)
    ].copy()

    method_results = []

    for method, group in fake_rows.groupby(
        "method"
    ):
        fake_recall = recall_score(
            group["label_id"],
            group["prediction_id"],
            pos_label=1,
            zero_division=0,
        )

        correct_count = int(
            group["correct"].sum()
        )

        mistake_count = int(
            (~group["correct"]).sum()
        )

        method_results.append(
            {
                "method": method,
                "count": len(group),
                "correct_count": correct_count,
                "mistake_count": mistake_count,
                "fake_recall": fake_recall,
            }
        )

    method_dataframe = pd.DataFrame(
        method_results
    )

    if not method_dataframe.empty:
        method_dataframe = (
            method_dataframe.sort_values(
                "fake_recall",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    return method_dataframe


def evaluate_model(
    model: LinearSVC,
    vectorizer: TfidfVectorizer,
    dataframe: pd.DataFrame,
    split_name: str,
    output_directory: Path,
) -> dict:
    """
    Modeli verilen split üzerinde değerlendirir ve sonuçları kaydeder.
    """
    print("\n" + "=" * 70)
    print(f"{split_name.upper()} SONUÇLARI")
    print("=" * 70)

    features = vectorizer.transform(
        dataframe["text"]
    )

    true_labels = dataframe["label_id"]

    predictions = model.predict(
        features
    )

    decision_scores = model.decision_function(
        features
    )

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
    )

    weighted_f1 = f1_score(
        true_labels,
        predictions,
        average="weighted",
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

    real_precision = precision_score(
        true_labels,
        predictions,
        pos_label=0,
        zero_division=0,
    )

    real_recall = recall_score(
        true_labels,
        predictions,
        pos_label=0,
        zero_division=0,
    )

    real_f1 = f1_score(
        true_labels,
        predictions,
        pos_label=0,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        true_labels,
        decision_scores,
    )

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1],
    )

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")
    print(f"ROC-AUC:         {roc_auc:.4f}")

    print("\nReal sınıfı:")
    print(f"Precision:       {real_precision:.4f}")
    print(f"Recall:          {real_recall:.4f}")
    print(f"F1:              {real_f1:.4f}")

    print("\nFake sınıfı:")
    print(f"Precision:       {fake_precision:.4f}")
    print(f"Recall:          {fake_recall:.4f}")
    print(f"F1:              {fake_f1:.4f}")

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
            target_names=[
                "real",
                "fake",
            ],
            digits=4,
            zero_division=0,
        )
    )

    result_dataframe = dataframe.copy()

    result_dataframe["prediction_id"] = (
        predictions
    )

    result_dataframe["prediction"] = (
        result_dataframe["prediction_id"]
        .map(
            {
                0: "real",
                1: "fake",
            }
        )
    )

    result_dataframe["decision_score"] = (
        decision_scores
    )

    result_dataframe["correct"] = (
        result_dataframe["label_id"]
        .eq(
            result_dataframe[
                "prediction_id"
            ]
        )
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

    mistakes_dataframe = result_dataframe[
        ~result_dataframe["correct"]
    ].copy()

    mistakes_path = (
        output_directory
        / f"{split_name}_mistakes.csv"
    )

    mistakes_dataframe.to_csv(
        mistakes_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Toplam yanlış tahmin:",
        f"{len(mistakes_dataframe):,}",
    )

    print(
        "\nSentetik üretim yöntemlerine "
        "göre fake recall:"
    )

    method_dataframe = evaluate_by_method(
        result_dataframe
    )

    if method_dataframe.empty:
        print("Yöntem sonucu bulunamadı.")
    else:
        print(
            method_dataframe.to_string(
                index=False
            )
        )

    method_results_path = (
        output_directory
        / f"{split_name}_method_results.csv"
    )

    method_dataframe.to_csv(
        method_results_path,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "split": split_name,
        "record_count": int(
            len(dataframe)
        ),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "roc_auc": float(roc_auc),
        "real_precision": float(
            real_precision
        ),
        "real_recall": float(
            real_recall
        ),
        "real_f1": float(real_f1),
        "fake_precision": float(
            fake_precision
        ),
        "fake_recall": float(
            fake_recall
        ),
        "fake_f1": float(fake_f1),
        "confusion_matrix": (
            matrix.tolist()
        ),
        "mistake_count": int(
            len(mistakes_dataframe)
        ),
        "method_results": (
            method_dataframe.to_dict(
                orient="records"
            )
        ),
    }


def save_top_features(
    model: LinearSVC,
    vectorizer: TfidfVectorizer,
    output_directory: Path,
    top_n: int = 100,
) -> None:
    """
    SVM'in real ve fake sınıfları için en güçlü kelime özelliklerini kaydeder.
    """
    feature_names = (
        vectorizer.get_feature_names_out()
    )

    coefficients = model.coef_[0]

    feature_dataframe = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )

    fake_features = (
        feature_dataframe
        .sort_values(
            "coefficient",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    fake_features["associated_class"] = (
        "fake"
    )

    real_features = (
        feature_dataframe
        .sort_values(
            "coefficient",
            ascending=True,
        )
        .head(top_n)
        .copy()
    )

    real_features["associated_class"] = (
        "real"
    )

    top_features = pd.concat(
        [
            fake_features,
            real_features,
        ],
        ignore_index=True,
    )

    top_features.to_csv(
        output_directory
        / "top_tfidf_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nEn güçlü özellikler kaydedildi:")
    print(
        output_directory
        / "top_tfidf_features.csv"
    )


def main() -> None:
    project_root = find_project_root()

    data_directory = (
        project_root
        / "data"
        / "training"
        / "article_60k_length_matched"
    )

    train_path = (
        data_directory
        / "article_length_matched_train.csv"
    )

    validation_path = (
        data_directory
        / "article_length_matched_validation.csv"
    )

    test_path = (
        data_directory
        / "article_length_matched_test.csv"
    )

    output_directory = (
        project_root
        / "models"
        / "baselines"
        / "article_tfidf_svm_length_matched"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Uzunluk eşitlenmiş veriler okunuyor...")

    train_data = load_dataset(
        train_path
    )

    validation_data = load_dataset(
        validation_path
    )

    test_data = load_dataset(
        test_path
    )

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

    check_split_leakage(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
    )

    print("\n" + "=" * 70)
    print("TF-IDF ÖZELLİKLERİ HAZIRLANIYOR")
    print("=" * 70)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        max_features=100_000,
        sublinear_tf=True,
        norm="l2",
    )

    train_features = (
        vectorizer.fit_transform(
            train_data["text"]
        )
    )

    print(
        "Eğitim özellik matrisi:",
        train_features.shape,
    )

    print(
        "Kullanılan TF-IDF özellik sayısı:",
        len(vectorizer.vocabulary_),
    )

    print("\n" + "=" * 70)
    print("LINEAR SVM EĞİTİLİYOR")
    print("=" * 70)

    model = LinearSVC(
        C=1.0,
        class_weight=None,
        max_iter=10_000,
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

    print("\nModel dosyaları kaydediliyor...")

    vectorizer_path = (
        output_directory
        / "tfidf_vectorizer.joblib"
    )

    model_path = (
        output_directory
        / "linear_svm.joblib"
    )

    joblib.dump(
        vectorizer,
        vectorizer_path,
    )

    joblib.dump(
        model,
        model_path,
    )

    save_top_features(
        model=model,
        vectorizer=vectorizer,
        output_directory=output_directory,
        top_n=100,
    )

    metrics = {
        "model_name": (
            "TF-IDF Linear SVM "
            "Length Matched"
        ),
        "random_state": RANDOM_STATE,
        "train_size": int(
            len(train_data)
        ),
        "validation_size": int(
            len(validation_data)
        ),
        "test_size": int(
            len(test_data)
        ),
        "train_pair_count": int(
            train_data[
                "pair_id"
            ].nunique()
        ),
        "validation_pair_count": int(
            validation_data[
                "pair_id"
            ].nunique()
        ),
        "test_pair_count": int(
            test_data[
                "pair_id"
            ].nunique()
        ),
        "tfidf_features": int(
            len(
                vectorizer.vocabulary_
            )
        ),
        "tfidf_parameters": {
            "ngram_range": [
                1,
                2,
            ],
            "min_df": 3,
            "max_df": 0.98,
            "max_features": 100_000,
            "sublinear_tf": True,
            "norm": "l2",
        },
        "svm_parameters": {
            "C": 1.0,
            "max_iter": 10_000,
            "class_weight": None,
        },
        "validation": (
            validation_results
        ),
        "test": test_results,
    }

    metrics_path = (
        output_directory
        / "metrics.json"
    )

    with metrics_path.open(
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

    print("Model klasörü:")
    print(output_directory)

    print("\nKaydedilen temel dosyalar:")

    saved_files = [
        vectorizer_path,
        model_path,
        metrics_path,
        (
            output_directory
            / "validation_predictions.csv"
        ),
        (
            output_directory
            / "validation_mistakes.csv"
        ),
        (
            output_directory
            / "validation_method_results.csv"
        ),
        (
            output_directory
            / "test_predictions.csv"
        ),
        (
            output_directory
            / "test_mistakes.csv"
        ),
        (
            output_directory
            / "test_method_results.csv"
        ),
        (
            output_directory
            / "top_tfidf_features.csv"
        ),
    ]

    for path in saved_files:
        print("-", path)


if __name__ == "__main__":
    main()