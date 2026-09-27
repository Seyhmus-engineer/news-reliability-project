# ============================================================
# BELİRSİZLİK EŞİĞİ ANALİZİ
#
# Dosya:
# training/analyze_uncertainty_threshold.py
#
# Amaç:
# - Validation verisindeki haberleri mevcut ModelService ile
#   analiz etmek.
# - Model güven değerlerini toplamak.
# - Farklı güven eşiklerini karşılaştırmak.
# - "Sonuç Belirsiz" durumu için veri tabanlı eşik önermek.
# ============================================================

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


# ============================================================
# PROJE YOLLARI
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from backend.app.services.model_service import (  # noqa: E402
    model_service,
)


REPORTS_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
)

PREDICTIONS_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "electra_validation_uncertainty_predictions.csv"
)

THRESHOLDS_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "electra_validation_uncertainty_thresholds.csv"
)

ERRORS_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "electra_validation_uncertainty_errors.csv"
)


# ============================================================
# ÇALIŞMA AYARLARI
# ============================================================

VALIDATION_FILE_NAME = (
    "article_validation_6k.csv"
)

# None olduğunda validation verisinin tamamı çalıştırılır.
# Hızlı deneme için geçici olarak 100 yapılabilir.
MAX_ROWS: int | None = None

# Her kaç haberde ilerleme yazdırılacağı.
PROGRESS_INTERVAL = 100

# İncelenecek güven eşikleri.
THRESHOLDS = np.arange(
    0.50,
    0.991,
    0.01,
)


# ============================================================
# VALIDATION DOSYASINI BUL
# ============================================================

def find_validation_file() -> Path:
    """
    Proje klasöründe article_validation_6k.csv dosyasını bulur.
    """

    direct_candidates = [
        (
            PROJECT_ROOT
            / "datasets"
            / VALIDATION_FILE_NAME
        ),
        (
            PROJECT_ROOT
            / "data"
            / VALIDATION_FILE_NAME
        ),
        (
            PROJECT_ROOT
            / VALIDATION_FILE_NAME
        ),
    ]

    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    recursive_matches = list(
        PROJECT_ROOT.rglob(
            VALIDATION_FILE_NAME
        )
    )

    if not recursive_matches:
        raise FileNotFoundError(
            "Validation dosyası bulunamadı.\n"
            f"Aranan dosya: {VALIDATION_FILE_NAME}\n"
            f"Proje kökü: {PROJECT_ROOT}"
        )

    recursive_matches.sort(
        key=lambda path: (
            len(path.parts),
            str(path),
        )
    )

    return recursive_matches[0]


# ============================================================
# SÜTUN YARDIMCILARI
# ============================================================

def create_column_map(
    dataframe: pd.DataFrame,
) -> dict[str, str]:
    """
    Küçük harf sütun adı -> gerçek sütun adı eşlemesi üretir.
    """

    return {
        str(column)
        .strip()
        .lower(): str(column)

        for column in dataframe.columns
    }


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Verilen aday isimler arasından mevcut sütunu bulur.
    """

    column_map = create_column_map(
        dataframe
    )

    for candidate in candidates:
        normalized_candidate = (
            candidate
            .strip()
            .lower()
        )

        if normalized_candidate in column_map:
            return column_map[
                normalized_candidate
            ]

    return None


# ============================================================
# METİN SÜTUNUNU HAZIRLA
# ============================================================

def resolve_text_series(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series, str]:
    """
    Haber metninin bulunduğu sütunu otomatik belirler.

    Tek bir text/content sütunu yoksa başlık ve içerik
    sütunlarını birleştirir.
    """

    direct_text_column = find_column(
        dataframe,
        [
            "text",
            "full_text",
            "combined_text",
            "article_text",
            "news_text",
            "haber_metni",
            "content",
            "body",
            "article",
        ],
    )

    if direct_text_column is not None:
        text_series = (
            dataframe[
                direct_text_column
            ]
            .fillna("")
            .astype(str)
            .str.replace(
                r"\s+",
                " ",
                regex=True,
            )
            .str.strip()
        )

        return (
            text_series,
            direct_text_column,
        )

    title_column = find_column(
        dataframe,
        [
            "title",
            "headline",
            "baslik",
            "başlık",
            "news_title",
        ],
    )

    content_column = find_column(
        dataframe,
        [
            "content",
            "body",
            "article_body",
            "description",
            "haber",
            "news",
        ],
    )

    if (
        title_column is not None
        and content_column is not None
    ):
        title_series = (
            dataframe[
                title_column
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        content_series = (
            dataframe[
                content_column
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        combined_series = (
            title_series
            + "\n\n"
            + content_series
        ).str.replace(
            r"\s+",
            " ",
            regex=True,
        ).str.strip()

        return (
            combined_series,
            (
                f"{title_column} + "
                f"{content_column}"
            ),
        )

    raise ValueError(
        "Haber metni sütunu otomatik bulunamadı.\n"
        f"Mevcut sütunlar: "
        f"{list(dataframe.columns)}"
    )


# ============================================================
# LABEL SÜTUNUNU HAZIRLA
# ============================================================

def normalize_label(
    value: Any,
) -> int:
    """
    Label değerini proje standardına çevirir:

    0 -> real
    1 -> fake
    """

    if pd.isna(value):
        raise ValueError(
            "Boş label değeri bulundu."
        )

    if isinstance(
        value,
        (
            int,
            np.integer,
            bool,
        ),
    ):
        numeric_value = int(value)

        if numeric_value in {0, 1}:
            return numeric_value

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):
        if float(value).is_integer():
            numeric_value = int(value)

            if numeric_value in {0, 1}:
                return numeric_value

    normalized_value = (
        str(value)
        .strip()
        .lower()
    )

    label_mapping = {
        "0": 0,
        "real": 0,
        "gerçek": 0,
        "gercek": 0,
        "true": 0,
        "reliable": 0,

        "1": 1,
        "fake": 1,
        "sahte": 1,
        "false": 1,
        "unreliable": 1,
    }

    if normalized_value not in label_mapping:
        raise ValueError(
            "Desteklenmeyen label değeri: "
            f"{value!r}"
        )

    return label_mapping[
        normalized_value
    ]


def resolve_label_series(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series, str]:
    label_column = find_column(
        dataframe,
        [
            "label",
            "label_id",
            "target",
            "class",
            "category",
            "y",
        ],
    )

    if label_column is None:
        raise ValueError(
            "Label sütunu bulunamadı.\n"
            f"Mevcut sütunlar: "
            f"{list(dataframe.columns)}"
        )

    normalized_labels = dataframe[
        label_column
    ].map(
        normalize_label
    )

    return (
        normalized_labels,
        label_column,
    )


# ============================================================
# MODEL TAHMİNLERİNİ ÜRET
# ============================================================

def generate_predictions(
    texts: pd.Series,
    labels: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    prediction_records: list[
        dict[str, Any]
    ] = []

    error_records: list[
        dict[str, Any]
    ] = []

    total_rows = len(texts)

    print()
    print("=" * 75)
    print("VALIDATION TAHMİNLERİ BAŞLADI")
    print("=" * 75)
    print(
        "Toplam haber:",
        total_rows,
    )

    start_time = time.perf_counter()

    for position, (
        original_index,
        text,
    ) in enumerate(
        texts.items(),
        start=1,
    ):
        true_label = int(
            labels.loc[
                original_index
            ]
        )

        try:
            prediction = (
                model_service.predict(
                    text
                )
            )

            predicted_label = int(
                prediction[
                    "label_id"
                ]
            )

            confidence = float(
                prediction[
                    "confidence"
                ]
            )

            real_probability = float(
                prediction[
                    "probabilities"
                ][
                    "real"
                ]
            )

            fake_probability = float(
                prediction[
                    "probabilities"
                ][
                    "fake"
                ]
            )

            prediction_records.append(
                {
                    "original_index": (
                        original_index
                    ),

                    "true_label": (
                        true_label
                    ),

                    "predicted_label": (
                        predicted_label
                    ),

                    "correct": (
                        predicted_label
                        == true_label
                    ),

                    "confidence": (
                        confidence
                    ),

                    "real_probability": (
                        real_probability
                    ),

                    "fake_probability": (
                        fake_probability
                    ),

                    "total_token_count": int(
                        prediction.get(
                            "total_token_count",
                            prediction.get(
                                "processed_token_count",
                                0,
                            ),
                        )
                    ),

                    "chunk_count": int(
                        prediction.get(
                            "chunk_count",
                            1,
                        )
                    ),

                    "is_chunked": bool(
                        prediction.get(
                            "is_chunked",
                            False,
                        )
                    ),
                }
            )

        except Exception as error:
            error_records.append(
                {
                    "original_index": (
                        original_index
                    ),

                    "true_label": (
                        true_label
                    ),

                    "text_preview": (
                        str(text)[:250]
                    ),

                    "error_type": (
                        error
                        .__class__
                        .__name__
                    ),

                    "error_message": (
                        str(error)
                    ),
                }
            )

        if (
            position % PROGRESS_INTERVAL == 0
            or position == total_rows
        ):
            elapsed_seconds = (
                time.perf_counter()
                - start_time
            )

            average_seconds = (
                elapsed_seconds
                / position
            )

            remaining_seconds = (
                average_seconds
                * (
                    total_rows
                    - position
                )
            )

            print(
                f"[{position:>5}/{total_rows}] "
                f"Tamamlandı | "
                f"Hata: {len(error_records)} | "
                f"Kalan yaklaşık: "
                f"{remaining_seconds / 60:.1f} dk"
            )

    prediction_dataframe = (
        pd.DataFrame(
            prediction_records
        )
    )

    error_dataframe = pd.DataFrame(
        error_records
    )

    return (
        prediction_dataframe,
        error_dataframe,
    )


# ============================================================
# EŞİK TABLOSUNU HESAPLA
# ============================================================

def calculate_threshold_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    true_labels = predictions[
        "true_label"
    ].to_numpy(
        dtype=int
    )

    predicted_labels = predictions[
        "predicted_label"
    ].to_numpy(
        dtype=int
    )

    confidences = predictions[
        "confidence"
    ].to_numpy(
        dtype=float
    )

    correct_predictions = (
        predicted_labels
        == true_labels
    )

    incorrect_predictions = (
        ~correct_predictions
    )

    total_count = len(
        predictions
    )

    total_error_count = int(
        incorrect_predictions.sum()
    )

    threshold_records: list[
        dict[str, Any]
    ] = []

    for threshold in THRESHOLDS:
        uncertain_mask = (
            confidences
            < threshold
        )

        accepted_mask = (
            ~uncertain_mask
        )

        accepted_count = int(
            accepted_mask.sum()
        )

        uncertain_count = int(
            uncertain_mask.sum()
        )

        coverage = (
            accepted_count
            / total_count
        )

        uncertainty_rate = (
            uncertain_count
            / total_count
        )

        if accepted_count > 0:
            accepted_true = true_labels[
                accepted_mask
            ]

            accepted_predictions = (
                predicted_labels[
                    accepted_mask
                ]
            )

            selective_accuracy = float(
                accuracy_score(
                    accepted_true,
                    accepted_predictions,
                )
            )

            selective_macro_f1 = float(
                f1_score(
                    accepted_true,
                    accepted_predictions,
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                )
            )

        else:
            selective_accuracy = np.nan
            selective_macro_f1 = np.nan

        errors_caught = int(
            (
                uncertain_mask
                & incorrect_predictions
            ).sum()
        )

        errors_missed = int(
            (
                accepted_mask
                & incorrect_predictions
            ).sum()
        )

        correct_marked_uncertain = int(
            (
                uncertain_mask
                & correct_predictions
            ).sum()
        )

        uncertainty_precision = (
            errors_caught
            / uncertain_count
            if uncertain_count > 0
            else 0.0
        )

        error_catch_rate = (
            errors_caught
            / total_error_count
            if total_error_count > 0
            else 0.0
        )

        precision_plus_recall = (
            uncertainty_precision
            + error_catch_rate
        )

        if precision_plus_recall > 0:
            error_detection_f1 = (
                2
                * uncertainty_precision
                * error_catch_rate
                / precision_plus_recall
            )

        else:
            error_detection_f1 = 0.0

        threshold_records.append(
            {
                "threshold": round(
                    float(threshold),
                    2,
                ),

                "accepted_count": (
                    accepted_count
                ),

                "uncertain_count": (
                    uncertain_count
                ),

                "coverage": (
                    coverage
                ),

                "uncertainty_rate": (
                    uncertainty_rate
                ),

                "selective_accuracy": (
                    selective_accuracy
                ),

                "selective_macro_f1": (
                    selective_macro_f1
                ),

                "total_model_errors": (
                    total_error_count
                ),

                "errors_caught": (
                    errors_caught
                ),

                "errors_missed": (
                    errors_missed
                ),

                "correct_marked_uncertain": (
                    correct_marked_uncertain
                ),

                "uncertainty_precision": (
                    uncertainty_precision
                ),

                "error_catch_rate": (
                    error_catch_rate
                ),

                "error_detection_f1": (
                    error_detection_f1
                ),
            }
        )

    return pd.DataFrame(
        threshold_records
    )


# ============================================================
# ÖNERİLEN EŞİKLER
# ============================================================

def find_error_detection_recommendation(
    threshold_table: pd.DataFrame,
) -> pd.Series:
    """
    Model hatalarını yakalama F1 değeri en yüksek eşiği seçer.
    """

    sorted_table = (
        threshold_table
        .sort_values(
            by=[
                "error_detection_f1",
                "coverage",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return sorted_table.iloc[0]


def find_selective_accuracy_recommendation(
    threshold_table: pd.DataFrame,
    target_accuracy: float,
) -> pd.Series | None:
    """
    Belirlenen doğruluk hedefini sağlayan eşikler arasında
    kapsamı en yüksek olanı seçer.
    """

    eligible_rows = threshold_table[
        (
            threshold_table[
                "selective_accuracy"
            ]
            >= target_accuracy
        )
        & (
            threshold_table[
                "accepted_count"
            ]
            > 0
        )
    ]

    if eligible_rows.empty:
        return None

    eligible_rows = (
        eligible_rows
        .sort_values(
            by=[
                "coverage",
                "threshold",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    return eligible_rows.iloc[0]


# ============================================================
# SONUÇLARI YAZDIR
# ============================================================

def print_recommendation(
    title: str,
    row: pd.Series | None,
) -> None:
    print()
    print("-" * 75)
    print(title)
    print("-" * 75)

    if row is None:
        print(
            "Bu hedefi sağlayan bir eşik bulunamadı."
        )

        return

    print(
        "Önerilen eşik:",
        f"{float(row['threshold']):.2f}",
    )

    print(
        "Kabul edilen haber oranı:",
        f"%{float(row['coverage']) * 100:.2f}",
    )

    print(
        "Belirsiz bırakılan haber oranı:",
        f"%{float(row['uncertainty_rate']) * 100:.2f}",
    )

    print(
        "Kabul edilen sonuçların doğruluğu:",
        f"%{float(row['selective_accuracy']) * 100:.2f}",
    )

    print(
        "Kabul edilen sonuçların Macro F1 değeri:",
        f"%{float(row['selective_macro_f1']) * 100:.2f}",
    )

    print(
        "Yakalanan model hatası:",
        (
            f"{int(row['errors_caught'])} / "
            f"{int(row['total_model_errors'])}"
        ),
    )

    print(
        "Hata yakalama oranı:",
        f"%{float(row['error_catch_rate']) * 100:.2f}",
    )

    print(
        "Hata yakalama F1:",
        f"{float(row['error_detection_f1']):.4f}",
    )


# ============================================================
# ANA PROGRAM
# ============================================================

def main() -> None:
    print("=" * 75)
    print("ELECTRA BELİRSİZLİK EŞİĞİ ANALİZİ")
    print("=" * 75)

    validation_path = (
        find_validation_file()
    )

    print(
        "Validation dosyası:",
        validation_path,
    )

    dataframe = pd.read_csv(
        validation_path
    )

    print(
        "Ham satır sayısı:",
        len(dataframe),
    )

    text_series, text_source = (
        resolve_text_series(
            dataframe
        )
    )

    label_series, label_column = (
        resolve_label_series(
            dataframe
        )
    )

    valid_mask = (
        text_series.str.len() >= 20
    )

    removed_empty_count = int(
        (~valid_mask).sum()
    )

    dataframe = dataframe.loc[
        valid_mask
    ].copy()

    text_series = text_series.loc[
        valid_mask
    ]

    label_series = label_series.loc[
        valid_mask
    ]

    if MAX_ROWS is not None:
        dataframe = dataframe.head(
            MAX_ROWS
        )

        selected_indices = (
            dataframe.index
        )

        text_series = text_series.loc[
            selected_indices
        ]

        label_series = label_series.loc[
            selected_indices
        ]

    print(
        "Metin kaynağı:",
        text_source,
    )

    print(
        "Label sütunu:",
        label_column,
    )

    print(
        "Boş/kısa olduğu için çıkarılan:",
        removed_empty_count,
    )

    print(
        "Analiz edilecek haber:",
        len(text_series),
    )

    label_counts = (
        label_series
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print(
        "Label dağılımı:",
        {
            "real": int(
                label_counts.get(
                    0,
                    0,
                )
            ),

            "fake": int(
                label_counts.get(
                    1,
                    0,
                )
            ),
        },
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Model yükleniyor...")

    model_service.load_model()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_start_time = (
        time.perf_counter()
    )

    predictions, errors = (
        generate_predictions(
            text_series,
            label_series,
        )
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_duration = (
        time.perf_counter()
        - total_start_time
    )

    if predictions.empty:
        raise RuntimeError(
            "Hiçbir validation tahmini üretilemedi."
        )

    predictions.to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if not errors.empty:
        errors.to_csv(
            ERRORS_OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    overall_accuracy = float(
        accuracy_score(
            predictions[
                "true_label"
            ],
            predictions[
                "predicted_label"
            ],
        )
    )

    overall_macro_f1 = float(
        f1_score(
            predictions[
                "true_label"
            ],
            predictions[
                "predicted_label"
            ],
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )

    total_errors = int(
        (
            predictions[
                "true_label"
            ]
            != predictions[
                "predicted_label"
            ]
        ).sum()
    )

    threshold_table = (
        calculate_threshold_table(
            predictions
        )
    )

    threshold_table.to_csv(
        THRESHOLDS_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    error_f1_recommendation = (
        find_error_detection_recommendation(
            threshold_table
        )
    )

    accuracy_97_recommendation = (
        find_selective_accuracy_recommendation(
            threshold_table,
            target_accuracy=0.97,
        )
    )

    accuracy_98_recommendation = (
        find_selective_accuracy_recommendation(
            threshold_table,
            target_accuracy=0.98,
        )
    )

    print()
    print("=" * 75)
    print("GENEL VALIDATION SONUCU")
    print("=" * 75)

    print(
        "Başarılı tahmin:",
        len(predictions),
    )

    print(
        "Tahmin hatası oluşan kayıt:",
        len(errors),
    )

    print(
        "Accuracy:",
        f"{overall_accuracy:.6f}",
    )

    print(
        "Macro F1:",
        f"{overall_macro_f1:.6f}",
    )

    print(
        "Yanlış tahmin:",
        (
            f"{total_errors} / "
            f"{len(predictions)}"
        ),
    )

    print(
        "Toplam analiz süresi:",
        f"{total_duration / 60:.2f} dakika",
    )

    print_recommendation(
        (
            "ÖNERİ 1 — MODEL HATALARINI "
            "YAKALAMA F1 DEĞERİ EN YÜKSEK EŞİK"
        ),
        error_f1_recommendation,
    )

    print_recommendation(
        (
            "ÖNERİ 2 — KABUL EDİLEN SONUÇLARDA "
            "EN AZ %97 DOĞRULUK"
        ),
        accuracy_97_recommendation,
    )

    print_recommendation(
        (
            "ÖNERİ 3 — KABUL EDİLEN SONUÇLARDA "
            "EN AZ %98 DOĞRULUK"
        ),
        accuracy_98_recommendation,
    )

    print()
    print("=" * 75)
    print("DOSYALAR KAYDEDİLDİ")
    print("=" * 75)

    print(
        "Tahminler:",
        PREDICTIONS_OUTPUT_PATH,
    )

    print(
        "Eşik tablosu:",
        THRESHOLDS_OUTPUT_PATH,
    )

    if not errors.empty:
        print(
            "Hatalı kayıtlar:",
            ERRORS_OUTPUT_PATH,
        )

    print()
    print(
        "Analiz tamamlandı."
    )


if __name__ == "__main__":
    main()