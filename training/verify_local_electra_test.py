# ============================================================
# LOCAL ELECTRA CHECKPOINT DOĞRULAMA
#
# Dosya:
# training/verify_local_electra_test.py
#
# Amaç:
# - Yerel model klasörünün doğru checkpoint olduğunu doğrulamak
# - Label sırasını kontrol etmek
# - article_test_6k.csv üzerinde kesin test metriklerini üretmek
# - Trainer kullan# - article_test_6k.csv üzerinde kesin test metriklerini üretmadan sıralı tahmin almak
#
# Proje sınıfları:
# 0 -> real
# 1 -> fake
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
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
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


MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "electra_turkish_article_60k_final"
)

TEST_FILE_NAME = (
    "article_test_6k.csv"
)

REPORTS_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
)

PREDICTIONS_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "local_electra_test_predictions.csv"
)


# ============================================================
# ÇALIŞMA AYARLARI
# ============================================================

MAX_LENGTH = 512

# RTX 5070 Laptop GPU için güvenli başlangıç değeri.
# CUDA bellek hatası alınırsa 8 yapılabilir.
BATCH_SIZE = 16

PROGRESS_INTERVAL = 200


# ============================================================
# PROJEDE BEKLENEN KESİN SONUÇLAR
# ============================================================

EXPECTED_RESULTS = {
    "accuracy": 0.952500,
    "macro_f1": 0.952494,
    "roc_auc": 0.990908,
    "real_recall": 0.963667,
    "fake_recall": 0.941333,
}

EXPECTED_CONFUSION_MATRIX = np.array(
    [
        [2891, 109],
        [176, 2824],
    ],
    dtype=int,
)


# ============================================================
# TEST DOSYASINI BUL
# ============================================================

def find_test_file() -> Path:
    direct_candidates = [
        (
            PROJECT_ROOT
            / "datasets"
            / TEST_FILE_NAME
        ),
        (
            PROJECT_ROOT
            / "data"
            / TEST_FILE_NAME
        ),
        (
            PROJECT_ROOT
            / TEST_FILE_NAME
        ),
    ]

    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    recursive_matches = list(
        PROJECT_ROOT.rglob(
            TEST_FILE_NAME
        )
    )

    if not recursive_matches:
        raise FileNotFoundError(
            "Test dosyası bulunamadı.\n"
            f"Aranan dosya: {TEST_FILE_NAME}\n"
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
# CSV DOSYASINI OKU
# ============================================================

def read_csv_safely(
    path: Path,
) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path
        )

    except Exception as first_error:
        print(
            "Standart CSV okuması başarısız oldu."
        )

        print(
            "Otomatik ayraç algılama deneniyor..."
        )

        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
            )

        except Exception:
            raise RuntimeError(
                "Test CSV dosyası okunamadı."
            ) from first_error


# ============================================================
# SÜTUN BULMA
# ============================================================

def create_column_map(
    dataframe: pd.DataFrame,
) -> dict[str, str]:
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
        "Test verisinde haber metni sütunu bulunamadı.\n"
        f"Mevcut sütunlar: "
        f"{list(dataframe.columns)}"
    )


# ============================================================
# LABEL NORMALLEŞTİRME
# ============================================================

def normalize_label(
    value: Any,
) -> int:
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

    mapping = {
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

    if normalized_value not in mapping:
        raise ValueError(
            "Desteklenmeyen label değeri: "
            f"{value!r}"
        )

    return mapping[
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
            "Test verisinde label sütunu bulunamadı.\n"
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
# MODEL KLASÖRÜNÜ DOĞRULA
# ============================================================

def validate_model_directory() -> None:
    if not MODEL_DIRECTORY.is_dir():
        raise FileNotFoundError(
            "Yerel model klasörü bulunamadı:\n"
            f"{MODEL_DIRECTORY}"
        )

    config_path = (
        MODEL_DIRECTORY
        / "config.json"
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            "config.json bulunamadı:\n"
            f"{config_path}"
        )

    weight_candidates = [
        (
            MODEL_DIRECTORY
            / "model.safetensors"
        ),
        (
            MODEL_DIRECTORY
            / "pytorch_model.bin"
        ),
    ]

    if not any(
        path.is_file()
        for path in weight_candidates
    ):
        raise FileNotFoundError(
            "Model ağırlık dosyası bulunamadı."
        )


# ============================================================
# MODEL VE TOKENIZER YÜKLE
# ============================================================

def load_model_and_tokenizer(
    device: torch.device,
) -> tuple[Any, Any]:
    validate_model_directory()

    print()
    print("Tokenizer yükleniyor...")

    tokenizer = (
        AutoTokenizer.from_pretrained(
            str(MODEL_DIRECTORY),
            local_files_only=True,
            use_fast=True,
        )
    )

    print("Model yükleniyor...")

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            str(MODEL_DIRECTORY),
            local_files_only=True,
        )
    )

    model.to(
        device
    )

    model.eval()

    if int(model.config.num_labels) != 2:
        raise RuntimeError(
            "Model iki sınıflı değil. "
            f"num_labels: "
            f"{model.config.num_labels}"
        )

    return (
        tokenizer,
        model,
    )


# ============================================================
# LABEL CONFIG KONTROLÜ
# ============================================================

def print_label_configuration(
    model: Any,
) -> None:
    id2label = (
        model.config.id2label
        or {}
    )

    print()
    print("=" * 75)
    print("MODEL LABEL YAPISI")
    print("=" * 75)

    print(
        "id2label:",
        id2label,
    )

    print(
        "Proje standardı:",
        {
            0: "real",
            1: "fake",
        },
    )

    label_zero = str(
        id2label.get(
            0,
            id2label.get(
                "0",
                "",
            ),
        )
    ).strip().lower()

    label_one = str(
        id2label.get(
            1,
            id2label.get(
                "1",
                "",
            ),
        )
    ).strip().lower()

    valid_zero = label_zero in {
        "real",
        "label_0",
        "0",
    }

    valid_one = label_one in {
        "fake",
        "label_1",
        "1",
    }

    if valid_zero and valid_one:
        print(
            "Label kontrolü: UYUMLU"
        )

    else:
        print(
            "UYARI: Model config label adları "
            "proje standardıyla açıkça eşleşmiyor."
        )

        print(
            "Tahmin değerlendirmesinde yine "
            "0=real ve 1=fake kullanılacak."
        )


# ============================================================
# SIRALI BATCH TAHMİNİ
# ============================================================

def predict_in_batches(
    texts: pd.Series,
    tokenizer: Any,
    model: Any,
    device: torch.device,
) -> pd.DataFrame:
    total_count = len(
        texts
    )

    records: list[
        dict[str, Any]
    ] = []

    start_time = time.perf_counter()

    print()
    print("=" * 75)
    print("SIRALI TEST TAHMİNLERİ BAŞLADI")
    print("=" * 75)

    for batch_start in range(
        0,
        total_count,
        BATCH_SIZE,
    ):
        batch_end = min(
            batch_start + BATCH_SIZE,
            total_count,
        )

        batch_series = texts.iloc[
            batch_start:batch_end
        ]

        batch_texts = (
            batch_series
            .astype(str)
            .tolist()
        )

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        model_inputs = {
            key: value.to(device)

            for key, value
            in encoded.items()
        }

        with torch.inference_mode():
            outputs = model(
                **model_inputs
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

        real_probabilities = (
            probabilities[:, 0]
            .detach()
            .cpu()
            .numpy()
        )

        fake_probabilities = (
            probabilities[:, 1]
            .detach()
            .cpu()
            .numpy()
        )

        predicted_labels = (
            probabilities
            .argmax(dim=-1)
            .detach()
            .cpu()
            .numpy()
        )

        attention_masks = (
            model_inputs[
                "attention_mask"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        token_counts = (
            attention_masks
            .sum(axis=1)
        )

        for local_position, (
            original_index,
            text,
        ) in enumerate(
            batch_series.items()
        ):
            real_probability = float(
                real_probabilities[
                    local_position
                ]
            )

            fake_probability = float(
                fake_probabilities[
                    local_position
                ]
            )

            predicted_label = int(
                predicted_labels[
                    local_position
                ]
            )

            records.append(
                {
                    "original_index": (
                        original_index
                    ),

                    "predicted_label": (
                        predicted_label
                    ),

                    "confidence": max(
                        real_probability,
                        fake_probability,
                    ),

                    "real_probability": (
                        real_probability
                    ),

                    "fake_probability": (
                        fake_probability
                    ),

                    "processed_token_count": int(
                        token_counts[
                            local_position
                        ]
                    ),

                    "text_preview": (
                        str(text)[:300]
                    ),
                }
            )

        completed = batch_end

        if (
            completed % PROGRESS_INTERVAL == 0
            or completed == total_count
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )

            average_per_item = (
                elapsed / completed
            )

            remaining_seconds = (
                average_per_item
                * (
                    total_count
                    - completed
                )
            )

            print(
                f"[{completed:>5}/{total_count}] "
                f"Tamamlandı | "
                f"Kalan yaklaşık: "
                f"{remaining_seconds / 60:.1f} dk"
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# SONUÇLARI HESAPLA
# ============================================================

def calculate_results(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
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

    fake_probabilities = predictions[
        "fake_probability"
    ].to_numpy(
        dtype=float
    )

    accuracy = float(
        accuracy_score(
            true_labels,
            predicted_labels,
        )
    )

    macro_f1 = float(
        f1_score(
            true_labels,
            predicted_labels,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )

    roc_auc = float(
        roc_auc_score(
            true_labels,
            fake_probabilities,
        )
    )

    recalls = recall_score(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    )

    wrong_count = int(
        (
            true_labels
            != predicted_labels
        ).sum()
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "real_recall": float(
            recalls[0]
        ),
        "fake_recall": float(
            recalls[1]
        ),
        "confusion_matrix": matrix,
        "wrong_count": wrong_count,
    }


# ============================================================
# BEKLENEN SONUÇLA KARŞILAŞTIR
# ============================================================

def compare_with_expected(
    results: dict[str, Any],
) -> None:
    print()
    print("=" * 75)
    print("ÖNCEKİ KESİN TEST SONUCUYLA KARŞILAŞTIRMA")
    print("=" * 75)

    metric_names = [
        "accuracy",
        "macro_f1",
        "roc_auc",
        "real_recall",
        "fake_recall",
    ]

    maximum_difference = 0.0

    for metric_name in metric_names:
        current_value = float(
            results[
                metric_name
            ]
        )

        expected_value = float(
            EXPECTED_RESULTS[
                metric_name
            ]
        )

        difference = abs(
            current_value
            - expected_value
        )

        maximum_difference = max(
            maximum_difference,
            difference,
        )

        print(
            f"{metric_name:<15} "
            f"Yerel: {current_value:.6f} | "
            f"Beklenen: {expected_value:.6f} | "
            f"Fark: {difference:.6f}"
        )

    current_matrix = results[
        "confusion_matrix"
    ]

    print()
    print("Yerel confusion matrix:")

    print(
        current_matrix
    )

    print()
    print("Beklenen confusion matrix:")

    print(
        EXPECTED_CONFUSION_MATRIX
    )

    matrix_difference = (
        current_matrix
        - EXPECTED_CONFUSION_MATRIX
    )

    print()
    print("Confusion matrix farkı:")

    print(
        matrix_difference
    )

    print()

    if maximum_difference <= 0.005:
        print(
            "SONUÇ: Yerel checkpoint ve label yapısı "
            "önceki test sonucuyla uyumlu görünüyor."
        )

        print(
            "Gerçek dünya sahte haberlerindeki sorun büyük "
            "olasılıkla model genellemesi/veri dağılımı sorunudur."
        )

    else:
        print(
            "SONUÇ: Yerel sonuç ile önceki kesin test sonucu "
            "arasında belirgin fark var."
        )

        print(
            "Checkpoint, tokenizer, metin sütunu ve "
            "label eşlemesi ayrıca incelenmelidir."
        )


# ============================================================
# FALSE NEGATIVE ÖRNEKLERİ
# ============================================================

def print_false_negative_examples(
    predictions: pd.DataFrame,
) -> None:
    false_negatives = predictions[
        (
            predictions[
                "true_label"
            ]
            == 1
        )
        & (
            predictions[
                "predicted_label"
            ]
            == 0
        )
    ].copy()

    false_negatives = (
        false_negatives
        .sort_values(
            by="real_probability",
            ascending=False,
        )
        .head(10)
    )

    print()
    print("=" * 75)
    print("EN YÜKSEK GÜVENLİ FALSE NEGATIVE ÖRNEKLERİ")
    print("Gerçek etiket: fake | Model tahmini: real")
    print("=" * 75)

    if false_negatives.empty:
        print(
            "False negative bulunmadı."
        )

        return

    for row_number, row in enumerate(
        false_negatives.itertuples(
            index=False
        ),
        start=1,
    ):
        print()
        print(
            f"{row_number}. CSV index: "
            f"{row.original_index}"
        )

        print(
            "Real olasılığı:",
            f"%{row.real_probability * 100:.2f}",
        )

        print(
            "Fake olasılığı:",
            f"%{row.fake_probability * 100:.2f}",
        )

        print(
            "Metin önizleme:",
            row.text_preview,
        )


# ============================================================
# ANA PROGRAM
# ============================================================

def main() -> None:
    print("=" * 75)
    print("LOCAL TURKISH ELECTRA TEST DOĞRULAMASI")
    print("=" * 75)

    test_path = find_test_file()

    print(
        "Test dosyası:",
        test_path,
    )

    dataframe = read_csv_safely(
        test_path
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

    invalid_count = int(
        (~valid_mask).sum()
    )

    if invalid_count > 0:
        print(
            "UYARI: Boş veya çok kısa metin:",
            invalid_count,
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

    print(
        "Metin kaynağı:",
        text_source,
    )

    print(
        "Label sütunu:",
        label_column,
    )

    print(
        "Analiz edilecek satır:",
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

    if len(text_series) != 6000:
        print(
            "UYARI: Beklenen test satırı 6000, "
            f"bulunan: {len(text_series)}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Kullanılan cihaz:",
        device,
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        torch.set_float32_matmul_precision(
            "high"
        )

    tokenizer, model = (
        load_model_and_tokenizer(
            device
        )
    )

    print(
        "Tokenizer sınıfı:",
        tokenizer.__class__.__name__,
    )

    print(
        "Model sınıfı:",
        model.__class__.__name__,
    )

    print_label_configuration(
        model
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    test_start_time = (
        time.perf_counter()
    )

    predictions = predict_in_batches(
        text_series,
        tokenizer,
        model,
        device,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    test_duration = (
        time.perf_counter()
        - test_start_time
    )

    predictions[
        "true_label"
    ] = label_series.to_numpy(
        dtype=int
    )

    predictions[
        "correct"
    ] = (
        predictions[
            "true_label"
        ]
        == predictions[
            "predicted_label"
        ]
    )

    results = calculate_results(
        predictions
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    matrix = results[
        "confusion_matrix"
    ]

    print()
    print("=" * 75)
    print("YEREL MODEL KESİN TEST SONUCU")
    print("=" * 75)

    print(
        "Accuracy:",
        f"{results['accuracy']:.6f}",
    )

    print(
        "Macro F1:",
        f"{results['macro_f1']:.6f}",
    )

    print(
        "ROC-AUC:",
        f"{results['roc_auc']:.6f}",
    )

    print(
        "Real recall:",
        f"{results['real_recall']:.6f}",
    )

    print(
        "Fake recall:",
        f"{results['fake_recall']:.6f}",
    )

    print(
        "Yanlış tahmin:",
        (
            f"{results['wrong_count']} / "
            f"{len(predictions)}"
        ),
    )

    print()
    print("Confusion matrix:")
    print("Satırlar gerçek, sütunlar tahmin")
    print("Sıra: [real, fake]")

    print(
        matrix
    )

    print()
    print(
        "Doğru sınıflandırılan real:",
        int(matrix[0, 0]),
    )

    print(
        "Fake denilen real:",
        int(matrix[0, 1]),
    )

    print(
        "Real denilen fake:",
        int(matrix[1, 0]),
    )

    print(
        "Doğru sınıflandırılan fake:",
        int(matrix[1, 1]),
    )

    print()
    print(
        "Toplam süre:",
        f"{test_duration / 60:.2f} dakika",
    )

    print(
        "Tahmin dosyası:",
        PREDICTIONS_OUTPUT_PATH,
    )

    compare_with_expected(
        results
    )

    print_false_negative_examples(
        predictions
    )

    print()
    print("=" * 75)
    print("DOĞRULAMA TAMAMLANDI")
    print("=" * 75)


if __name__ == "__main__":
    main()