# ============================================================
# ELECTRA FALSE NEGATIVE ANALİZİ
#
# Dosya:
# training/analyze_electra_false_negatives.py
#
# Amaç:
# - local_electra_test_predictions.csv içinden
#   gerçek etiketi fake olduğu hâlde modelin real dediği
#   kayıtları çıkarmak.
# - Bu kayıtları article_test_6k.csv içindeki tam metin ve
#   metadata ile birleştirmek.
# - Güven, token uzunluğu, üretim yöntemi, kaynak ve kategori
#   bazında özet tablolar oluşturmak.
#
# Proje sınıfları:
# 0 -> real
# 1 -> fake
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PROJE YOLLARI
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

REPORTS_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
)

PREDICTIONS_PATH = (
    REPORTS_DIRECTORY
    / "local_electra_test_predictions.csv"
)

TEST_FILE_NAME = (
    "article_test_6k.csv"
)


# ============================================================
# ÇIKTI DOSYALARI
# ============================================================

FALSE_NEGATIVES_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "electra_false_negatives_full.csv"
)

CONFIDENCE_SUMMARY_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "electra_false_negative_confidence_summary.csv"
)

TOKEN_SUMMARY_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "electra_false_negative_token_summary.csv"
)

METADATA_SUMMARY_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "electra_false_negative_metadata_summary.csv"
)

TEXT_SUMMARY_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "electra_false_negative_analysis_summary.txt"
)


# ============================================================
# BEKLENEN SONUÇ
# ============================================================

EXPECTED_FALSE_NEGATIVE_COUNT = 176


# ============================================================
# DOSYA BULMA
# ============================================================

def find_test_file() -> Path:
    """
    article_test_6k.csv dosyasını proje içinde bulur.
    """

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
# CSV OKUMA
# ============================================================

def read_csv_safely(
    path: Path,
) -> pd.DataFrame:
    """
    CSV dosyasını standart yöntemle, gerekirse otomatik
    ayraç algılamasıyla okur.
    """

    try:
        return pd.read_csv(
            path
        )

    except Exception as first_error:
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
            )

        except Exception:
            raise RuntimeError(
                f"CSV dosyası okunamadı:\n{path}"
            ) from first_error


# ============================================================
# SÜTUN YARDIMCILARI
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
# TAM HABER METNİNİ HAZIRLA
# ============================================================

def resolve_text_series(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series, str]:
    """
    Test veri setindeki haber metnini bulur.

    Tek bir text sütunu yoksa başlık ve içerik sütunlarını
    birleştirir.
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
        "Test veri setinde haber metni sütunu bulunamadı.\n"
        f"Mevcut sütunlar: {list(dataframe.columns)}"
    )


# ============================================================
# GÜVEN ARALIĞI
# ============================================================

def create_confidence_band(
    confidence: float,
) -> str:
    if confidence >= 0.99:
        return "01 — %99 ve üzeri"

    if confidence >= 0.95:
        return "02 — %95–%99"

    if confidence >= 0.90:
        return "03 — %90–%95"

    if confidence >= 0.80:
        return "04 — %80–%90"

    if confidence >= 0.70:
        return "05 — %70–%80"

    if confidence >= 0.60:
        return "06 — %60–%70"

    return "07 — %50–%60"


# ============================================================
# TOKEN UZUNLUK ARALIĞI
# ============================================================

def create_token_band(
    token_count: int,
) -> str:
    if token_count < 150:
        return "01 — 80–149 token"

    if token_count < 200:
        return "02 — 150–199 token"

    if token_count < 250:
        return "03 — 200–249 token"

    if token_count < 300:
        return "04 — 250–299 token"

    if token_count < 400:
        return "05 — 300–399 token"

    return "06 — 400 ve üzeri"


# ============================================================
# İNCELEME ÖNCELİĞİ
# ============================================================

def create_review_priority(
    confidence: float,
) -> str:
    """
    En yüksek güvenle yanlış sınıflandırılan kayıtları önce
    incelemek için öncelik alanı oluşturur.
    """

    if confidence >= 0.99:
        return "1 — Çok yüksek öncelik"

    if confidence >= 0.95:
        return "2 — Yüksek öncelik"

    if confidence >= 0.90:
        return "3 — Orta öncelik"

    return "4 — Düşük öncelik"


# ============================================================
# ÖNİZLEME EŞLEŞMESİ
# ============================================================

def normalize_for_comparison(
    value: Any,
) -> str:
    return (
        str(value)
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
        .lower()
    )


def preview_matches_full_text(
    preview: Any,
    full_text: Any,
) -> bool:
    """
    Tahmin dosyasındaki önizlemenin doğru test satırıyla
    birleştirilip birleştirilmediğini kontrol eder.
    """

    normalized_preview = (
        normalize_for_comparison(
            preview
        )[:120]
    )

    normalized_full_text = (
        normalize_for_comparison(
            full_text
        )[:120]
    )

    if not normalized_preview:
        return False

    return (
        normalized_preview
        == normalized_full_text
    )


# ============================================================
# METADATA ÖZETİ
# ============================================================

def calculate_metadata_summary(
    false_negatives: pd.DataFrame,
    original_columns: list[str],
) -> pd.DataFrame:
    """
    Test veri setinde mevcutsa yöntem, kaynak ve kategori
    sütunlarının dağılımını hesaplar.
    """

    metadata_candidate_groups = {
        "generation_method": [
            "method",
            "generation_method",
            "synthetic_method",
            "fake_method",
            "manipulation_method",
            "manipulation_type",
            "transformation",
            "operation",
        ],

        "source": [
            "source",
            "news_source",
            "publisher",
            "site",
            "domain",
            "origin",
        ],

        "category": [
            "category",
            "news_category",
            "topic",
            "subject",
            "class_name",
        ],

        "dataset": [
            "dataset",
            "dataset_name",
            "data_source",
            "origin_dataset",
        ],
    }

    summary_records: list[
        dict[str, Any]
    ] = []

    original_column_map = {
        str(column)
        .strip()
        .lower(): column

        for column in original_columns
    }

    for metadata_type, candidates in (
        metadata_candidate_groups.items()
    ):
        selected_column = None

        for candidate in candidates:
            if candidate in original_column_map:
                selected_column = (
                    original_column_map[
                        candidate
                    ]
                )

                break

        if selected_column is None:
            continue

        output_column = (
            f"test_{selected_column}"
        )

        if output_column not in false_negatives.columns:
            continue

        value_counts = (
            false_negatives[
                output_column
            ]
            .fillna("BOŞ")
            .astype(str)
            .value_counts(
                dropna=False
            )
        )

        for value, count in value_counts.items():
            summary_records.append(
                {
                    "metadata_type": (
                        metadata_type
                    ),

                    "column_name": (
                        selected_column
                    ),

                    "value": value,

                    "false_negative_count": int(
                        count
                    ),

                    "percentage": (
                        float(count)
                        / len(false_negatives)
                        if len(false_negatives) > 0
                        else 0.0
                    ),
                }
            )

    return pd.DataFrame(
        summary_records
    )


# ============================================================
# ANA ANALİZ
# ============================================================

def main() -> None:
    print("=" * 76)
    print("ELECTRA FALSE NEGATIVE ANALİZİ")
    print("=" * 76)

    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            "Yerel tahmin dosyası bulunamadı:\n"
            f"{PREDICTIONS_PATH}\n\n"
            "Önce verify_local_electra_test.py "
            "dosyasını çalıştır."
        )

    test_path = find_test_file()

    print(
        "Tahmin dosyası:",
        PREDICTIONS_PATH,
    )

    print(
        "Test dosyası:",
        test_path,
    )

    predictions = read_csv_safely(
        PREDICTIONS_PATH
    )

    test_dataframe = read_csv_safely(
        test_path
    )

    print()
    print(
        "Tahmin satırı:",
        len(predictions),
    )

    print(
        "Test satırı:",
        len(test_dataframe),
    )

    required_prediction_columns = {
        "original_index",
        "true_label",
        "predicted_label",
        "confidence",
        "real_probability",
        "fake_probability",
        "processed_token_count",
    }

    missing_columns = (
        required_prediction_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Tahmin dosyasında zorunlu sütunlar eksik:\n"
            f"{sorted(missing_columns)}"
        )

    if len(predictions) != len(test_dataframe):
        raise ValueError(
            "Tahmin ve test dosyasının satır sayıları "
            "eşleşmiyor.\n"
            f"Tahmin: {len(predictions)}\n"
            f"Test: {len(test_dataframe)}"
        )

    # --------------------------------------------------------
    # Tam metni çöz
    # --------------------------------------------------------

    full_text_series, text_source = (
        resolve_text_series(
            test_dataframe
        )
    )

    print(
        "Tam metin kaynağı:",
        text_source,
    )

    # --------------------------------------------------------
    # Test veri setine sabit satır numarası ekle
    # --------------------------------------------------------

    test_with_index = (
        test_dataframe
        .reset_index(
            drop=True
        )
        .copy()
    )

    test_with_index.insert(
        0,
        "original_index",
        np.arange(
            len(test_with_index),
            dtype=int,
        ),
    )

    test_with_index.insert(
        1,
        "resolved_full_text",
        full_text_series
        .reset_index(
            drop=True
        ),
    )

    original_test_columns = [
        column

        for column in test_with_index.columns

        if column not in {
            "original_index",
            "resolved_full_text",
        }
    ]

    # Tahmin sütunlarıyla çakışmayı önlemek için test
    # metadata sütunlarına test_ ön eki eklenir.
    rename_mapping = {
        column: f"test_{column}"

        for column in original_test_columns
    }

    test_with_index = (
        test_with_index
        .rename(
            columns=rename_mapping
        )
    )

    # --------------------------------------------------------
    # Tahminlerle test verisini birleştir
    # --------------------------------------------------------

    merged = predictions.merge(
        test_with_index,
        on="original_index",
        how="left",
        validate="one_to_one",
    )

    if (
        merged[
            "resolved_full_text"
        ].isna().any()
    ):
        missing_count = int(
            merged[
                "resolved_full_text"
            ]
            .isna()
            .sum()
        )

        raise RuntimeError(
            "Bazı tahmin kayıtları test metniyle "
            "birleştirilemedi.\n"
            f"Eşleşmeyen kayıt: {missing_count}"
        )

    # --------------------------------------------------------
    # False negative kayıtları
    # --------------------------------------------------------

    false_negatives = merged[
        (
            merged[
                "true_label"
            ]
            == 1
        )
        & (
            merged[
                "predicted_label"
            ]
            == 0
        )
    ].copy()

    false_negatives[
        "confidence_band"
    ] = false_negatives[
        "confidence"
    ].map(
        create_confidence_band
    )

    false_negatives[
        "token_band"
    ] = false_negatives[
        "processed_token_count"
    ].map(
        create_token_band
    )

    false_negatives[
        "review_priority"
    ] = false_negatives[
        "confidence"
    ].map(
        create_review_priority
    )

    false_negatives[
        "preview_matches_full_text"
    ] = false_negatives.apply(
        lambda row: preview_matches_full_text(
            row.get(
                "text_preview",
                "",
            ),
            row[
                "resolved_full_text"
            ],
        ),
        axis=1,
    )

    false_negatives[
        "full_text_character_count"
    ] = false_negatives[
        "resolved_full_text"
    ].astype(str).str.len()

    false_negatives[
        "model_error_type"
    ] = "fake → real"

    false_negatives = (
        false_negatives
        .sort_values(
            by=[
                "confidence",
                "real_probability",
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

    false_negatives.insert(
        0,
        "false_negative_rank",
        np.arange(
            1,
            len(false_negatives) + 1,
            dtype=int,
        ),
    )

    # --------------------------------------------------------
    # Güven özeti
    # --------------------------------------------------------

    confidence_summary = (
        false_negatives
        .groupby(
            "confidence_band",
            dropna=False,
        )
        .agg(
            false_negative_count=(
                "original_index",
                "count",
            ),

            mean_confidence=(
                "confidence",
                "mean",
            ),

            minimum_confidence=(
                "confidence",
                "min",
            ),

            maximum_confidence=(
                "confidence",
                "max",
            ),

            mean_token_count=(
                "processed_token_count",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "confidence_band"
        )
    )

    confidence_summary[
        "percentage"
    ] = (
        confidence_summary[
            "false_negative_count"
        ]
        / len(false_negatives)
    )

    # --------------------------------------------------------
    # Token özeti
    # --------------------------------------------------------

    token_summary = (
        false_negatives
        .groupby(
            "token_band",
            dropna=False,
        )
        .agg(
            false_negative_count=(
                "original_index",
                "count",
            ),

            mean_confidence=(
                "confidence",
                "mean",
            ),

            mean_real_probability=(
                "real_probability",
                "mean",
            ),

            minimum_token_count=(
                "processed_token_count",
                "min",
            ),

            maximum_token_count=(
                "processed_token_count",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            "token_band"
        )
    )

    token_summary[
        "percentage"
    ] = (
        token_summary[
            "false_negative_count"
        ]
        / len(false_negatives)
    )

    # --------------------------------------------------------
    # Metadata özeti
    # --------------------------------------------------------

    metadata_summary = (
        calculate_metadata_summary(
            false_negatives,
            original_test_columns,
        )
    )

    # --------------------------------------------------------
    # Dosyaları kaydet
    # --------------------------------------------------------

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    false_negatives.to_csv(
        FALSE_NEGATIVES_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    confidence_summary.to_csv(
        CONFIDENCE_SUMMARY_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    token_summary.to_csv(
        TOKEN_SUMMARY_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if not metadata_summary.empty:
        metadata_summary.to_csv(
            METADATA_SUMMARY_OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # Temel istatistikler
    # --------------------------------------------------------

    false_negative_count = len(
        false_negatives
    )

    mean_confidence = float(
        false_negatives[
            "confidence"
        ].mean()
    )

    median_confidence = float(
        false_negatives[
            "confidence"
        ].median()
    )

    minimum_confidence = float(
        false_negatives[
            "confidence"
        ].min()
    )

    maximum_confidence = float(
        false_negatives[
            "confidence"
        ].max()
    )

    count_above_99 = int(
        (
            false_negatives[
                "confidence"
            ]
            >= 0.99
        ).sum()
    )

    count_above_95 = int(
        (
            false_negatives[
                "confidence"
            ]
            >= 0.95
        ).sum()
    )

    count_above_90 = int(
        (
            false_negatives[
                "confidence"
            ]
            >= 0.90
        ).sum()
    )

    mean_token_count = float(
        false_negatives[
            "processed_token_count"
        ].mean()
    )

    median_token_count = float(
        false_negatives[
            "processed_token_count"
        ].median()
    )

    minimum_token_count = int(
        false_negatives[
            "processed_token_count"
        ].min()
    )

    maximum_token_count = int(
        false_negatives[
            "processed_token_count"
        ].max()
    )

    preview_match_count = int(
        false_negatives[
            "preview_matches_full_text"
        ].sum()
    )

    # --------------------------------------------------------
    # Metin özet raporu
    # --------------------------------------------------------

    summary_lines = [
        "=" * 76,
        "ELECTRA FALSE NEGATIVE ANALİZ ÖZETİ",
        "=" * 76,
        "",
        f"Test satırı: {len(test_dataframe)}",
        (
            "False negative "
            f"(gerçek fake, tahmin real): "
            f"{false_negative_count}"
        ),
        (
            "Beklenen false negative sayısı: "
            f"{EXPECTED_FALSE_NEGATIVE_COUNT}"
        ),
        "",
        "GÜVEN İSTATİSTİKLERİ",
        "-" * 76,
        (
            "Ortalama güven: "
            f"%{mean_confidence * 100:.2f}"
        ),
        (
            "Ortanca güven: "
            f"%{median_confidence * 100:.2f}"
        ),
        (
            "Minimum güven: "
            f"%{minimum_confidence * 100:.2f}"
        ),
        (
            "Maksimum güven: "
            f"%{maximum_confidence * 100:.2f}"
        ),
        (
            "%99 ve üzeri güvenli hata: "
            f"{count_above_99}"
        ),
        (
            "%95 ve üzeri güvenli hata: "
            f"{count_above_95}"
        ),
        (
            "%90 ve üzeri güvenli hata: "
            f"{count_above_90}"
        ),
        "",
        "TOKEN İSTATİSTİKLERİ",
        "-" * 76,
        (
            "Ortalama token: "
            f"{mean_token_count:.2f}"
        ),
        (
            "Ortanca token: "
            f"{median_token_count:.2f}"
        ),
        (
            "Minimum token: "
            f"{minimum_token_count}"
        ),
        (
            "Maksimum token: "
            f"{maximum_token_count}"
        ),
        "",
        "BİRLEŞTİRME KONTROLÜ",
        "-" * 76,
        (
            "Önizlemesi tam metinle eşleşen: "
            f"{preview_match_count} / "
            f"{false_negative_count}"
        ),
        "",
        "KAYDEDİLEN DOSYALAR",
        "-" * 76,
        str(
            FALSE_NEGATIVES_OUTPUT_PATH
        ),
        str(
            CONFIDENCE_SUMMARY_OUTPUT_PATH
        ),
        str(
            TOKEN_SUMMARY_OUTPUT_PATH
        ),
    ]

    if not metadata_summary.empty:
        summary_lines.append(
            str(
                METADATA_SUMMARY_OUTPUT_PATH
            )
        )

    TEXT_SUMMARY_OUTPUT_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Konsola sonuçları yazdır
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print("FALSE NEGATIVE GENEL SONUCU")
    print("=" * 76)

    print(
        "False negative:",
        false_negative_count,
    )

    if (
        false_negative_count
        == EXPECTED_FALSE_NEGATIVE_COUNT
    ):
        print(
            "Beklenen sayı kontrolü: UYUMLU"
        )

    else:
        print(
            "UYARI: Beklenen sayıdan farklı."
        )

    print()
    print(
        "Ortalama güven:",
        f"%{mean_confidence * 100:.2f}",
    )

    print(
        "Ortanca güven:",
        f"%{median_confidence * 100:.2f}",
    )

    print(
        "%99 ve üzeri güvenli hata:",
        count_above_99,
    )

    print(
        "%95 ve üzeri güvenli hata:",
        count_above_95,
    )

    print(
        "%90 ve üzeri güvenli hata:",
        count_above_90,
    )

    print()
    print(
        "Token ortalaması:",
        f"{mean_token_count:.2f}",
    )

    print(
        "Token aralığı:",
        (
            f"{minimum_token_count}–"
            f"{maximum_token_count}"
        ),
    )

    print()
    print(
        "Önizleme/tam metin eşleşmesi:",
        (
            f"{preview_match_count} / "
            f"{false_negative_count}"
        ),
    )

    print()
    print("=" * 76)
    print("GÜVEN ARALIĞI ÖZETİ")
    print("=" * 76)

    print(
        confidence_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 76)
    print("TOKEN UZUNLUĞU ÖZETİ")
    print("=" * 76)

    print(
        token_summary.to_string(
            index=False
        )
    )

    if not metadata_summary.empty:
        print()
        print("=" * 76)
        print("METADATA ÖZETİ")
        print("=" * 76)

        print(
            metadata_summary.to_string(
                index=False
            )
        )

    else:
        print()
        print(
            "Test dosyasında yöntem, kaynak veya kategori "
            "metadata sütunu bulunamadı."
        )

    print()
    print("=" * 76)
    print("EN YÜKSEK GÜVENLİ İLK 20 FALSE NEGATIVE")
    print("=" * 76)

    for row in false_negatives.head(
        20
    ).itertuples(
        index=False
    ):
        print()
        print(
            f"Sıra: {row.false_negative_rank}"
        )

        print(
            f"CSV index: {row.original_index}"
        )

        print(
            "Model güveni:",
            f"%{row.confidence * 100:.2f}",
        )

        print(
            "Token:",
            row.processed_token_count,
        )

        print(
            "Metin:",
            str(
                row.resolved_full_text
            )[:500],
        )

    print()
    print("=" * 76)
    print("DOSYALAR KAYDEDİLDİ")
    print("=" * 76)

    print(
        "Tam false negative kayıtları:",
        FALSE_NEGATIVES_OUTPUT_PATH,
    )

    print(
        "Güven özeti:",
        CONFIDENCE_SUMMARY_OUTPUT_PATH,
    )

    print(
        "Token özeti:",
        TOKEN_SUMMARY_OUTPUT_PATH,
    )

    if not metadata_summary.empty:
        print(
            "Metadata özeti:",
            METADATA_SUMMARY_OUTPUT_PATH,
        )

    print(
        "Metin raporu:",
        TEXT_SUMMARY_OUTPUT_PATH,
    )

    print()
    print(
        "False negative analizi tamamlandı."
    )


if __name__ == "__main__":
    main()