# ============================================================
# ELECTRA FALSE NEGATIVE GERÇEK–SAHTE ÇİFT KARŞILAŞTIRMASI
#
# Dosya:
# training/compare_electra_false_negative_pairs.py
#
# Amaç:
# - Modelin fake olduğu hâlde real dediği 176 kaydı almak.
# - Aynı pair_id değerine sahip gerçek haberle eşleştirmek.
# - Gerçek ve sentetik sahte metin arasındaki değişiklikleri
#   kelime/token düzeyinde çıkarmak.
# - Değişiklik boyutu ve yöntem bazında rapor oluşturmak.
# - Elle kalite kontrolü için hazır CSV ve HTML üretmek.
#
# ÖNEMLİ:
# Bu script yalnızca analiz yapar.
# Test verisini eğitim verisine eklemez.
# ============================================================

from __future__ import annotations

import difflib
import html
import re
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

FALSE_NEGATIVES_PATH = (
    REPORTS_DIRECTORY
    / "electra_false_negatives_full.csv"
)

TEST_FILE_NAME = (
    "article_test_6k.csv"
)


# ============================================================
# ÇIKTI DOSYALARI
# ============================================================

PAIR_COMPARISON_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "electra_false_negative_pair_comparison.csv"
)

METHOD_SUMMARY_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "electra_false_negative_pair_method_summary.csv"
)

EDIT_SIZE_SUMMARY_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "electra_false_negative_edit_size_summary.csv"
)

MISSING_PAIRS_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "electra_false_negative_missing_pairs.csv"
)

HTML_REPORT_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "electra_false_negative_pair_review.html"
)

TEXT_SUMMARY_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "electra_false_negative_pair_summary.txt"
)


# ============================================================
# BEKLENEN DEĞERLER
# ============================================================

EXPECTED_FALSE_NEGATIVE_COUNT = 176


# ============================================================
# CSV OKUMA
# ============================================================

def read_csv_safely(
    path: Path,
) -> pd.DataFrame:
    """
    CSV dosyasını güvenli biçimde okumayı dener.
    """

    try:
        return pd.read_csv(
            path,
            low_memory=False,
        )

    except Exception as first_error:
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                low_memory=False,
            )

        except Exception:
            raise RuntimeError(
                "CSV dosyası okunamadı:\n"
                f"{path}"
            ) from first_error


# ============================================================
# TEST DOSYASINI BUL
# ============================================================

def find_test_file() -> Path:
    """
    article_test_6k.csv dosyasını proje içerisinde bulur.
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
            "Test veri seti bulunamadı.\n"
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
# SÜTUN YARDIMCILARI
# ============================================================

def create_column_map(
    dataframe: pd.DataFrame,
) -> dict[str, str]:
    return {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Aday sütun isimlerinden veri setinde bulunanı döndürür.
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
# LABEL NORMALLEŞTİRME
# ============================================================

def normalize_label_id(
    value: Any,
) -> int:
    """
    Proje sınıf standardı:

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


# ============================================================
# TEST VERİSİNİ STANDARDİZE ET
# ============================================================

def prepare_test_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, str | None],
]:
    """
    Test veri setindeki gerekli sütunları belirler ve
    standart yardımcı sütunlar oluşturur.
    """

    pair_id_column = find_column(
        dataframe,
        [
            "pair_id",
            "original_id",
            "group_id",
            "split_group_id",
        ],
    )

    text_column = find_column(
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

    label_id_column = find_column(
        dataframe,
        [
            "label_id",
            "target",
            "class_id",
            "y",
        ],
    )

    label_column = find_column(
        dataframe,
        [
            "label",
            "label_standard",
            "class",
            "category",
        ],
    )

    method_column = find_column(
        dataframe,
        [
            "method",
            "generation_method",
            "synthetic_method",
            "manipulation_method",
            "fake_method",
        ],
    )

    source_column = find_column(
        dataframe,
        [
            "source",
            "dataset",
            "dataset_name",
            "news_source",
            "publisher",
        ],
    )

    if pair_id_column is None:
        raise ValueError(
            "Test veri setinde pair_id sütunu bulunamadı.\n"
            f"Mevcut sütunlar: {list(dataframe.columns)}"
        )

    if text_column is None:
        raise ValueError(
            "Test veri setinde metin sütunu bulunamadı.\n"
            f"Mevcut sütunlar: {list(dataframe.columns)}"
        )

    if (
        label_id_column is None
        and label_column is None
    ):
        raise ValueError(
            "Test veri setinde label veya label_id "
            "sütunu bulunamadı.\n"
            f"Mevcut sütunlar: {list(dataframe.columns)}"
        )

    prepared = dataframe.copy()

    prepared[
        "_pair_id"
    ] = (
        prepared[
            pair_id_column
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    prepared[
        "_text"
    ] = (
        prepared[
            text_column
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

    selected_label_column = (
        label_id_column
        if label_id_column is not None
        else label_column
    )

    prepared[
        "_label_id"
    ] = prepared[
        selected_label_column
    ].map(
        normalize_label_id
    )

    if method_column is not None:
        prepared[
            "_method"
        ] = (
            prepared[
                method_column
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    else:
        prepared[
            "_method"
        ] = ""

    if source_column is not None:
        prepared[
            "_source"
        ] = (
            prepared[
                source_column
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    else:
        prepared[
            "_source"
        ] = ""

    selected_columns = {
        "pair_id": pair_id_column,
        "text": text_column,
        "label_id": label_id_column,
        "label": label_column,
        "method": method_column,
        "source": source_column,
    }

    return (
        prepared,
        selected_columns,
    )


# ============================================================
# METİN TOKENIZATION
# ============================================================

TOKEN_PATTERN = re.compile(
    r"""
    [\wÇĞİÖŞÜçğıöşü]+
    (?:['’][\wÇĞİÖŞÜçğıöşü]+)*
    |
    [^\w\s]
    """,
    flags=(
        re.UNICODE
        | re.VERBOSE
    ),
)


def tokenize_text(
    text: Any,
) -> list[str]:
    """
    Türkçe kelimeleri ve noktalama işaretlerini ayrı tokenlar
    hâlinde çıkarır.
    """

    normalized_text = (
        str(text)
        .replace("\r", " ")
        .replace("\n", " ")
    )

    return TOKEN_PATTERN.findall(
        normalized_text
    )


def normalize_text_for_exact_match(
    text: Any,
) -> str:
    """
    Boşluk farklılıklarını kaldırarak tam metin kontrolü yapar.
    """

    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()


# ============================================================
# TOKENLARI OKUNABİLİR METNE ÇEVİR
# ============================================================

def join_tokens(
    tokens: list[str],
) -> str:
    if not tokens:
        return ""

    text = " ".join(
        tokens
    )

    punctuation_without_left_space = [
        ".",
        ",",
        ":",
        ";",
        "!",
        "?",
        "%",
        ")",
        "]",
        "}",
        "…",
    ]

    for punctuation in punctuation_without_left_space:
        text = text.replace(
            f" {punctuation}",
            punctuation,
        )

    punctuation_without_right_space = [
        "(",
        "[",
        "{",
    ]

    for punctuation in punctuation_without_right_space:
        text = text.replace(
            f"{punctuation} ",
            punctuation,
        )

    text = text.replace(
        " ’ ",
        "’",
    )

    text = text.replace(
        " ' ",
        "'",
    )

    return text


# ============================================================
# DEĞİŞİKLİK BOYUTU SINIFI
# ============================================================

def create_edit_size_band(
    changed_token_count: int,
    change_ratio: float,
) -> str:
    """
    Bu değer semantik kalite kararı değildir.

    Yalnızca gerçek ve sentetik metin arasındaki mekanik
    değişiklik miktarını sınıflandırır.
    """

    if changed_token_count == 0:
        return "00 — Değişiklik yok"

    if (
        changed_token_count <= 2
        or change_ratio <= 0.01
    ):
        return "01 — Çok küçük değişiklik"

    if (
        changed_token_count <= 6
        or change_ratio <= 0.03
    ):
        return "02 — Küçük değişiklik"

    if (
        changed_token_count <= 20
        or change_ratio <= 0.10
    ):
        return "03 — Orta değişiklik"

    return "04 — Büyük değişiklik"


# ============================================================
# FARK BAĞLAMI
# ============================================================

def create_difference_context(
    real_tokens: list[str],
    fake_tokens: list[str],
    opcode: tuple[
        str,
        int,
        int,
        int,
        int,
    ],
    context_size: int = 8,
) -> tuple[str, str]:
    """
    Bir değişikliğin öncesinden ve sonrasından kısa bağlam
    çıkarır.
    """

    tag, i1, i2, j1, j2 = opcode

    real_start = max(
        0,
        i1 - context_size,
    )

    real_end = min(
        len(real_tokens),
        i2 + context_size,
    )

    fake_start = max(
        0,
        j1 - context_size,
    )

    fake_end = min(
        len(fake_tokens),
        j2 + context_size,
    )

    real_context = join_tokens(
        real_tokens[
            real_start:real_end
        ]
    )

    fake_context = join_tokens(
        fake_tokens[
            fake_start:fake_end
        ]
    )

    return (
        f"[{tag}] {real_context}",
        f"[{tag}] {fake_context}",
    )


# ============================================================
# METİN ÇİFTİ ANALİZİ
# ============================================================

def compare_text_pair(
    real_text: Any,
    fake_text: Any,
) -> dict[str, Any]:
    real_text_string = str(
        real_text
    )

    fake_text_string = str(
        fake_text
    )

    real_tokens = tokenize_text(
        real_text_string
    )

    fake_tokens = tokenize_text(
        fake_text_string
    )

    matcher = difflib.SequenceMatcher(
        None,
        real_tokens,
        fake_tokens,
        autojunk=False,
    )

    opcodes = matcher.get_opcodes()

    changed_real_token_count = 0
    changed_fake_token_count = 0
    changed_token_count = 0
    difference_block_count = 0

    real_changed_spans: list[str] = []
    fake_changed_spans: list[str] = []

    real_contexts: list[str] = []
    fake_contexts: list[str] = []

    for opcode in opcodes:
        tag, i1, i2, j1, j2 = opcode

        if tag == "equal":
            continue

        difference_block_count += 1

        real_span = real_tokens[
            i1:i2
        ]

        fake_span = fake_tokens[
            j1:j2
        ]

        changed_real_token_count += len(
            real_span
        )

        changed_fake_token_count += len(
            fake_span
        )

        changed_token_count += max(
            len(real_span),
            len(fake_span),
        )

        real_changed_spans.append(
            join_tokens(
                real_span
            )
            or "∅"
        )

        fake_changed_spans.append(
            join_tokens(
                fake_span
            )
            or "∅"
        )

        real_context, fake_context = (
            create_difference_context(
                real_tokens,
                fake_tokens,
                opcode,
            )
        )

        real_contexts.append(
            real_context
        )

        fake_contexts.append(
            fake_context
        )

    maximum_token_count = max(
        len(real_tokens),
        len(fake_tokens),
        1,
    )

    change_ratio = (
        changed_token_count
        / maximum_token_count
    )

    token_similarity = float(
        matcher.ratio()
    )

    exact_text_match = (
        normalize_text_for_exact_match(
            real_text_string
        )
        ==
        normalize_text_for_exact_match(
            fake_text_string
        )
    )

    length_ratio = (
        len(fake_tokens)
        / len(real_tokens)
        if len(real_tokens) > 0
        else np.nan
    )

    edit_size_band = (
        create_edit_size_band(
            changed_token_count,
            change_ratio,
        )
    )

    return {
        "real_token_count": len(
            real_tokens
        ),

        "fake_token_count": len(
            fake_tokens
        ),

        "changed_real_token_count": (
            changed_real_token_count
        ),

        "changed_fake_token_count": (
            changed_fake_token_count
        ),

        "changed_token_count": (
            changed_token_count
        ),

        "difference_block_count": (
            difference_block_count
        ),

        "change_ratio": (
            change_ratio
        ),

        "token_similarity": (
            token_similarity
        ),

        "length_ratio": (
            length_ratio
        ),

        "exact_text_match": (
            exact_text_match
        ),

        "edit_size_band": (
            edit_size_band
        ),

        "real_changed_spans": (
            " || ".join(
                real_changed_spans
            )
        ),

        "fake_changed_spans": (
            " || ".join(
                fake_changed_spans
            )
        ),

        "real_difference_contexts": (
            " || ".join(
                real_contexts[:5]
            )
        ),

        "fake_difference_contexts": (
            " || ".join(
                fake_contexts[:5]
            )
        ),
    }


# ============================================================
# HTML FARK GÖRÜNÜMÜ
# ============================================================

def build_highlighted_pair(
    real_text: Any,
    fake_text: Any,
) -> tuple[str, str]:
    real_tokens = tokenize_text(
        real_text
    )

    fake_tokens = tokenize_text(
        fake_text
    )

    matcher = difflib.SequenceMatcher(
        None,
        real_tokens,
        fake_tokens,
        autojunk=False,
    )

    real_parts: list[str] = []
    fake_parts: list[str] = []

    for tag, i1, i2, j1, j2 in (
        matcher.get_opcodes()
    ):
        real_span = html.escape(
            join_tokens(
                real_tokens[i1:i2]
            )
        )

        fake_span = html.escape(
            join_tokens(
                fake_tokens[j1:j2]
            )
        )

        if tag == "equal":
            real_parts.append(
                real_span
            )

            fake_parts.append(
                fake_span
            )

        elif tag == "replace":
            real_parts.append(
                (
                    '<span class="removed">'
                    f"{real_span}"
                    "</span>"
                )
            )

            fake_parts.append(
                (
                    '<span class="added">'
                    f"{fake_span}"
                    "</span>"
                )
            )

        elif tag == "delete":
            real_parts.append(
                (
                    '<span class="removed">'
                    f"{real_span}"
                    "</span>"
                )
            )

        elif tag == "insert":
            fake_parts.append(
                (
                    '<span class="added">'
                    f"{fake_span}"
                    "</span>"
                )
            )

    return (
        " ".join(
            real_parts
        ),
        " ".join(
            fake_parts
        ),
    )


# ============================================================
# HTML RAPORU
# ============================================================

def create_html_report(
    comparison_dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    cards: list[str] = []

    for row in comparison_dataframe.itertuples(
        index=False
    ):
        real_highlighted, fake_highlighted = (
            build_highlighted_pair(
                row.real_text,
                row.fake_text,
            )
        )

        confidence_percentage = (
            float(row.model_real_confidence)
            * 100
        )

        change_percentage = (
            float(row.change_ratio)
            * 100
        )

        cards.append(
            f"""
            <section class="pair-card">
                <div class="pair-heading">
                    <div>
                        <strong>
                            Sıra {int(row.false_negative_rank)}
                        </strong>

                        <span>
                            Pair ID:
                            {html.escape(str(row.pair_id))}
                        </span>
                    </div>

                    <div class="badges">
                        <span class="badge method">
                            {html.escape(str(row.method))}
                        </span>

                        <span class="badge confidence">
                            Model real:
                            %{confidence_percentage:.2f}
                        </span>
                    </div>
                </div>

                <div class="metrics">
                    <span>
                        Token benzerliği:
                        %{float(row.token_similarity) * 100:.2f}
                    </span>

                    <span>
                        Değişen token:
                        {int(row.changed_token_count)}
                    </span>

                    <span>
                        Değişiklik oranı:
                        %{change_percentage:.2f}
                    </span>

                    <span>
                        Fark bloğu:
                        {int(row.difference_block_count)}
                    </span>

                    <span>
                        Boyut:
                        {html.escape(str(row.edit_size_band))}
                    </span>
                </div>

                <div class="pair-columns">
                    <article>
                        <h3>Gerçek haber</h3>

                        <p class="text-content">
                            {real_highlighted}
                        </p>
                    </article>

                    <article>
                        <h3>Sentetik sahte haber</h3>

                        <p class="text-content">
                            {fake_highlighted}
                        </p>
                    </article>
                </div>

                <div class="manual-box">
                    Manuel inceleme seçenekleri:
                    <strong>valid_hard_fake</strong>,
                    <strong>weak_transformation</strong>,
                    <strong>possibly_still_true</strong>,
                    <strong>invalid_synthetic</strong>
                </div>
            </section>
            """
        )

    html_document = f"""
<!doctype html>
<html lang="tr">
<head>
    <meta charset="utf-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >

    <title>
        ELECTRA False Negative Çift İncelemesi
    </title>

    <style>
        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 28px;
            font-family: Arial, Helvetica, sans-serif;
            color: #1f2937;
            background: #eef2f7;
        }}

        .page-header {{
            width: min(1500px, 100%);
            margin: 0 auto 24px;
            padding: 24px;
            background: white;
            border: 1px solid #dbe3ed;
            border-radius: 14px;
        }}

        .page-header h1 {{
            margin: 0 0 12px;
        }}

        .page-header p {{
            margin: 0;
            line-height: 1.6;
        }}

        .pair-card {{
            width: min(1500px, 100%);
            margin: 0 auto 22px;
            padding: 22px;
            background: white;
            border: 1px solid #dbe3ed;
            border-radius: 14px;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
        }}

        .pair-heading {{
            display: flex;
            justify-content: space-between;
            gap: 16px;
            padding-bottom: 14px;
            border-bottom: 1px solid #e5e7eb;
        }}

        .pair-heading strong,
        .pair-heading span {{
            display: block;
        }}

        .pair-heading span {{
            margin-top: 5px;
            font-size: 13px;
            color: #64748b;
        }}

        .badges {{
            display: flex;
            align-items: flex-start;
            gap: 8px;
        }}

        .badge {{
            display: inline-block;
            padding: 7px 10px;
            font-size: 12px;
            font-weight: 700;
            border-radius: 999px;
        }}

        .badge.method {{
            color: #1e3a8a;
            background: #dbeafe;
        }}

        .badge.confidence {{
            color: #991b1b;
            background: #fee2e2;
        }}

        .metrics {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px 18px;
            padding: 14px 0;
            font-size: 13px;
            color: #475569;
        }}

        .pair-columns {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
        }}

        article {{
            min-width: 0;
            padding: 18px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
        }}

        article h3 {{
            margin: 0 0 14px;
        }}

        .text-content {{
            margin: 0;
            font-size: 15px;
            line-height: 1.75;
            overflow-wrap: anywhere;
        }}

        .removed {{
            padding: 2px 3px;
            color: #991b1b;
            background: #fecaca;
            text-decoration: line-through;
            border-radius: 4px;
        }}

        .added {{
            padding: 2px 3px;
            color: #166534;
            background: #bbf7d0;
            font-weight: 700;
            border-radius: 4px;
        }}

        .manual-box {{
            margin-top: 16px;
            padding: 12px;
            font-size: 13px;
            line-height: 1.6;
            background: #fff7d6;
            border-radius: 8px;
        }}

        @media (max-width: 900px) {{
            body {{
                padding: 12px;
            }}

            .pair-columns {{
                grid-template-columns: 1fr;
            }}

            .pair-heading {{
                flex-direction: column;
            }}
        }}
    </style>
</head>

<body>
    <header class="page-header">
        <h1>
            ELECTRA False Negative Gerçek–Sahte Çift İncelemesi
        </h1>

        <p>
            Kırmızı alanlar gerçek haberden çıkarılan veya
            değiştirilen bölümleri, yeşil alanlar sentetik sahte
            habere eklenen bölümleri göstermektedir.
            Toplam eşleşen kayıt:
            <strong>{len(comparison_dataframe)}</strong>
        </p>
    </header>

    {''.join(cards)}
</body>
</html>
"""

    output_path.write_text(
        html_document,
        encoding="utf-8",
    )


# ============================================================
# YÖNTEM ÖZETİ
# ============================================================

def create_method_summary(
    comparison_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        comparison_dataframe
        .groupby(
            "method",
            dropna=False,
        )
        .agg(
            false_negative_count=(
                "pair_id",
                "count",
            ),

            mean_model_real_confidence=(
                "model_real_confidence",
                "mean",
            ),

            median_model_real_confidence=(
                "model_real_confidence",
                "median",
            ),

            mean_token_similarity=(
                "token_similarity",
                "mean",
            ),

            median_token_similarity=(
                "token_similarity",
                "median",
            ),

            mean_changed_token_count=(
                "changed_token_count",
                "mean",
            ),

            median_changed_token_count=(
                "changed_token_count",
                "median",
            ),

            mean_change_ratio=(
                "change_ratio",
                "mean",
            ),

            median_change_ratio=(
                "change_ratio",
                "median",
            ),

            exact_same_text_count=(
                "exact_text_match",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            by="false_negative_count",
            ascending=False,
        )
    )

    return summary


# ============================================================
# ANA PROGRAM
# ============================================================

def main() -> None:
    print("=" * 78)
    print("ELECTRA FALSE NEGATIVE ÇİFT KARŞILAŞTIRMASI")
    print("=" * 78)

    if not FALSE_NEGATIVES_PATH.is_file():
        raise FileNotFoundError(
            "False-negative dosyası bulunamadı:\n"
            f"{FALSE_NEGATIVES_PATH}\n\n"
            "Önce analyze_electra_false_negatives.py "
            "dosyasını çalıştır."
        )

    test_path = find_test_file()

    print(
        "False-negative dosyası:",
        FALSE_NEGATIVES_PATH,
    )

    print(
        "Test veri seti:",
        test_path,
    )

    false_negatives = read_csv_safely(
        FALSE_NEGATIVES_PATH
    )

    raw_test_dataframe = read_csv_safely(
        test_path
    )

    test_dataframe, selected_columns = (
        prepare_test_dataframe(
            raw_test_dataframe
        )
    )

    print()
    print(
        "False-negative kayıt:",
        len(false_negatives),
    )

    print(
        "Test kayıt:",
        len(test_dataframe),
    )

    print(
        "Kullanılan sütunlar:",
        selected_columns,
    )

    required_false_negative_columns = {
        "false_negative_rank",
        "original_index",
        "confidence",
        "real_probability",
        "fake_probability",
        "test_pair_id",
        "test_method",
        "resolved_full_text",
    }

    missing_false_negative_columns = (
        required_false_negative_columns
        - set(false_negatives.columns)
    )

    if missing_false_negative_columns:
        raise ValueError(
            "False-negative dosyasında eksik sütunlar:\n"
            f"{sorted(missing_false_negative_columns)}"
        )

    comparison_records: list[
        dict[str, Any]
    ] = []

    missing_records: list[
        dict[str, Any]
    ] = []

    for row in false_negatives.itertuples(
        index=False
    ):
        pair_id = str(
            row.test_pair_id
        ).strip()

        pair_rows = test_dataframe[
            test_dataframe[
                "_pair_id"
            ]
            == pair_id
        ]

        real_rows = pair_rows[
            pair_rows[
                "_label_id"
            ]
            == 0
        ]

        fake_rows = pair_rows[
            pair_rows[
                "_label_id"
            ]
            == 1
        ]

        if len(real_rows) != 1:
            missing_records.append(
                {
                    "false_negative_rank": (
                        row.false_negative_rank
                    ),

                    "original_index": (
                        row.original_index
                    ),

                    "pair_id": pair_id,

                    "real_row_count": len(
                        real_rows
                    ),

                    "fake_row_count": len(
                        fake_rows
                    ),

                    "reason": (
                        "Aynı pair_id için tam bir adet "
                        "real kayıt bulunamadı."
                    ),
                }
            )

            continue

        real_row = real_rows.iloc[0]

        real_text = str(
            real_row[
                "_text"
            ]
        )

        fake_text = str(
            row.resolved_full_text
        )

        comparison = compare_text_pair(
            real_text,
            fake_text,
        )

        method = str(
            row.test_method
        ).strip()

        if not method:
            if len(fake_rows) > 0:
                method = str(
                    fake_rows.iloc[0][
                        "_method"
                    ]
                ).strip()

        source = str(
            real_row[
                "_source"
            ]
        ).strip()

        comparison_records.append(
            {
                "false_negative_rank": int(
                    row.false_negative_rank
                ),

                "original_index": int(
                    row.original_index
                ),

                "pair_id": pair_id,

                "method": method,

                "source": source,

                "model_prediction": "real",

                "true_label": "fake",

                "model_real_confidence": float(
                    row.real_probability
                ),

                "model_fake_probability": float(
                    row.fake_probability
                ),

                "real_text": real_text,

                "fake_text": fake_text,

                **comparison,

                # Bu üç alan elle doldurulacaktır.
                "manual_quality_label": "",

                "manual_edit_type": "",

                "manual_review_notes": "",
            }
        )

    comparison_dataframe = pd.DataFrame(
        comparison_records
    )

    missing_dataframe = pd.DataFrame(
        missing_records
    )

    if comparison_dataframe.empty:
        raise RuntimeError(
            "Hiçbir gerçek–sahte çift eşleştirilemedi."
        )

    comparison_dataframe = (
        comparison_dataframe
        .sort_values(
            by=[
                "model_real_confidence",
                "false_negative_rank",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    method_summary = create_method_summary(
        comparison_dataframe
    )

    edit_size_summary = (
        comparison_dataframe
        .groupby(
            [
                "method",
                "edit_size_band",
            ],
            dropna=False,
        )
        .agg(
            pair_count=(
                "pair_id",
                "count",
            ),

            mean_model_real_confidence=(
                "model_real_confidence",
                "mean",
            ),

            mean_token_similarity=(
                "token_similarity",
                "mean",
            ),

            mean_changed_token_count=(
                "changed_token_count",
                "mean",
            ),

            mean_change_ratio=(
                "change_ratio",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            by=[
                "method",
                "edit_size_band",
            ]
        )
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_dataframe.to_csv(
        PAIR_COMPARISON_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    method_summary.to_csv(
        METHOD_SUMMARY_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    edit_size_summary.to_csv(
        EDIT_SIZE_SUMMARY_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if not missing_dataframe.empty:
        missing_dataframe.to_csv(
            MISSING_PAIRS_OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    create_html_report(
        comparison_dataframe,
        HTML_REPORT_OUTPUT_PATH,
    )

    matched_pair_count = len(
        comparison_dataframe
    )

    missing_pair_count = len(
        missing_dataframe
    )

    exact_same_text_count = int(
        comparison_dataframe[
            "exact_text_match"
        ].sum()
    )

    very_small_edit_count = int(
        (
            comparison_dataframe[
                "edit_size_band"
            ]
            ==
            "01 — Çok küçük değişiklik"
        ).sum()
    )

    small_edit_count = int(
        (
            comparison_dataframe[
                "edit_size_band"
            ]
            ==
            "02 — Küçük değişiklik"
        ).sum()
    )

    median_similarity = float(
        comparison_dataframe[
            "token_similarity"
        ].median()
    )

    median_changed_tokens = float(
        comparison_dataframe[
            "changed_token_count"
        ].median()
    )

    median_change_ratio = float(
        comparison_dataframe[
            "change_ratio"
        ].median()
    )

    summary_lines = [
        "=" * 78,
        "ELECTRA FALSE NEGATIVE GERÇEK–SAHTE ÇİFT ÖZETİ",
        "=" * 78,
        "",
        (
            "False-negative kayıt: "
            f"{len(false_negatives)}"
        ),
        (
            "Eşleşen gerçek–sahte çift: "
            f"{matched_pair_count}"
        ),
        (
            "Eşleşmeyen çift: "
            f"{missing_pair_count}"
        ),
        (
            "Gerçek ve fake metni tamamen aynı: "
            f"{exact_same_text_count}"
        ),
        "",
        (
            "Ortanca token benzerliği: "
            f"%{median_similarity * 100:.2f}"
        ),
        (
            "Ortanca değişen token: "
            f"{median_changed_tokens:.2f}"
        ),
        (
            "Ortanca değişiklik oranı: "
            f"%{median_change_ratio * 100:.2f}"
        ),
        (
            "Çok küçük değişiklik: "
            f"{very_small_edit_count}"
        ),
        (
            "Küçük değişiklik: "
            f"{small_edit_count}"
        ),
        "",
        "MANUEL KALİTE ETİKETLERİ",
        "-" * 78,
        (
            "valid_hard_fake: "
            "Yanlış bilgi üretilmiş ve metin doğal kalmış."
        ),
        (
            "weak_transformation: "
            "Değişiklik çok zayıf veya etkisiz kalmış."
        ),
        (
            "possibly_still_true: "
            "Değişiklikten sonra haber hâlâ doğru olabilir."
        ),
        (
            "invalid_synthetic: "
            "Bozuk, anlamsız veya yanlış etiketlenmiş kayıt."
        ),
        "",
        "ÇIKTI DOSYALARI",
        "-" * 78,
        str(
            PAIR_COMPARISON_OUTPUT_PATH
        ),
        str(
            METHOD_SUMMARY_OUTPUT_PATH
        ),
        str(
            EDIT_SIZE_SUMMARY_OUTPUT_PATH
        ),
        str(
            HTML_REPORT_OUTPUT_PATH
        ),
    ]

    if not missing_dataframe.empty:
        summary_lines.append(
            str(
                MISSING_PAIRS_OUTPUT_PATH
            )
        )

    TEXT_SUMMARY_OUTPUT_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("PAIR KARŞILAŞTIRMA GENEL SONUCU")
    print("=" * 78)

    print(
        "False-negative kayıt:",
        len(false_negatives),
    )

    print(
        "Eşleşen gerçek–sahte çift:",
        matched_pair_count,
    )

    print(
        "Eşleşmeyen çift:",
        missing_pair_count,
    )

    print(
        "Gerçek ve fake metni tamamen aynı:",
        exact_same_text_count,
    )

    print()
    print(
        "Ortanca token benzerliği:",
        f"%{median_similarity * 100:.2f}",
    )

    print(
        "Ortanca değişen token:",
        f"{median_changed_tokens:.2f}",
    )

    print(
        "Ortanca değişiklik oranı:",
        f"%{median_change_ratio * 100:.2f}",
    )

    print(
        "Çok küçük değişiklik:",
        very_small_edit_count,
    )

    print(
        "Küçük değişiklik:",
        small_edit_count,
    )

    print()
    print("=" * 78)
    print("YÖNTEM BAZLI ÖZET")
    print("=" * 78)

    print(
        method_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print("DEĞİŞİKLİK BOYUTU ÖZETİ")
    print("=" * 78)

    print(
        edit_size_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print("EN YÜKSEK GÜVENLİ İLK 20 ÇİFT")
    print("=" * 78)

    for row in comparison_dataframe.head(
        20
    ).itertuples(
        index=False
    ):
        print()
        print(
            "Sıra:",
            row.false_negative_rank,
        )

        print(
            "Pair ID:",
            row.pair_id,
        )

        print(
            "Yöntem:",
            row.method,
        )

        print(
            "Model real güveni:",
            f"%{row.model_real_confidence * 100:.2f}",
        )

        print(
            "Token benzerliği:",
            f"%{row.token_similarity * 100:.2f}",
        )

        print(
            "Değişen token:",
            row.changed_token_count,
        )

        print(
            "Değişiklik boyutu:",
            row.edit_size_band,
        )

        print(
            "Gerçek değişen bölüm:",
            row.real_changed_spans[:400],
        )

        print(
            "Fake değişen bölüm:",
            row.fake_changed_spans[:400],
        )

    print()
    print("=" * 78)
    print("DOSYALAR KAYDEDİLDİ")
    print("=" * 78)

    print(
        "Tam karşılaştırma:",
        PAIR_COMPARISON_OUTPUT_PATH,
    )

    print(
        "Yöntem özeti:",
        METHOD_SUMMARY_OUTPUT_PATH,
    )

    print(
        "Değişiklik boyutu özeti:",
        EDIT_SIZE_SUMMARY_OUTPUT_PATH,
    )

    print(
        "Görsel HTML raporu:",
        HTML_REPORT_OUTPUT_PATH,
    )

    if not missing_dataframe.empty:
        print(
            "Eşleşmeyen çiftler:",
            MISSING_PAIRS_OUTPUT_PATH,
        )

    print(
        "Metin özeti:",
        TEXT_SUMMARY_OUTPUT_PATH,
    )

    print()
    print(
        "Çift karşılaştırma analizi tamamlandı."
    )


if __name__ == "__main__":
    main()