from pathlib import Path
import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kaggle_turkish_fake_real"
)

PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

ANALYSIS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)


FLAGGED_OUTPUT_PATH = (
    PROCESSED_DIRECTORY
    / "kaggle_false_claim_candidates_flagged.csv"
)

NEW_CANDIDATES_OUTPUT_PATH = (
    PROCESSED_DIRECTORY
    / "kaggle_false_claim_new_candidates.csv"
)

OVERLAP_REPORT_PATH = (
    ANALYSIS_DIRECTORY
    / "kaggle_false_claim_overlap_report.csv"
)

SUMMARY_PATH = (
    ANALYSIS_DIRECTORY
    / "kaggle_false_claim_summary.csv"
)


# Aynı veri setinin farklı hazırlık aşamalarındaki
# olası konumları. İlk bulunan dosya kullanılacak.
EXISTING_DATASET_PATHS = {
    "facturk": [
        PROJECT_ROOT
        / "data"
        / "processed"
        / "quality_flagged"
        / "facturk_training_candidates_quality_flagged.csv",

        PROJECT_ROOT
        / "data"
        / "processed"
        / "destyled"
        / "facturk_training_candidates_destyled.csv",

        PROJECT_ROOT
        / "data"
        / "processed"
        / "unified"
        / "facturk_training_candidates.csv",
    ],

    "fctr": [
        PROJECT_ROOT
        / "data"
        / "processed"
        / "quality_flagged"
        / "fctr_quality_flagged.csv",

        PROJECT_ROOT
        / "data"
        / "processed"
        / "destyled"
        / "fctr_destyled.csv",

        PROJECT_ROOT
        / "data"
        / "processed"
        / "unified"
        / "fctr_unified.csv",
    ],

    "article_main": [
        PROJECT_ROOT
        / "data"
        / "processed"
        / "quality_flagged"
        / "article_main_quality_flagged.csv",

        PROJECT_ROOT
        / "data"
        / "processed"
        / "destyled"
        / "article_main_destyled.csv",

        PROJECT_ROOT
        / "data"
        / "processed"
        / "unified"
        / "article_main_unified.csv",
    ],
}


FALSE_LABEL_VALUES = {
    "false",
    "0",
    "fake",
    "sahte",
    "yanlis",
    "yanlış",
}


# Teyit raporunda asıl iddiadan sonra başlayan
# inceleme bölümleri.
REPORT_HEADING_PATTERN = re.compile(
    r"(?i)"
    r"(?:^|\n|\r|\s{2,})"
    r"(?:"
    r"bulgular(?:ımız)?|"
    r"analiz|"
    r"inceleme|"
    r"sonuç|"
    r"doğrulama|"
    r"kanıtlar|"
    r"değerlendirme"
    r")"
    r"\s*:?\s*"
)

CLAIM_PREFIX_PATTERN = re.compile(
    r"^(?:"
    r"iddia|"
    r"iddia edilen|"
    r"kontrol edilen iddia|"
    r"sosyal medyadaki iddia"
    r")"
    r"\s*[:\-–—]\s*",
    flags=re.IGNORECASE,
)

SOURCE_BRAND_PATTERN = re.compile(
    r"\b(?:"
    r"teyit(?:\.org)?|"
    r"doğruluk\s+payı|"
    r"doğrula(?:\.org)?|"
    r"malumatfuruş"
    r")\b",
    flags=re.IGNORECASE,
)

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

MENTION_PATTERN = re.compile(
    r"(?<!\w)@[A-Za-z0-9_]+"
)

HASHTAG_PATTERN = re.compile(
    r"(?<!\w)#([^\s#]+)"
)

PUNCTUATION_PATTERN = re.compile(
    r"[^0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+"
)

TOKEN_PATTERN = re.compile(
    r"[A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+"
)


NEAR_DUPLICATE_THRESHOLD = 0.85
SHORT_TEXT_THRESHOLD = 0.90
MINIMUM_LENGTH_RATIO = 0.55
MAX_FEATURES = 60000


def normalize_unicode(text: object) -> str:
    if pd.isna(text):
        return ""

    normalized = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )

    normalized = normalized.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    return normalized.strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def normalize_column_name(column: object) -> str:
    normalized = unicodedata.normalize(
        "NFKD",
        str(column),
    )

    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )

    normalized = normalized.casefold()

    normalized = re.sub(
        r"[^a-z0-9]+",
        "",
        normalized,
    )

    return normalized


def normalize_for_duplicate_analysis(
    text: object,
) -> str:
    normalized = normalize_unicode(
        text
    ).casefold()

    normalized = URL_PATTERN.sub(
        " ",
        normalized,
    )

    normalized = MENTION_PATTERN.sub(
        " ",
        normalized,
    )

    normalized = SOURCE_BRAND_PATTERN.sub(
        " ",
        normalized,
    )

    normalized = PUNCTUATION_PATTERN.sub(
        " ",
        normalized,
    )

    return re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()


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


def find_kaggle_data_file() -> Path:
    if not RAW_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Klasör bulunamadı: {RAW_DIRECTORY}\n"
            "Kaggle arşivini bu klasöre çıkar."
        )

    supported_extensions = {
        ".xlsx",
        ".xls",
        ".csv",
    }

    candidates = [
        file_path
        for file_path in RAW_DIRECTORY.rglob("*")
        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in supported_extensions
            and not file_path.name.startswith("~$")
        )
    ]

    if not candidates:
        raise FileNotFoundError(
            "Kaggle klasöründe Excel veya CSV "
            "dosyası bulunamadı."
        )

    candidates = sorted(
        candidates,
        key=lambda path: path.stat().st_size,
        reverse=True,
    )

    return candidates[0]


def read_kaggle_dataset(
    file_path: Path,
) -> pd.DataFrame:
    suffix = file_path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(
                file_path
            )
        except ImportError as error:
            raise ImportError(
                "Excel dosyasını okumak için "
                "openpyxl kütüphanesi gerekiyor."
            ) from error

    encodings = (
        "utf-8-sig",
        "utf-8",
        "cp1254",
        "latin-1",
    )

    for encoding in encodings:
        try:
            return pd.read_csv(
                file_path,
                encoding=encoding,
                sep=None,
                engine="python",
            )
        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"CSV dosyası okunamadı: {file_path}"
    )


def detect_columns(
    dataframe: pd.DataFrame,
) -> tuple[str, str]:
    normalized_columns = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    text_candidates = (
        "haberler",
        "haber",
        "metin",
        "text",
        "content",
    )

    label_candidates = (
        "sonuc",
        "etiket",
        "label",
        "result",
    )

    text_column = next(
        (
            normalized_columns[name]
            for name in text_candidates
            if name in normalized_columns
        ),
        None,
    )

    label_column = next(
        (
            normalized_columns[name]
            for name in label_candidates
            if name in normalized_columns
        ),
        None,
    )

    if text_column is None:
        raise ValueError(
            "Haber metni sütunu bulunamadı. "
            f"Sütunlar: {list(dataframe.columns)}"
        )

    if label_column is None:
        raise ValueError(
            "Etiket sütunu bulunamadı. "
            f"Sütunlar: {list(dataframe.columns)}"
        )

    return text_column, label_column


def is_false_label(value: object) -> bool:
    normalized = normalize_unicode(
        value
    ).casefold()

    normalized_ascii = unicodedata.normalize(
        "NFKD",
        normalized,
    )

    normalized_ascii = "".join(
        character
        for character in normalized_ascii
        if unicodedata.category(character) != "Mn"
    )

    return (
        normalized in FALSE_LABEL_VALUES
        or normalized_ascii in FALSE_LABEL_VALUES
    )


def shorten_long_claim(
    text: str,
    maximum_length: int = 600,
) -> str:
    if len(text) <= maximum_length:
        return text

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    selected_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        new_length = (
            current_length
            + len(sentence)
            + 1
        )

        if (
            selected_sentences
            and new_length > maximum_length
        ):
            break

        selected_sentences.append(
            sentence
        )

        current_length = new_length

        if current_length >= maximum_length:
            break

    if selected_sentences:
        return " ".join(
            selected_sentences
        )[:maximum_length].strip()

    return text[:maximum_length].strip()


def extract_claim(
    report_text: object,
) -> tuple[str, str]:
    text = normalize_unicode(
        report_text
    )

    if not text:
        return "", "empty_report"

    split_match = REPORT_HEADING_PATTERN.search(
        text
    )

    if split_match is not None:
        claim_section = text[
            :split_match.start()
        ].strip()

        extraction_method = (
            "before_report_heading"
        )
    else:
        claim_section = text

        extraction_method = (
            "first_text_section"
        )

    nonempty_lines = [
        normalize_whitespace(line)
        for line in claim_section.split("\n")
        if normalize_whitespace(line)
    ]

    if not nonempty_lines:
        return "", "empty_claim_section"

    # Çoğu doğrulama yazısında ilk satır asıl iddia
    # veya yazının başlığıdır.
    claim = nonempty_lines[0]

    if (
        len(claim) < 20
        and len(nonempty_lines) > 1
    ):
        claim = (
            claim
            + " "
            + nonempty_lines[1]
        )

        extraction_method += (
            "_first_two_lines"
        )
    else:
        extraction_method += (
            "_first_line"
        )

    claim = CLAIM_PREFIX_PATTERN.sub(
        "",
        claim,
    )

    claim = URL_PATTERN.sub(
        "<URL>",
        claim,
    )

    claim = MENTION_PATTERN.sub(
        "<USER>",
        claim,
    )

    claim = HASHTAG_PATTERN.sub(
        lambda match: match.group(1),
        claim,
    )

    claim = SOURCE_BRAND_PATTERN.sub(
        "<SOURCE>",
        claim,
    )

    claim = normalize_whitespace(
        claim
    )

    claim = shorten_long_claim(
        claim
    )

    return claim, extraction_method


def calculate_claim_quality(
    claim: str,
) -> tuple[str, str, int]:
    tokens = TOKEN_PATTERN.findall(
        claim
    )

    meaningful_tokens = [
        token
        for token in tokens
        if len(token) >= 2
    ]

    reasons = []

    if len(claim) < 20:
        reasons.append(
            "below_20_characters"
        )

    if len(meaningful_tokens) < 3:
        reasons.append(
            "fewer_than_3_tokens"
        )

    if claim in {
        "<URL>",
        "<USER>",
        "<SOURCE>",
    }:
        reasons.append(
            "placeholder_only"
        )

    quality_status = (
        "review_required"
        if reasons
        else "quality_candidate"
    )

    return (
        quality_status,
        ";".join(reasons),
        len(meaningful_tokens),
    )


def create_kaggle_claim_dataframe(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    false_data = dataframe[
        dataframe[label_column]
        .apply(is_false_label)
    ].copy()

    rows = []

    for row_number, (
        original_index,
        row,
    ) in enumerate(
        false_data.iterrows(),
        start=1,
    ):
        original_report_text = (
            normalize_unicode(
                row[text_column]
            )
        )

        (
            claim_text,
            extraction_method,
        ) = extract_claim(
            original_report_text
        )

        (
            quality_status,
            quality_reasons,
            meaningful_token_count,
        ) = calculate_claim_quality(
            claim_text
        )

        comparison_text = (
            normalize_for_duplicate_analysis(
                claim_text
            )
        )

        exact_hash = hashlib.sha1(
            comparison_text.encode("utf-8")
        ).hexdigest()

        rows.append(
            {
                "record_id": (
                    f"kaggle_claim_{row_number:05d}"
                ),
                "dataset_name": (
                    "kaggle_turkish_fake_real"
                ),
                "task_name": (
                    "claim_veracity_multiclass"
                ),
                "text_type": (
                    "fact_checked_claim"
                ),
                "original_row_index": (
                    original_index
                ),
                "original_report_text": (
                    original_report_text
                ),
                "claim_raw": claim_text,
                "text_raw": claim_text,
                "text_basic": claim_text,
                "text_destyled": claim_text,
                "label_original": str(
                    row[label_column]
                ),
                "label_standard": "fake",
                "source": (
                    "kaggle_turkish_fake_real"
                ),
                "url": "",
                "published_date": "",
                "verification_status": (
                    "secondary_dataset_unknown_provenance"
                ),
                "label_confidence": "medium",
                "is_synthetic": False,
                "extraction_method": (
                    extraction_method
                ),
                "quality_status": (
                    quality_status
                ),
                "quality_reasons": (
                    quality_reasons
                ),
                "meaningful_token_count": (
                    meaningful_token_count
                ),
                "text_character_count": len(
                    claim_text
                ),
                "comparison_text": (
                    comparison_text
                ),
                "exact_text_hash": exact_hash,
                "split_role": (
                    "claim_candidate_pending_review"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def mark_internal_exact_duplicates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    group_sizes = (
        output
        .groupby("comparison_text")[
            "record_id"
        ]
        .transform("size")
    )

    output[
        "internal_exact_group_size"
    ] = group_sizes

    output["internal_exact_rank"] = (
        output
        .groupby(
            "comparison_text",
            sort=False,
        )
        .cumcount()
        + 1
    )

    output["internal_exact_keep"] = (
        output["internal_exact_rank"] == 1
    )

    duplicate_values = sorted(
        output.loc[
            group_sizes > 1,
            "comparison_text",
        ].unique()
    )

    duplicate_group_map = {
        value: (
            f"kaggle_claim_exact_"
            f"{index:05d}"
        )
        for index, value in enumerate(
            duplicate_values,
            start=1,
        )
    }

    output[
        "internal_exact_group_id"
    ] = (
        output["comparison_text"]
        .map(duplicate_group_map)
        .fillna("")
    )

    return output


def resolve_existing_file(
    dataset_key: str,
) -> Path:
    for file_path in (
        EXISTING_DATASET_PATHS[
            dataset_key
        ]
    ):
        if file_path.exists():
            return file_path

    searched_paths = "\n".join(
        str(path)
        for path in EXISTING_DATASET_PATHS[
            dataset_key
        ]
    )

    raise FileNotFoundError(
        f"{dataset_key} için uygun dosya "
        f"bulunamadı:\n{searched_paths}"
    )


def load_existing_datasets() -> pd.DataFrame:
    loaded = []

    for dataset_key in (
        "facturk",
        "fctr",
        "article_main",
    ):
        file_path = resolve_existing_file(
            dataset_key
        )

        dataframe = pd.read_csv(
            file_path
        )

        required_columns = {
            "record_id",
            "label_standard",
        }

        missing_columns = (
            required_columns.difference(
                dataframe.columns
            )
        )

        if missing_columns:
            raise ValueError(
                f"{file_path.name} dosyasında "
                f"eksik sütunlar: "
                f"{sorted(missing_columns)}"
            )

        text_column = next(
            (
                column
                for column in (
                    "text_destyled",
                    "text_basic",
                    "text_raw",
                )
                if column
                in dataframe.columns
            ),
            None,
        )

        if text_column is None:
            raise ValueError(
                f"{file_path.name} dosyasında "
                "kullanılabilir metin sütunu yok."
            )

        if (
            "low_information"
            in dataframe.columns
        ):
            low_information_mask = (
                parse_boolean_series(
                    dataframe[
                        "low_information"
                    ]
                )
            )

            dataframe = dataframe[
                ~low_information_mask
            ].copy()

        selected = dataframe[
            [
                "record_id",
                "label_standard",
                text_column,
            ]
        ].copy()

        selected.rename(
            columns={
                text_column: "existing_text"
            },
            inplace=True,
        )

        selected[
            "existing_dataset"
        ] = dataset_key

        selected[
            "comparison_text"
        ] = selected[
            "existing_text"
        ].apply(
            normalize_for_duplicate_analysis
        )

        selected = selected[
            selected[
                "comparison_text"
            ].str.len() > 0
        ].copy()

        loaded.append(selected)

        print(
            f"{dataset_key}: "
            f"{len(selected)} karşılaştırma kaydı "
            f"({file_path.name})"
        )

    return pd.concat(
        loaded,
        ignore_index=True,
    )


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

    return min(
        len(first_text),
        len(second_text),
    ) / maximum_length


def binary_risk_label(
    label: object,
) -> str:
    normalized = normalize_unicode(
        label
    ).casefold()

    if normalized in {
        "fake",
        "misleading",
    }:
        return "risky"

    if normalized == "real":
        return "real"

    return "uncertain"


def compare_with_existing(
    candidates: pd.DataFrame,
    existing: pd.DataFrame,
) -> pd.DataFrame:
    output = candidates.copy()

    output["exact_overlap"] = False
    output["near_overlap"] = False
    output["overlap_dataset"] = ""
    output["overlap_record_id"] = ""
    output["overlap_label"] = ""
    output["overlap_similarity"] = 0.0
    output["overlap_length_ratio"] = 0.0
    output["overlap_threshold"] = 0.0
    output["binary_label_match"] = False

    existing_exact_lookup = (
        existing
        .drop_duplicates(
            subset=["comparison_text"]
        )
        .set_index("comparison_text")
    )

    for row_index, row in output.iterrows():
        comparison_text = row[
            "comparison_text"
        ]

        if (
            comparison_text
            in existing_exact_lookup.index
        ):
            match = existing_exact_lookup.loc[
                comparison_text
            ]

            output.loc[
                row_index,
                "exact_overlap",
            ] = True

            output.loc[
                row_index,
                "overlap_dataset",
            ] = match[
                "existing_dataset"
            ]

            output.loc[
                row_index,
                "overlap_record_id",
            ] = match[
                "record_id"
            ]

            output.loc[
                row_index,
                "overlap_label",
            ] = match[
                "label_standard"
            ]

            output.loc[
                row_index,
                "overlap_similarity",
            ] = 1.0

            output.loc[
                row_index,
                "overlap_length_ratio",
            ] = 1.0

            output.loc[
                row_index,
                "binary_label_match",
            ] = (
                binary_risk_label(
                    row["label_standard"]
                )
                == binary_risk_label(
                    match[
                        "label_standard"
                    ]
                )
            )

    near_candidates = output[
        output["internal_exact_keep"]
        & output["quality_status"].eq(
            "quality_candidate"
        )
        & ~output["exact_overlap"]
    ].copy()

    if near_candidates.empty:
        return output

    combined_texts = pd.concat(
        [
            existing["comparison_text"],
            near_candidates[
                "comparison_text"
            ],
        ],
        ignore_index=True,
    )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        lowercase=False,
        norm="l2",
        dtype=np.float32,
    )

    vectorizer.fit(
        combined_texts
    )

    existing_matrix = (
        vectorizer.transform(
            existing["comparison_text"]
        )
    )

    candidate_matrix = (
        vectorizer.transform(
            near_candidates[
                "comparison_text"
            ]
        )
    )

    model = NearestNeighbors(
        n_neighbors=1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    model.fit(
        existing_matrix
    )

    distances, indexes = model.kneighbors(
        candidate_matrix,
        return_distance=True,
    )

    for position, candidate_index in enumerate(
        near_candidates.index
    ):
        nearest_existing_index = int(
            indexes[position][0]
        )

        similarity = (
            1.0
            - float(
                distances[position][0]
            )
        )

        candidate_text = output.loc[
            candidate_index,
            "comparison_text",
        ]

        existing_row = existing.iloc[
            nearest_existing_index
        ]

        length_ratio = (
            calculate_length_ratio(
                candidate_text,
                existing_row[
                    "comparison_text"
                ],
            )
        )

        threshold = (
            SHORT_TEXT_THRESHOLD
            if len(candidate_text) < 60
            else NEAR_DUPLICATE_THRESHOLD
        )

        is_near_overlap = (
            similarity >= threshold
            and length_ratio
            >= MINIMUM_LENGTH_RATIO
        )

        if not is_near_overlap:
            continue

        output.loc[
            candidate_index,
            "near_overlap",
        ] = True

        output.loc[
            candidate_index,
            "overlap_dataset",
        ] = existing_row[
            "existing_dataset"
        ]

        output.loc[
            candidate_index,
            "overlap_record_id",
        ] = existing_row[
            "record_id"
        ]

        output.loc[
            candidate_index,
            "overlap_label",
        ] = existing_row[
            "label_standard"
        ]

        output.loc[
            candidate_index,
            "overlap_similarity",
        ] = round(
            similarity,
            4,
        )

        output.loc[
            candidate_index,
            "overlap_length_ratio",
        ] = round(
            length_ratio,
            4,
        )

        output.loc[
            candidate_index,
            "overlap_threshold",
        ] = threshold

        output.loc[
            candidate_index,
            "binary_label_match",
        ] = (
            binary_risk_label(
                output.loc[
                    candidate_index,
                    "label_standard",
                ]
            )
            == binary_risk_label(
                existing_row[
                    "label_standard"
                ]
            )
        )

    return output


def assign_candidate_decisions(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    output["candidate_status"] = (
        "new_claim_candidate"
    )

    output.loc[
        ~output["internal_exact_keep"],
        "candidate_status",
    ] = "exclude_internal_exact_duplicate"

    output.loc[
        output["quality_status"].eq(
            "review_required"
        ),
        "candidate_status",
    ] = "review_low_information"

    output.loc[
        output["exact_overlap"],
        "candidate_status",
    ] = "exclude_existing_exact_overlap"

    output.loc[
        (
            ~output["exact_overlap"]
            & output["near_overlap"]
        ),
        "candidate_status",
    ] = "exclude_existing_near_overlap"

    output["new_claim_candidate"] = (
        output["candidate_status"]
        .eq("new_claim_candidate")
    )

    output.loc[
        output["new_claim_candidate"],
        "split_role",
    ] = (
        "claim_training_candidate_pending_manual_review"
    )

    return output


def create_summary(
    dataframe: pd.DataFrame,
    original_total: int,
    false_total: int,
) -> pd.DataFrame:
    status_counts = (
        dataframe[
            "candidate_status"
        ]
        .value_counts()
    )

    summary = {
        "original_dataset_records": (
            original_total
        ),
        "false_label_records": false_total,
        "extracted_claim_records": len(
            dataframe
        ),
        "quality_candidate_records": int(
            dataframe[
                "quality_status"
            ]
            .eq("quality_candidate")
            .sum()
        ),
        "review_required_records": int(
            dataframe[
                "quality_status"
            ]
            .eq("review_required")
            .sum()
        ),
        "internal_exact_duplicate_groups": int(
            dataframe.loc[
                dataframe[
                    "internal_exact_group_size"
                ].gt(1),
                "internal_exact_group_id",
            ].nunique()
        ),
        "internal_exact_duplicate_records": int(
            dataframe[
                "internal_exact_group_size"
            ].gt(1).sum()
        ),
        "existing_exact_overlaps": int(
            dataframe[
                "exact_overlap"
            ].sum()
        ),
        "existing_near_overlaps": int(
            dataframe[
                "near_overlap"
            ].sum()
        ),
        "new_claim_candidates": int(
            dataframe[
                "new_claim_candidate"
            ].sum()
        ),
    }

    for status, count in (
        status_counts.items()
    ):
        summary[
            f"status_{status}"
        ] = int(count)

    return pd.DataFrame(
        [summary]
    )


def main() -> None:
    print(
        "Kaggle False kayıtlarından "
        "iddialar çıkarılıyor..."
    )

    kaggle_file = (
        find_kaggle_data_file()
    )

    print(
        f"\nBulunan Kaggle dosyası: "
        f"{kaggle_file}"
    )

    original_dataframe = (
        read_kaggle_dataset(
            kaggle_file
        )
    )

    (
        text_column,
        label_column,
    ) = detect_columns(
        original_dataframe
    )

    print(
        f"Metin sütunu: {text_column}"
    )

    print(
        f"Etiket sütunu: {label_column}"
    )

    false_mask = (
        original_dataframe[
            label_column
        ].apply(is_false_label)
    )

    false_count = int(
        false_mask.sum()
    )

    print(
        f"Toplam kayıt: "
        f"{len(original_dataframe)}"
    )

    print(
        f"False etiketli kayıt: "
        f"{false_count}"
    )

    claims = (
        create_kaggle_claim_dataframe(
            dataframe=original_dataframe,
            text_column=text_column,
            label_column=label_column,
        )
    )

    claims = (
        mark_internal_exact_duplicates(
            claims
        )
    )

    print(
        "\nMevcut veri setleri yükleniyor..."
    )

    existing = (
        load_existing_datasets()
    )

    print(
        "\nExact ve yakın kopya "
        "karşılaştırması yapılıyor..."
    )

    claims = compare_with_existing(
        candidates=claims,
        existing=existing,
    )

    claims = assign_candidate_decisions(
        claims
    )

    summary = create_summary(
        dataframe=claims,
        original_total=len(
            original_dataframe
        ),
        false_total=false_count,
    )

    new_candidates = claims[
        claims["new_claim_candidate"]
    ].copy()

    overlap_report = claims[
        claims["exact_overlap"]
        | claims["near_overlap"]
    ].copy()

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANALYSIS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    claims.to_csv(
        FLAGGED_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    new_candidates.to_csv(
        NEW_CANDIDATES_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    overlap_report.to_csv(
        OVERLAP_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print("SONUÇ")
    print("=" * 72)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nKarar dağılımı:")

    print(
        claims[
            "candidate_status"
        ].value_counts()
    )

    if not overlap_report.empty:
        print(
            "\nÖrtüşmelerin veri "
            "setlerine göre dağılımı:"
        )

        print(
            overlap_report[
                "overlap_dataset"
            ].value_counts()
        )

        print(
            "\nÖrtüşmelerin mevcut "
            "etiketlerine göre dağılımı:"
        )

        print(
            overlap_report[
                "overlap_label"
            ].value_counts()
        )

    print("\nOluşturulan dosyalar:")

    print(
        f"- {FLAGGED_OUTPUT_PATH}"
    )

    print(
        f"- {NEW_CANDIDATES_OUTPUT_PATH}"
    )

    print(
        f"- {OVERLAP_REPORT_PATH}"
    )

    print(
        f"- {SUMMARY_PATH}"
    )

    print(
        "\nMevcut veri setlerinde hiçbir "
        "değişiklik yapılmadı."
    )

    print(
        "Yeni iddialar henüz eğitime "
        "eklenmedi; manuel inceleme "
        "adayı olarak kaydedildi."
    )


if __name__ == "__main__":
    main()