from pathlib import Path
from collections import Counter
from urllib.parse import urlparse
import csv
import hashlib
import html
import random
import re
import sqlite3
import unicodedata

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "trnews_2025"
    / "TRNews-2025.csv"
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

CLEAN_OUTPUT_PATH = (
    PROCESSED_DIRECTORY
    / "trnews_2025_clean_real_pool.csv"
)

REJECTED_OUTPUT_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_2025_rejected_records.csv"
)

SUMMARY_OUTPUT_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_2025_cleaning_summary.csv"
)

REJECTION_SUMMARY_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_2025_rejection_summary.csv"
)

SOURCE_DISTRIBUTION_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_2025_clean_source_distribution.csv"
)

CLASS_DISTRIBUTION_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_2025_original_class_distribution.csv"
)

SAMPLE_OUTPUT_PATH = (
    PROCESSED_DIRECTORY
    / "trnews_2025_clean_sample.csv"
)

HASH_DATABASE_PATH = (
    PROCESSED_DIRECTORY
    / "trnews_2025_exact_hashes.sqlite"
)


CHUNK_SIZE = 50_000
MINIMUM_WORD_COUNT = 80
MAXIMUM_WORD_COUNT = 3_000
MINIMUM_ALPHABETIC_RATIO = 0.55
MINIMUM_UNIQUE_WORD_RATIO = 0.08

RANDOM_STATE = 42
MAXIMUM_SAMPLE_SIZE = 500


HTML_SCRIPT_PATTERN = re.compile(
    r"<(?:script|style)[^>]*>.*?</(?:script|style)>",
    flags=re.IGNORECASE | re.DOTALL,
)

HTML_TAG_PATTERN = re.compile(
    r"<[^>]+>"
)

URL_IN_TEXT_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

WHITESPACE_PATTERN = re.compile(
    r"\s+"
)

WORD_PATTERN = re.compile(
    r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+"
)

NORMALIZATION_PATTERN = re.compile(
    r"[^0-9a-zçğıöşüâîû<>]+"
)

BOILERPLATE_PATTERNS = [
    re.compile(
        r"çerez(?:leri)? kabul et",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"gizlilik politikası",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"üyelik sözleşmesi",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"javascript(?:i)? etkinleştirin",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"tarayıcınız desteklemiyor",
        flags=re.IGNORECASE,
    ),
]


CLEAN_COLUMNS = [
    "record_id",
    "original_id",
    "dataset_name",
    "task_name",
    "text_type",
    "text_basic",
    "label_standard",
    "label_confidence",
    "verification_status",
    "source",
    "url",
    "access_date",
    "original_class",
    "word_count",
    "character_count",
    "alphabetic_ratio",
    "unique_word_ratio",
    "exact_text_hash",
    "quality_status",
    "is_synthetic",
    "parent_record_id",
    "pair_id",
    "split_role",
]

REJECTED_COLUMNS = [
    "original_id",
    "url",
    "source",
    "access_date",
    "original_class",
    "word_count",
    "character_count",
    "rejection_reasons",
    "text_preview",
]


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

    return normalized.strip()


def clean_article_text(text: object) -> str:
    cleaned = normalize_unicode(text)

    if not cleaned:
        return ""

    cleaned = html.unescape(cleaned)

    cleaned = HTML_SCRIPT_PATTERN.sub(
        " ",
        cleaned,
    )

    cleaned = HTML_TAG_PATTERN.sub(
        " ",
        cleaned,
    )

    cleaned = URL_IN_TEXT_PATTERN.sub(
        " <URL> ",
        cleaned,
    )

    cleaned = EMAIL_PATTERN.sub(
        " <EMAIL> ",
        cleaned,
    )

    cleaned = (
        cleaned
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    cleaned = WHITESPACE_PATTERN.sub(
        " ",
        cleaned,
    )

    return cleaned.strip()


def normalize_for_exact_duplicate(
    text: str,
) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        text,
    ).casefold()

    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )

    normalized = NORMALIZATION_PATTERN.sub(
        " ",
        normalized,
    )

    return WHITESPACE_PATTERN.sub(
        " ",
        normalized,
    ).strip()


def create_exact_hash(
    normalized_text: str,
) -> str:
    return hashlib.blake2b(
        normalized_text.encode("utf-8"),
        digest_size=16,
    ).hexdigest()


def extract_domain(url: object) -> str:
    normalized_url = normalize_unicode(url)

    if not normalized_url:
        return "unknown"

    candidate = normalized_url

    if not candidate.startswith(
        ("http://", "https://")
    ):
        candidate = "https://" + candidate

    try:
        hostname = urlparse(
            candidate
        ).hostname

    except ValueError:
        return "unknown"

    if not hostname:
        return "unknown"

    hostname = hostname.casefold()

    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname


def calculate_alphabetic_ratio(
    text: str,
) -> float:
    visible_characters = [
        character
        for character in text
        if not character.isspace()
    ]

    if not visible_characters:
        return 0.0

    alphabetic_count = sum(
        character.isalpha()
        for character in visible_characters
    )

    return (
        alphabetic_count
        / len(visible_characters)
    )


def calculate_unique_word_ratio(
    words: list[str],
) -> float:
    if not words:
        return 0.0

    normalized_words = [
        word.casefold()
        for word in words
    ]

    return (
        len(set(normalized_words))
        / len(normalized_words)
    )


def detect_quality_reasons(
    text: str,
    word_count: int,
    alphabetic_ratio: float,
    unique_word_ratio: float,
) -> list[str]:
    reasons = []

    if not text:
        reasons.append(
            "empty_text"
        )

        return reasons

    if word_count < MINIMUM_WORD_COUNT:
        reasons.append(
            "below_minimum_word_count"
        )

    if word_count > MAXIMUM_WORD_COUNT:
        reasons.append(
            "above_maximum_word_count"
        )

    if (
        alphabetic_ratio
        < MINIMUM_ALPHABETIC_RATIO
    ):
        reasons.append(
            "low_alphabetic_ratio"
        )

    if (
        word_count >= 100
        and unique_word_ratio
        < MINIMUM_UNIQUE_WORD_RATIO
    ):
        reasons.append(
            "excessive_word_repetition"
        )

    boilerplate_match_count = sum(
        bool(pattern.search(text))
        for pattern in BOILERPLATE_PATTERNS
    )

    if boilerplate_match_count >= 3:
        reasons.append(
            "probable_webpage_boilerplate"
        )

    return reasons


def prepare_output_files() -> None:
    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANALYSIS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths_to_reset = [
        CLEAN_OUTPUT_PATH,
        REJECTED_OUTPUT_PATH,
        SUMMARY_OUTPUT_PATH,
        REJECTION_SUMMARY_PATH,
        SOURCE_DISTRIBUTION_PATH,
        CLASS_DISTRIBUTION_PATH,
        SAMPLE_OUTPUT_PATH,
        HASH_DATABASE_PATH,
    ]

    for file_path in paths_to_reset:
        if file_path.exists():
            file_path.unlink()


def create_hash_database() -> sqlite3.Connection:
    connection = sqlite3.connect(
        HASH_DATABASE_PATH
    )

    connection.execute(
        """
        CREATE TABLE seen_hashes (
            exact_hash TEXT PRIMARY KEY
        )
        """
    )

    connection.execute(
        "PRAGMA journal_mode=WAL"
    )

    connection.execute(
        "PRAGMA synchronous=NORMAL"
    )

    return connection


def is_new_exact_hash(
    connection: sqlite3.Connection,
    exact_hash: str,
) -> bool:
    cursor = connection.execute(
        """
        INSERT OR IGNORE INTO seen_hashes (
            exact_hash
        )
        VALUES (?)
        """,
        (exact_hash,),
    )

    return cursor.rowcount == 1


def write_csv_rows(
    file_path: Path,
    rows: list[dict],
    columns: list[str],
    write_header: bool,
) -> None:
    if not rows:
        return

    dataframe = pd.DataFrame(
        rows,
        columns=columns,
    )

    dataframe.to_csv(
        file_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )


def update_reservoir_sample(
    sample_rows: list[dict],
    row: dict,
    accepted_count: int,
    random_generator: random.Random,
) -> None:
    if len(sample_rows) < MAXIMUM_SAMPLE_SIZE:
        sample_rows.append(
            row.copy()
        )

        return

    replacement_position = (
        random_generator.randint(
            1,
            accepted_count,
        )
    )

    if replacement_position <= MAXIMUM_SAMPLE_SIZE:
        sample_rows[
            replacement_position - 1
        ] = row.copy()


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"TRNews dosyası bulunamadı: "
            f"{INPUT_PATH}"
        )

    prepare_output_files()

    hash_connection = (
        create_hash_database()
    )

    source_counter = Counter()
    original_class_counter = Counter()
    rejection_counter = Counter()

    random_generator = random.Random(
        RANDOM_STATE
    )

    sample_rows = []

    total_records = 0
    accepted_records = 0
    rejected_records = 0
    exact_duplicate_records = 0

    clean_header_written = False
    rejected_header_written = False

    reader = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8-sig",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
        on_bad_lines="skip",
        low_memory=False,
    )

    required_columns = {
        "ID",
        "Erişim Tarihi",
        "URL",
        "Haber Gövdesi",
        "Sınıf",
    }

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):
        missing_columns = (
            required_columns.difference(
                chunk.columns
            )
        )

        if missing_columns:
            raise ValueError(
                "TRNews dosyasında eksik "
                f"sütunlar var: "
                f"{sorted(missing_columns)}"
            )

        clean_rows = []
        rejected_rows = []

        for _, row in chunk.iterrows():
            total_records += 1

            original_id = normalize_unicode(
                row["ID"]
            )

            url = normalize_unicode(
                row["URL"]
            )

            access_date = normalize_unicode(
                row["Erişim Tarihi"]
            )

            original_class = (
                normalize_unicode(
                    row["Sınıf"]
                )
            )

            source = extract_domain(
                url
            )

            text = clean_article_text(
                row["Haber Gövdesi"]
            )

            words = WORD_PATTERN.findall(
                text
            )

            word_count = len(
                words
            )

            character_count = len(
                text
            )

            alphabetic_ratio = (
                calculate_alphabetic_ratio(
                    text
                )
            )

            unique_word_ratio = (
                calculate_unique_word_ratio(
                    words
                )
            )

            rejection_reasons = (
                detect_quality_reasons(
                    text=text,
                    word_count=word_count,
                    alphabetic_ratio=(
                        alphabetic_ratio
                    ),
                    unique_word_ratio=(
                        unique_word_ratio
                    ),
                )
            )

            exact_hash = ""

            if not rejection_reasons:
                normalized_text = (
                    normalize_for_exact_duplicate(
                        text
                    )
                )

                if not normalized_text:
                    rejection_reasons.append(
                        "empty_after_normalization"
                    )

                else:
                    exact_hash = (
                        create_exact_hash(
                            normalized_text
                        )
                    )

                    is_new = (
                        is_new_exact_hash(
                            connection=(
                                hash_connection
                            ),
                            exact_hash=exact_hash,
                        )
                    )

                    if not is_new:
                        rejection_reasons.append(
                            "exact_duplicate"
                        )

                        exact_duplicate_records += 1

            if rejection_reasons:
                rejected_records += 1

                for reason in rejection_reasons:
                    rejection_counter[
                        reason
                    ] += 1

                rejected_rows.append(
                    {
                        "original_id": (
                            original_id
                        ),
                        "url": url,
                        "source": source,
                        "access_date": (
                            access_date
                        ),
                        "original_class": (
                            original_class
                        ),
                        "word_count": (
                            word_count
                        ),
                        "character_count": (
                            character_count
                        ),
                        "rejection_reasons": (
                            ";".join(
                                rejection_reasons
                            )
                        ),
                        "text_preview": (
                            text[:300]
                        ),
                    }
                )

                continue

            accepted_records += 1

            record_id = (
                f"trnews_2025_"
                f"{accepted_records:07d}"
            )

            clean_row = {
                "record_id": record_id,
                "original_id": original_id,
                "dataset_name": (
                    "TRNews-2025"
                ),
                "task_name": (
                    "article_fake_news_binary"
                ),
                "text_type": (
                    "full_news_article"
                ),
                "text_basic": text,
                "label_standard": "real",
                "label_confidence": "weak",
                "verification_status": (
                    "not_fact_checked"
                ),
                "source": source,
                "url": url,
                "access_date": access_date,
                "original_class": (
                    original_class
                ),
                "word_count": word_count,
                "character_count": (
                    character_count
                ),
                "alphabetic_ratio": round(
                    alphabetic_ratio,
                    4,
                ),
                "unique_word_ratio": round(
                    unique_word_ratio,
                    4,
                ),
                "exact_text_hash": exact_hash,
                "quality_status": (
                    "accepted_exact_deduplicated"
                ),
                "is_synthetic": False,
                "parent_record_id": "",
                "pair_id": (
                    f"trnews_pair_"
                    f"{accepted_records:07d}"
                ),
                "split_role": (
                    "real_parent_candidate"
                ),
            }

            clean_rows.append(
                clean_row
            )

            source_counter[
                source
            ] += 1

            original_class_counter[
                original_class
            ] += 1

            update_reservoir_sample(
                sample_rows=sample_rows,
                row=clean_row,
                accepted_count=(
                    accepted_records
                ),
                random_generator=(
                    random_generator
                ),
            )

        write_csv_rows(
            file_path=CLEAN_OUTPUT_PATH,
            rows=clean_rows,
            columns=CLEAN_COLUMNS,
            write_header=(
                not clean_header_written
            ),
        )

        if clean_rows:
            clean_header_written = True

        write_csv_rows(
            file_path=REJECTED_OUTPUT_PATH,
            rows=rejected_rows,
            columns=REJECTED_COLUMNS,
            write_header=(
                not rejected_header_written
            ),
        )

        if rejected_rows:
            rejected_header_written = True

        hash_connection.commit()

        print(
            f"Parça {chunk_number}: "
            f"okunan={total_records:,}, "
            f"kabul={accepted_records:,}, "
            f"dışlanan={rejected_records:,}"
        )

    hash_connection.commit()
    hash_connection.close()

    summary = pd.DataFrame(
        [
            {
                "metric": (
                    "total_read_records"
                ),
                "value": total_records,
            },
            {
                "metric": (
                    "accepted_real_parent_records"
                ),
                "value": accepted_records,
            },
            {
                "metric": (
                    "rejected_records"
                ),
                "value": rejected_records,
            },
            {
                "metric": (
                    "exact_duplicate_records"
                ),
                "value": (
                    exact_duplicate_records
                ),
            },
            {
                "metric": (
                    "minimum_word_count"
                ),
                "value": (
                    MINIMUM_WORD_COUNT
                ),
            },
            {
                "metric": (
                    "maximum_word_count"
                ),
                "value": (
                    MAXIMUM_WORD_COUNT
                ),
            },
            {
                "metric": (
                    "planned_synthetic_records"
                ),
                "value": accepted_records,
            },
            {
                "metric": (
                    "planned_balanced_total_records"
                ),
                "value": (
                    accepted_records * 2
                ),
            },
        ]
    )

    rejection_summary = pd.DataFrame(
        [
            {
                "rejection_reason": reason,
                "record_count": count,
            }
            for reason, count
            in rejection_counter.most_common()
        ]
    )

    source_distribution = pd.DataFrame(
        [
            {
                "source": source,
                "record_count": count,
                "percentage": round(
                    count
                    / accepted_records
                    * 100,
                    4,
                )
                if accepted_records
                else 0.0,
            }
            for source, count
            in source_counter.most_common()
        ]
    )

    class_distribution = pd.DataFrame(
        [
            {
                "original_class": (
                    original_class
                ),
                "record_count": count,
                "percentage": round(
                    count
                    / accepted_records
                    * 100,
                    4,
                )
                if accepted_records
                else 0.0,
            }
            for original_class, count
            in original_class_counter.most_common()
        ]
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    rejection_summary.to_csv(
        REJECTION_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    source_distribution.to_csv(
        SOURCE_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    class_distribution.to_csv(
        CLASS_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        sample_rows,
        columns=CLEAN_COLUMNS,
    ).to_csv(
        SAMPLE_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 78)
    print("TRNEWS-2025 TEMİZLEME SONUCU")
    print("=" * 78)

    print(
        f"Toplam okunan: "
        f"{total_records:,}"
    )

    print(
        f"Kabul edilen gerçek haber: "
        f"{accepted_records:,}"
    )

    print(
        f"Dışlanan kayıt: "
        f"{rejected_records:,}"
    )

    print(
        f"Exact tekrar nedeniyle dışlanan: "
        f"{exact_duplicate_records:,}"
    )

    print(
        "\nPlanlanan sentetik asılsız "
        f"haber: {accepted_records:,}"
    )

    print(
        "Planlanan toplam dengeli veri: "
        f"{accepted_records * 2:,}"
    )

    print("\nDışlama nedenleri:")

    if rejection_counter:
        for reason, count in (
            rejection_counter.most_common()
        ):
            print(
                f"- {reason}: {count:,}"
            )

    else:
        print("- Dışlanan kayıt yok.")

    print("\nOluşturulan dosyalar:")

    print(
        f"- {CLEAN_OUTPUT_PATH}"
    )

    print(
        f"- {REJECTED_OUTPUT_PATH}"
    )

    print(
        f"- {SUMMARY_OUTPUT_PATH}"
    )

    print(
        f"- {REJECTION_SUMMARY_PATH}"
    )

    print(
        f"- {SOURCE_DISTRIBUTION_PATH}"
    )

    print(
        f"- {CLASS_DISTRIBUTION_PATH}"
    )

    print(
        f"- {SAMPLE_OUTPUT_PATH}"
    )

    print(
        "\nBu dosya henüz final eğitim "
        "verisi değildir. Yakın kopya "
        "kontrolünden sonra kesinleşecektir."
    )


if __name__ == "__main__":
    main()