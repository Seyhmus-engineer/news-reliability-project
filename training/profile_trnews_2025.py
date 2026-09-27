from pathlib import Path
from collections import Counter
import hashlib
import json
import random
import re
import unicodedata

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "trnews_2025"
)

EXPECTED_INPUT_PATH = (
    RAW_DIRECTORY
    / "TRNews-2025.csv"
)

ANALYSIS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)

PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


SUMMARY_OUTPUT_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_2025_profile_summary.csv"
)

COLUMN_PROFILE_OUTPUT_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_2025_column_profile.csv"
)

SOURCE_OUTPUT_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_2025_source_distribution.csv"
)

CATEGORY_OUTPUT_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_2025_category_distribution.csv"
)

YEAR_OUTPUT_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_2025_year_distribution.csv"
)

SCHEMA_OUTPUT_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_2025_detected_schema.json"
)

SAMPLE_OUTPUT_PATH = (
    PROCESSED_DIRECTORY
    / "trnews_2025_profile_sample.csv"
)


CHUNK_SIZE = 50_000
RANDOM_STATE = 42

MAX_LENGTH_SAMPLE_SIZE = 200_000
MAX_RECORD_SAMPLE_SIZE = 500

WORD_THRESHOLDS = [
    20,
    50,
    80,
    120,
    200,
]


URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

WHITESPACE_PATTERN = re.compile(
    r"\s+"
)

WORD_PATTERN = re.compile(
    r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+"
)

YEAR_PATTERN = re.compile(
    r"\b((?:19|20)\d{2})\b"
)


TITLE_COLUMN_CANDIDATES = [
    "title",
    "headline",
    "baslik",
    "haber_basligi",
    "news_title",
    "article_title",
]

CONTENT_COLUMN_CANDIDATES = [
    "content",
    "body",
    "text",
    "metin",
    "haber_metni",
    "haber_icerigi",
    "article",
    "article_content",
    "news_content",
    "description",
]

SOURCE_COLUMN_CANDIDATES = [
    "source",
    "kaynak",
    "publisher",
    "gazete",
    "site",
    "domain",
    "news_source",
]

CATEGORY_COLUMN_CANDIDATES = [
    "category",
    "kategori",
    "class",
    "topic",
    "konu",
    "section",
]

DATE_COLUMN_CANDIDATES = [
    "date",
    "tarih",
    "published_date",
    "publication_date",
    "publish_date",
    "created_at",
    "datetime",
]

URL_COLUMN_CANDIDATES = [
    "url",
    "link",
    "news_url",
    "article_url",
]


def normalize_column_name(
    column_name: object,
) -> str:
    text = unicodedata.normalize(
        "NFKD",
        str(column_name),
    )

    text = "".join(
        character
        for character in text
        if unicodedata.category(character) != "Mn"
    )

    text = text.casefold()

    return re.sub(
        r"[^a-z0-9]+",
        "",
        text,
    )


def normalize_text_for_duplicate(
    text: object,
) -> str:
    if pd.isna(text):
        return ""

    normalized = unicodedata.normalize(
        "NFKC",
        str(text),
    ).casefold()

    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )

    normalized = URL_PATTERN.sub(
        " <url> ",
        normalized,
    )

    normalized = re.sub(
        r"[^0-9a-zçğıöşüâîû<>]+",
        " ",
        normalized,
    )

    return WHITESPACE_PATTERN.sub(
        " ",
        normalized,
    ).strip()


def create_text_hash(
    normalized_text: str,
) -> int:
    digest = hashlib.blake2b(
        normalized_text.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="big",
        signed=False,
    )


def find_input_file() -> Path:
    if EXPECTED_INPUT_PATH.exists():
        return EXPECTED_INPUT_PATH

    if not RAW_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Klasör bulunamadı: {RAW_DIRECTORY}"
        )

    csv_files = [
        file_path
        for file_path in RAW_DIRECTORY.rglob("*.csv")
        if file_path.is_file()
    ]

    if not csv_files:
        raise FileNotFoundError(
            "trnews_2025 klasöründe CSV "
            "dosyası bulunamadı."
        )

    return max(
        csv_files,
        key=lambda path: path.stat().st_size,
    )


def detect_encoding(
    file_path: Path,
) -> str:
    sample_bytes = file_path.read_bytes()[
        :2_000_000
    ]

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1254",
        "windows-1254",
        "latin-1",
    ]

    for encoding in encodings:
        try:
            sample_bytes.decode(
                encoding
            )

            return encoding

        except UnicodeDecodeError:
            continue

    return "latin-1"


def detect_separator(
    file_path: Path,
    encoding: str,
) -> str:
    separators = [
        ",",
        ";",
        "\t",
        "|",
    ]

    best_separator = ","
    best_column_count = 0

    for separator in separators:
        try:
            sample = pd.read_csv(
                file_path,
                sep=separator,
                encoding=encoding,
                encoding_errors="replace",
                dtype=str,
                nrows=30,
                on_bad_lines="skip",
            )

        except Exception:
            continue

        column_count = len(
            sample.columns
        )

        if column_count > best_column_count:
            best_column_count = (
                column_count
            )

            best_separator = separator

    return best_separator


def read_schema_sample(
    file_path: Path,
    encoding: str,
    separator: str,
) -> pd.DataFrame:
    return pd.read_csv(
        file_path,
        sep=separator,
        encoding=encoding,
        encoding_errors="replace",
        dtype=str,
        nrows=2_000,
        keep_default_na=False,
        on_bad_lines="skip",
    )


def find_named_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    normalized_columns = {
        normalize_column_name(column): column
        for column in columns
    }

    normalized_candidates = [
        normalize_column_name(candidate)
        for candidate in candidates
    ]

    for candidate in normalized_candidates:
        if candidate in normalized_columns:
            return normalized_columns[
                candidate
            ]

    for candidate in normalized_candidates:
        for (
            normalized_column,
            original_column,
        ) in normalized_columns.items():
            if (
                candidate in normalized_column
                or normalized_column in candidate
            ):
                return original_column

    return None


def infer_longest_text_column(
    sample: pd.DataFrame,
    excluded_columns: set[str],
) -> str:
    average_lengths = {}

    for column in sample.columns:
        if column in excluded_columns:
            continue

        values = (
            sample[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        average_lengths[column] = float(
            values.str.len().mean()
        )

    if not average_lengths:
        raise ValueError(
            "Metin sütunu otomatik "
            "olarak belirlenemedi."
        )

    return max(
        average_lengths,
        key=average_lengths.get,
    )


def detect_schema(
    sample: pd.DataFrame,
) -> dict:
    columns = list(
        sample.columns
    )

    source_column = find_named_column(
        columns,
        SOURCE_COLUMN_CANDIDATES,
    )

    category_column = find_named_column(
        columns,
        CATEGORY_COLUMN_CANDIDATES,
    )

    date_column = find_named_column(
        columns,
        DATE_COLUMN_CANDIDATES,
    )

    url_column = find_named_column(
        columns,
        URL_COLUMN_CANDIDATES,
    )

    title_column = find_named_column(
        columns,
        TITLE_COLUMN_CANDIDATES,
    )

    content_column = find_named_column(
        columns,
        CONTENT_COLUMN_CANDIDATES,
    )

    excluded_for_content = {
        column
        for column in [
            source_column,
            category_column,
            date_column,
            url_column,
            title_column,
        ]
        if column is not None
    }

    if content_column is None:
        content_column = (
            infer_longest_text_column(
                sample=sample,
                excluded_columns=(
                    excluded_for_content
                ),
            )
        )

    if (
        title_column is not None
        and title_column == content_column
    ):
        title_column = None

    return {
        "title_column": title_column,
        "content_column": content_column,
        "source_column": source_column,
        "category_column": category_column,
        "date_column": date_column,
        "url_column": url_column,
        "all_columns": columns,
    }


def create_combined_text(
    chunk: pd.DataFrame,
    title_column: str | None,
    content_column: str,
) -> pd.Series:
    content = (
        chunk[content_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if title_column is None:
        return content

    title = (
        chunk[title_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    combined = np.where(
        title.ne("") & content.ne(""),
        title + "\n" + content,
        title + content,
    )

    return pd.Series(
        combined,
        index=chunk.index,
        dtype=str,
    )


def update_distribution_counter(
    counter: Counter,
    series: pd.Series,
) -> None:
    values = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )

    counter.update(
        value
        for value in values
        if value
    )


def update_reservoir_sample(
    reservoir: list,
    value: tuple[int, int],
    item_number: int,
    random_generator: random.Random,
) -> None:
    if len(reservoir) < (
        MAX_LENGTH_SAMPLE_SIZE
    ):
        reservoir.append(
            value
        )

        return

    replacement_index = (
        random_generator.randint(
            1,
            item_number,
        )
    )

    if (
        replacement_index
        <= MAX_LENGTH_SAMPLE_SIZE
    ):
        reservoir[
            replacement_index - 1
        ] = value


def counter_to_dataframe(
    counter: Counter,
    value_column_name: str,
    total_records: int,
) -> pd.DataFrame:
    rows = []

    for value, count in counter.most_common():
        rows.append(
            {
                value_column_name: value,
                "record_count": int(count),
                "percentage": round(
                    (
                        count
                        / total_records
                        * 100
                    )
                    if total_records
                    else 0.0,
                    4,
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def safe_percentage(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return round(
        numerator
        / denominator
        * 100,
        4,
    )


def main() -> None:
    print(
        "TRNews-2025 veri profili "
        "hazırlanıyor..."
    )

    input_path = find_input_file()

    file_size_bytes = (
        input_path.stat().st_size
    )

    print(
        f"\nDosya: {input_path}"
    )

    print(
        "Dosya boyutu: "
        f"{file_size_bytes / 1024 ** 3:.2f} GB"
    )

    encoding = detect_encoding(
        input_path
    )

    separator = detect_separator(
        file_path=input_path,
        encoding=encoding,
    )

    print(
        f"Algılanan encoding: {encoding}"
    )

    visible_separator = {
        "\t": "\\t",
        ",": ",",
        ";": ";",
        "|": "|",
    }.get(
        separator,
        separator,
    )

    print(
        f"Algılanan ayraç: {visible_separator}"
    )

    schema_sample = read_schema_sample(
        file_path=input_path,
        encoding=encoding,
        separator=separator,
    )

    schema = detect_schema(
        schema_sample
    )

    print("\nAlgılanan sütunlar:")

    for column in schema[
        "all_columns"
    ]:
        print(
            f"- {column}"
        )

    print("\nAlgılanan veri şeması:")

    for key in [
        "title_column",
        "content_column",
        "source_column",
        "category_column",
        "date_column",
        "url_column",
    ]:
        print(
            f"{key}: {schema[key]}"
        )

    title_column = schema[
        "title_column"
    ]

    content_column = schema[
        "content_column"
    ]

    source_column = schema[
        "source_column"
    ]

    category_column = schema[
        "category_column"
    ]

    date_column = schema[
        "date_column"
    ]

    source_counter = Counter()
    category_counter = Counter()
    year_counter = Counter()

    column_missing_counts = Counter()
    column_nonempty_counts = Counter()

    seen_text_hashes = set()

    total_records = 0
    nonempty_text_records = 0
    empty_text_records = 0
    exact_duplicate_records = 0
    unique_text_records = 0

    word_threshold_counts = {
        threshold: 0
        for threshold in WORD_THRESHOLDS
    }

    unique_word_threshold_counts = {
        threshold: 0
        for threshold in WORD_THRESHOLDS
    }

    length_reservoir = []
    sampled_record_parts = []

    random_generator = random.Random(
        RANDOM_STATE
    )

    processed_text_number = 0
    chunk_number = 0

    reader = pd.read_csv(
        input_path,
        sep=separator,
        encoding=encoding,
        encoding_errors="replace",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
        on_bad_lines="skip",
        low_memory=False,
    )

    for chunk in reader:
        chunk_number += 1

        chunk_record_count = len(
            chunk
        )

        if chunk_record_count == 0:
            continue

        total_records += (
            chunk_record_count
        )

        for column in chunk.columns:
            values = (
                chunk[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            missing_count = int(
                values.eq("").sum()
            )

            column_missing_counts[
                column
            ] += missing_count

            column_nonempty_counts[
                column
            ] += (
                chunk_record_count
                - missing_count
            )

        combined_texts = (
            create_combined_text(
                chunk=chunk,
                title_column=title_column,
                content_column=content_column,
            )
        )

        if source_column is not None:
            update_distribution_counter(
                source_counter,
                chunk[source_column],
            )

        if category_column is not None:
            update_distribution_counter(
                category_counter,
                chunk[category_column],
            )

        if date_column is not None:
            date_values = (
                chunk[date_column]
                .fillna("")
                .astype(str)
            )

            years = date_values.str.extract(
                YEAR_PATTERN,
                expand=False,
            )

            year_counter.update(
                year
                for year in years
                if (
                    isinstance(year, str)
                    and year
                )
            )

        for text in combined_texts:
            processed_text_number += 1

            text = str(text).strip()

            if not text:
                empty_text_records += 1
                continue

            nonempty_text_records += 1

            character_count = len(
                text
            )

            word_count = len(
                WORD_PATTERN.findall(
                    text
                )
            )

            for threshold in (
                WORD_THRESHOLDS
            ):
                if word_count >= threshold:
                    word_threshold_counts[
                        threshold
                    ] += 1

            normalized_text = (
                normalize_text_for_duplicate(
                    text
                )
            )

            if not normalized_text:
                continue

            text_hash = create_text_hash(
                normalized_text
            )

            is_new_text = (
                text_hash
                not in seen_text_hashes
            )

            if is_new_text:
                seen_text_hashes.add(
                    text_hash
                )

                unique_text_records += 1

                for threshold in (
                    WORD_THRESHOLDS
                ):
                    if word_count >= threshold:
                        unique_word_threshold_counts[
                            threshold
                        ] += 1

            else:
                exact_duplicate_records += 1

            update_reservoir_sample(
                reservoir=length_reservoir,
                value=(
                    character_count,
                    word_count,
                ),
                item_number=(
                    processed_text_number
                ),
                random_generator=(
                    random_generator
                ),
            )

        remaining_sample_capacity = (
            MAX_RECORD_SAMPLE_SIZE
            - sum(
                len(part)
                for part in sampled_record_parts
            )
        )

        if remaining_sample_capacity > 0:
            sample_size = min(
                20,
                chunk_record_count,
                remaining_sample_capacity,
            )

            chunk_sample = chunk.sample(
                n=sample_size,
                random_state=(
                    RANDOM_STATE
                    + chunk_number
                ),
            ).copy()

            chunk_sample[
                "_profile_combined_text"
            ] = create_combined_text(
                chunk=chunk_sample,
                title_column=title_column,
                content_column=content_column,
            )

            for column in (
                chunk_sample.columns
            ):
                chunk_sample[column] = (
                    chunk_sample[column]
                    .fillna("")
                    .astype(str)
                    .str.slice(
                        0,
                        2_000,
                    )
                )

            sampled_record_parts.append(
                chunk_sample
            )

        print(
            f"Parça {chunk_number}: "
            f"toplam okunan kayıt "
            f"{total_records:,}"
        )

    length_array = np.array(
        length_reservoir,
        dtype=np.int64,
    )

    if len(length_array) > 0:
        character_lengths = (
            length_array[:, 0]
        )

        word_lengths = (
            length_array[:, 1]
        )

        character_mean = round(
            float(
                character_lengths.mean()
            ),
            2,
        )

        word_mean = round(
            float(
                word_lengths.mean()
            ),
            2,
        )

        character_quantiles = (
            np.quantile(
                character_lengths,
                [
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99,
                ],
            )
        )

        word_quantiles = np.quantile(
            word_lengths,
            [
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ],
        )

    else:
        character_mean = 0.0
        word_mean = 0.0

        character_quantiles = np.zeros(
            6
        )

        word_quantiles = np.zeros(
            6
        )

    summary = {
        "input_file": str(
            input_path
        ),
        "file_size_bytes": (
            file_size_bytes
        ),
        "file_size_gb": round(
            file_size_bytes
            / 1024 ** 3,
            4,
        ),
        "encoding": encoding,
        "separator": (
            visible_separator
        ),
        "detected_title_column": (
            title_column or ""
        ),
        "detected_content_column": (
            content_column or ""
        ),
        "detected_source_column": (
            source_column or ""
        ),
        "detected_category_column": (
            category_column or ""
        ),
        "detected_date_column": (
            date_column or ""
        ),
        "total_read_records": (
            total_records
        ),
        "nonempty_text_records": (
            nonempty_text_records
        ),
        "empty_text_records": (
            empty_text_records
        ),
        "unique_normalized_text_records": (
            unique_text_records
        ),
        "exact_duplicate_records": (
            exact_duplicate_records
        ),
        "exact_duplicate_percentage": (
            safe_percentage(
                exact_duplicate_records,
                nonempty_text_records,
            )
        ),
        "average_character_count": (
            character_mean
        ),
        "average_word_count": (
            word_mean
        ),
        "character_q25": int(
            character_quantiles[0]
        ),
        "character_q50": int(
            character_quantiles[1]
        ),
        "character_q75": int(
            character_quantiles[2]
        ),
        "character_q90": int(
            character_quantiles[3]
        ),
        "character_q95": int(
            character_quantiles[4]
        ),
        "character_q99": int(
            character_quantiles[5]
        ),
        "word_q25": int(
            word_quantiles[0]
        ),
        "word_q50": int(
            word_quantiles[1]
        ),
        "word_q75": int(
            word_quantiles[2]
        ),
        "word_q90": int(
            word_quantiles[3]
        ),
        "word_q95": int(
            word_quantiles[4]
        ),
        "word_q99": int(
            word_quantiles[5]
        ),
    }

    for threshold in WORD_THRESHOLDS:
        summary[
            f"records_at_least_"
            f"{threshold}_words"
        ] = word_threshold_counts[
            threshold
        ]

        summary[
            f"unique_records_at_least_"
            f"{threshold}_words"
        ] = (
            unique_word_threshold_counts[
                threshold
            ]
        )

    summary_dataframe = pd.DataFrame(
        [
            {
                "metric": metric,
                "value": value,
            }
            for metric, value
            in summary.items()
        ]
    )

    column_profile_rows = []

    for column in schema[
        "all_columns"
    ]:
        missing_count = int(
            column_missing_counts[
                column
            ]
        )

        nonempty_count = int(
            column_nonempty_counts[
                column
            ]
        )

        column_profile_rows.append(
            {
                "column_name": column,
                "missing_count": (
                    missing_count
                ),
                "nonempty_count": (
                    nonempty_count
                ),
                "missing_percentage": (
                    safe_percentage(
                        missing_count,
                        total_records,
                    )
                ),
            }
        )

    column_profile = pd.DataFrame(
        column_profile_rows
    )

    source_distribution = (
        counter_to_dataframe(
            counter=source_counter,
            value_column_name="source",
            total_records=total_records,
        )
    )

    category_distribution = (
        counter_to_dataframe(
            counter=category_counter,
            value_column_name="category",
            total_records=total_records,
        )
    )

    year_distribution = (
        counter_to_dataframe(
            counter=year_counter,
            value_column_name="year",
            total_records=total_records,
        )
    )

    if sampled_record_parts:
        sample_dataframe = pd.concat(
            sampled_record_parts,
            ignore_index=True,
        )

    else:
        sample_dataframe = pd.DataFrame()

    ANALYSIS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_dataframe.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    column_profile.to_csv(
        COLUMN_PROFILE_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    source_distribution.to_csv(
        SOURCE_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    category_distribution.to_csv(
        CATEGORY_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    year_distribution.to_csv(
        YEAR_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    sample_dataframe.to_csv(
        SAMPLE_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    schema_document = {
        "input_file": str(
            input_path
        ),
        "encoding": encoding,
        "separator": separator,
        **schema,
    }

    with open(
        SCHEMA_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as schema_file:
        json.dump(
            schema_document,
            schema_file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 76)
    print("TRNEWS-2025 PROFİL SONUCU")
    print("=" * 76)

    print(
        f"Toplam okunan kayıt: "
        f"{total_records:,}"
    )

    print(
        f"Boş olmayan metin: "
        f"{nonempty_text_records:,}"
    )

    print(
        f"Boş metin: "
        f"{empty_text_records:,}"
    )

    print(
        f"Benzersiz normalize metin: "
        f"{unique_text_records:,}"
    )

    print(
        f"Exact tekrar kayıtları: "
        f"{exact_duplicate_records:,}"
    )

    print(
        f"Ortalama kelime sayısı: "
        f"{word_mean}"
    )

    print("\nKelime eşiğine göre kayıtlar:")

    for threshold in WORD_THRESHOLDS:
        print(
            f"En az {threshold} kelime: "
            f"{word_threshold_counts[threshold]:,}"
            " | benzersiz: "
            f"{unique_word_threshold_counts[threshold]:,}"
        )

    print("\nOluşturulan raporlar:")

    print(
        f"- {SUMMARY_OUTPUT_PATH}"
    )

    print(
        f"- {COLUMN_PROFILE_OUTPUT_PATH}"
    )

    print(
        f"- {SOURCE_OUTPUT_PATH}"
    )

    print(
        f"- {CATEGORY_OUTPUT_PATH}"
    )

    print(
        f"- {YEAR_OUTPUT_PATH}"
    )

    print(
        f"- {SCHEMA_OUTPUT_PATH}"
    )

    print(
        f"- {SAMPLE_OUTPUT_PATH}"
    )

    print(
        "\nBu aşamada herhangi bir kayıt "
        "silinmedi veya eğitim verisine eklenmedi."
    )


if __name__ == "__main__":
    main()