from array import array
from collections import Counter
from pathlib import Path
import csv
import math
import re
import unicodedata
import zlib

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trnews_2025_clean_real_pool.csv"
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


FINAL_POOL_PATH = (
    PROCESSED_DIRECTORY
    / "trnews_2025_final_real_pool.csv"
)

NEAR_DUPLICATE_REPORT_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_2025_near_duplicate_records.csv"
)

SUMMARY_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_2025_final_pool_summary.csv"
)

SOURCE_DISTRIBUTION_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_2025_final_source_distribution.csv"
)

CLASS_DISTRIBUTION_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_2025_final_class_distribution.csv"
)

SAMPLE_PATH = (
    PROCESSED_DIRECTORY
    / "trnews_2025_final_pool_sample.csv"
)


CHUNK_SIZE = 10_000

SHINGLE_SIZE = 3
MAX_SHINGLES_PER_TEXT = 64

SIMHASH_BIT_COUNT = 64
BAND_COUNT = 4
BITS_PER_BAND = 16

MAX_HAMMING_DISTANCE = 5
MINIMUM_LENGTH_RATIO = 0.78

# Çok kalabalık bir band oluşursa yalnızca
# en yakın zamanda eklenen temsilciler incelenir.
MAX_BUCKET_ITEMS_TO_CHECK = 150

MAXIMUM_SAMPLE_SIZE = 500
RANDOM_STATE = 42


WORD_PATTERN = re.compile(
    r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+"
)

RECORD_NUMBER_PATTERN = re.compile(
    r"(\d+)$"
)


FINAL_EXTRA_COLUMNS = [
    "simhash64_hex",
    "near_duplicate_status",
    "near_duplicate_representative_id",
    "final_parent_index",
    "final_pair_id",
]

NEAR_DUPLICATE_COLUMNS = [
    "record_id",
    "representative_record_id",
    "source",
    "original_class",
    "word_count",
    "representative_word_count",
    "hamming_distance",
    "length_ratio",
    "simhash64_hex",
    "representative_simhash64_hex",
    "text_preview",
]


def normalize_text(
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

    return " ".join(
        WORD_PATTERN.findall(
            normalized
        )
    )


def select_shingles(
    tokens: list[str],
) -> list[str]:
    if not tokens:
        return []

    if len(tokens) < SHINGLE_SIZE:
        return tokens

    available_shingle_count = (
        len(tokens)
        - SHINGLE_SIZE
        + 1
    )

    if (
        available_shingle_count
        <= MAX_SHINGLES_PER_TEXT
    ):
        indexes = range(
            available_shingle_count
        )

    else:
        indexes = np.linspace(
            0,
            available_shingle_count - 1,
            num=MAX_SHINGLES_PER_TEXT,
            dtype=np.int64,
        )

    return [
        " ".join(
            tokens[
                index:index + SHINGLE_SIZE
            ]
        )
        for index in indexes
    ]


def create_feature_hash(
    feature: str,
) -> int:
    encoded = feature.encode(
        "utf-8",
        errors="ignore",
    )

    first_half = (
        zlib.crc32(encoded)
        & 0xFFFFFFFF
    )

    second_half = (
        zlib.crc32(
            encoded,
            0x9E3779B9,
        )
        & 0xFFFFFFFF
    )

    return (
        first_half << 32
    ) | second_half


def calculate_simhash(
    text: object,
) -> tuple[int, int]:
    normalized = normalize_text(
        text
    )

    tokens = normalized.split()

    word_count = len(tokens)

    shingles = select_shingles(
        tokens
    )

    if not shingles:
        return 0, word_count

    feature_hashes = np.asarray(
        [
            create_feature_hash(shingle)
            for shingle in shingles
        ],
        dtype=">u8",
    )

    bit_matrix = np.unpackbits(
        feature_hashes.view(np.uint8)
    ).reshape(
        -1,
        SIMHASH_BIT_COUNT,
    )

    one_counts = bit_matrix.sum(
        axis=0
    )

    majority_bits = (
        one_counts * 2
        >= len(feature_hashes)
    ).astype(
        np.uint8
    )

    packed = np.packbits(
        majority_bits
    )

    simhash = int.from_bytes(
        packed.tobytes(),
        byteorder="big",
        signed=False,
    )

    return simhash, word_count


def calculate_length_ratio(
    first_length: int,
    second_length: int,
) -> float:
    maximum_length = max(
        first_length,
        second_length,
    )

    if maximum_length == 0:
        return 0.0

    return (
        min(
            first_length,
            second_length,
        )
        / maximum_length
    )


def get_band_keys(
    simhash: int,
) -> list[int]:
    keys = []

    band_mask = (
        1 << BITS_PER_BAND
    ) - 1

    for band_index in range(
        BAND_COUNT
    ):
        band_value = (
            simhash
            >> (
                band_index
                * BITS_PER_BAND
            )
        ) & band_mask

        key = (
            band_index
            << BITS_PER_BAND
        ) | band_value

        keys.append(
            key
        )

    return keys


def parse_record_number(
    record_id: str,
) -> int:
    match = RECORD_NUMBER_PATTERN.search(
        record_id
    )

    if match is None:
        raise ValueError(
            f"record_id sayısal ek içermiyor: "
            f"{record_id}"
        )

    return int(
        match.group(1)
    )


def create_record_id(
    record_number: int,
) -> str:
    return (
        f"trnews_2025_"
        f"{record_number:07d}"
    )


def write_dataframe_chunk(
    dataframe: pd.DataFrame,
    output_path: Path,
    write_header: bool,
) -> None:
    if dataframe.empty:
        return

    dataframe.to_csv(
        output_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
        encoding=(
            "utf-8-sig"
            if write_header
            else "utf-8"
        ),
        quoting=csv.QUOTE_MINIMAL,
    )


def get_bucket_candidates(
    buckets: dict[int, array],
    simhash: int,
) -> set[int]:
    candidates = set()

    for band_key in get_band_keys(
        simhash
    ):
        bucket = buckets.get(
            band_key
        )

        if bucket is None:
            continue

        if (
            len(bucket)
            > MAX_BUCKET_ITEMS_TO_CHECK
        ):
            selected_indexes = bucket[
                -MAX_BUCKET_ITEMS_TO_CHECK:
            ]

        else:
            selected_indexes = bucket

        candidates.update(
            selected_indexes
        )

    return candidates


def find_near_duplicate(
    simhash: int,
    word_count: int,
    candidate_indexes: set[int],
    representative_hashes: array,
    representative_word_counts: array,
) -> tuple[int | None, int, float]:
    best_candidate = None
    best_hamming_distance = (
        SIMHASH_BIT_COUNT + 1
    )
    best_length_ratio = 0.0

    for candidate_index in (
        candidate_indexes
    ):
        candidate_hash = int(
            representative_hashes[
                candidate_index
            ]
        )

        hamming_distance = (
            simhash ^ candidate_hash
        ).bit_count()

        if (
            hamming_distance
            > MAX_HAMMING_DISTANCE
        ):
            continue

        candidate_word_count = int(
            representative_word_counts[
                candidate_index
            ]
        )

        length_ratio = (
            calculate_length_ratio(
                word_count,
                candidate_word_count,
            )
        )

        if (
            length_ratio
            < MINIMUM_LENGTH_RATIO
        ):
            continue

        is_better = (
            hamming_distance
            < best_hamming_distance
            or (
                hamming_distance
                == best_hamming_distance
                and length_ratio
                > best_length_ratio
            )
        )

        if is_better:
            best_candidate = (
                candidate_index
            )

            best_hamming_distance = (
                hamming_distance
            )

            best_length_ratio = (
                length_ratio
            )

    return (
        best_candidate,
        best_hamming_distance,
        best_length_ratio,
    )


def add_representative_to_buckets(
    buckets: dict[int, array],
    simhash: int,
    representative_index: int,
) -> None:
    for band_key in get_band_keys(
        simhash
    ):
        bucket = buckets.get(
            band_key
        )

        if bucket is None:
            buckets[band_key] = array(
                "I",
                [representative_index],
            )

        else:
            bucket.append(
                representative_index
            )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Temiz TRNews dosyası bulunamadı: "
            f"{INPUT_PATH}"
        )

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANALYSIS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = [
        FINAL_POOL_PATH,
        NEAR_DUPLICATE_REPORT_PATH,
        SUMMARY_PATH,
        SOURCE_DISTRIBUTION_PATH,
        CLASS_DISTRIBUTION_PATH,
        SAMPLE_PATH,
    ]

    for output_path in output_paths:
        if output_path.exists():
            output_path.unlink()

    # Her temsilci için yalnızca gerekli sayısal
    # bilgiler tutulur. Böylece bellek kullanımı
    # azaltılır.
    representative_hashes = array(
        "Q"
    )

    representative_word_counts = array(
        "I"
    )

    representative_record_numbers = array(
        "I"
    )

    buckets: dict[int, array] = {}

    source_counter = Counter()
    class_counter = Counter()

    random_generator = np.random.default_rng(
        RANDOM_STATE
    )

    sample_rows = []

    total_records = 0
    final_unique_records = 0
    near_duplicate_records = 0

    final_header_written = False
    duplicate_header_written = False

    input_columns = None

    reader = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8-sig",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
        on_bad_lines="skip",
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):
        if input_columns is None:
            input_columns = list(
                chunk.columns
            )

            required_columns = {
                "record_id",
                "text_basic",
                "source",
                "original_class",
                "word_count",
            }

            missing_columns = (
                required_columns.difference(
                    input_columns
                )
            )

            if missing_columns:
                raise ValueError(
                    "Temiz TRNews dosyasında "
                    f"eksik sütunlar var: "
                    f"{sorted(missing_columns)}"
                )

        column_indexes = {
            column: input_columns.index(
                column
            )
            for column in [
                "record_id",
                "text_basic",
                "source",
                "original_class",
                "word_count",
            ]
        }

        final_rows = []
        duplicate_rows = []

        for values in chunk.itertuples(
            index=False,
            name=None,
        ):
            total_records += 1

            record_id = str(
                values[
                    column_indexes[
                        "record_id"
                    ]
                ]
            )

            text = str(
                values[
                    column_indexes[
                        "text_basic"
                    ]
                ]
            )

            source = str(
                values[
                    column_indexes[
                        "source"
                    ]
                ]
            )

            original_class = str(
                values[
                    column_indexes[
                        "original_class"
                    ]
                ]
            )

            stored_word_count_text = str(
                values[
                    column_indexes[
                        "word_count"
                    ]
                ]
            )

            try:
                stored_word_count = int(
                    float(
                        stored_word_count_text
                    )
                )

            except ValueError:
                stored_word_count = 0

            (
                simhash,
                calculated_word_count,
            ) = calculate_simhash(
                text
            )

            word_count = (
                stored_word_count
                if stored_word_count > 0
                else calculated_word_count
            )

            candidate_indexes = (
                get_bucket_candidates(
                    buckets=buckets,
                    simhash=simhash,
                )
            )

            (
                matching_index,
                hamming_distance,
                length_ratio,
            ) = find_near_duplicate(
                simhash=simhash,
                word_count=word_count,
                candidate_indexes=(
                    candidate_indexes
                ),
                representative_hashes=(
                    representative_hashes
                ),
                representative_word_counts=(
                    representative_word_counts
                ),
            )

            if matching_index is not None:
                near_duplicate_records += 1

                representative_number = int(
                    representative_record_numbers[
                        matching_index
                    ]
                )

                representative_id = (
                    create_record_id(
                        representative_number
                    )
                )

                representative_hash = int(
                    representative_hashes[
                        matching_index
                    ]
                )

                representative_word_count = int(
                    representative_word_counts[
                        matching_index
                    ]
                )

                duplicate_rows.append(
                    {
                        "record_id": record_id,
                        "representative_record_id": (
                            representative_id
                        ),
                        "source": source,
                        "original_class": (
                            original_class
                        ),
                        "word_count": (
                            word_count
                        ),
                        "representative_word_count": (
                            representative_word_count
                        ),
                        "hamming_distance": (
                            hamming_distance
                        ),
                        "length_ratio": round(
                            length_ratio,
                            4,
                        ),
                        "simhash64_hex": (
                            f"{simhash:016x}"
                        ),
                        "representative_simhash64_hex": (
                            f"{representative_hash:016x}"
                        ),
                        "text_preview": (
                            text[:300]
                        ),
                    }
                )

                continue

            representative_index = len(
                representative_hashes
            )

            representative_hashes.append(
                simhash
            )

            representative_word_counts.append(
                word_count
            )

            record_number = (
                parse_record_number(
                    record_id
                )
            )

            representative_record_numbers.append(
                record_number
            )

            add_representative_to_buckets(
                buckets=buckets,
                simhash=simhash,
                representative_index=(
                    representative_index
                ),
            )

            final_unique_records += 1

            row_dictionary = dict(
                zip(
                    input_columns,
                    values,
                )
            )

            row_dictionary[
                "simhash64_hex"
            ] = f"{simhash:016x}"

            row_dictionary[
                "near_duplicate_status"
            ] = "unique_representative"

            row_dictionary[
                "near_duplicate_representative_id"
            ] = ""

            row_dictionary[
                "final_parent_index"
            ] = final_unique_records

            row_dictionary[
                "final_pair_id"
            ] = (
                f"trnews_final_pair_"
                f"{final_unique_records:07d}"
            )

            row_dictionary[
                "pair_id"
            ] = row_dictionary[
                "final_pair_id"
            ]

            row_dictionary[
                "split_role"
            ] = "real_parent_final"

            final_rows.append(
                row_dictionary
            )

            source_counter[
                source
            ] += 1

            class_counter[
                original_class
            ] += 1

            if len(sample_rows) < (
                MAXIMUM_SAMPLE_SIZE
            ):
                sample_rows.append(
                    row_dictionary.copy()
                )

            else:
                replacement_index = int(
                    random_generator.integers(
                        1,
                        final_unique_records + 1,
                    )
                )

                if (
                    replacement_index
                    <= MAXIMUM_SAMPLE_SIZE
                ):
                    sample_rows[
                        replacement_index - 1
                    ] = (
                        row_dictionary.copy()
                    )

        final_dataframe = pd.DataFrame(
            final_rows,
            columns=(
                input_columns
                + FINAL_EXTRA_COLUMNS
            ),
        )

        duplicate_dataframe = pd.DataFrame(
            duplicate_rows,
            columns=(
                NEAR_DUPLICATE_COLUMNS
            ),
        )

        write_dataframe_chunk(
            dataframe=final_dataframe,
            output_path=FINAL_POOL_PATH,
            write_header=(
                not final_header_written
            ),
        )

        if not final_dataframe.empty:
            final_header_written = True

        write_dataframe_chunk(
            dataframe=duplicate_dataframe,
            output_path=(
                NEAR_DUPLICATE_REPORT_PATH
            ),
            write_header=(
                not duplicate_header_written
            ),
        )

        if not duplicate_dataframe.empty:
            duplicate_header_written = True

        print(
            f"Parça {chunk_number}: "
            f"okunan={total_records:,}, "
            f"final={final_unique_records:,}, "
            f"yakın_kopya={near_duplicate_records:,}"
        )

    planned_synthetic_records = (
        final_unique_records
    )

    planned_total_records = (
        final_unique_records * 2
    )

    summary = pd.DataFrame(
        [
            {
                "metric": (
                    "input_clean_records"
                ),
                "value": total_records,
            },
            {
                "metric": (
                    "near_duplicate_records_excluded"
                ),
                "value": near_duplicate_records,
            },
            {
                "metric": (
                    "final_real_parent_records"
                ),
                "value": final_unique_records,
            },
            {
                "metric": (
                    "planned_synthetic_fake_records"
                ),
                "value": (
                    planned_synthetic_records
                ),
            },
            {
                "metric": (
                    "planned_balanced_training_records"
                ),
                "value": (
                    planned_total_records
                ),
            },
            {
                "metric": (
                    "maximum_hamming_distance"
                ),
                "value": (
                    MAX_HAMMING_DISTANCE
                ),
            },
            {
                "metric": (
                    "minimum_length_ratio"
                ),
                "value": (
                    MINIMUM_LENGTH_RATIO
                ),
            },
            {
                "metric": (
                    "maximum_shingles_per_text"
                ),
                "value": (
                    MAX_SHINGLES_PER_TEXT
                ),
            },
        ]
    )

    source_distribution = pd.DataFrame(
        [
            {
                "source": source,
                "record_count": count,
                "percentage": round(
                    (
                        count
                        / final_unique_records
                        * 100
                    )
                    if final_unique_records
                    else 0.0,
                    4,
                ),
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
                    (
                        count
                        / final_unique_records
                        * 100
                    )
                    if final_unique_records
                    else 0.0,
                    4,
                ),
            }
            for original_class, count
            in class_counter.most_common()
        ]
    )

    summary.to_csv(
        SUMMARY_PATH,
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
        columns=(
            input_columns
            + FINAL_EXTRA_COLUMNS
        ),
    ).to_csv(
        SAMPLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("TRNEWS-2025 FİNAL GERÇEK HABER HAVUZU")
    print("=" * 80)

    print(
        f"Girdi temiz kayıt: "
        f"{total_records:,}"
    )

    print(
        "Yakın kopya olarak dışlanan: "
        f"{near_duplicate_records:,}"
    )

    print(
        f"Final gerçek haber: "
        f"{final_unique_records:,}"
    )

    print(
        "Planlanan sentetik asılsız: "
        f"{planned_synthetic_records:,}"
    )

    print(
        "Planlanan dengeli toplam veri: "
        f"{planned_total_records:,}"
    )

    print("\nOluşturulan dosyalar:")

    print(
        f"- {FINAL_POOL_PATH}"
    )

    print(
        f"- {NEAR_DUPLICATE_REPORT_PATH}"
    )

    print(
        f"- {SUMMARY_PATH}"
    )

    print(
        f"- {SOURCE_DISTRIBUTION_PATH}"
    )

    print(
        f"- {CLASS_DISTRIBUTION_PATH}"
    )

    print(
        f"- {SAMPLE_PATH}"
    )

    print(
        "\nSentetik üretim hedefi, final "
        "gerçek haber sayısına tam eşit "
        "olarak belirlendi."
    )


if __name__ == "__main__":
    main()