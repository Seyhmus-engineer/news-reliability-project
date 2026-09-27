from pathlib import Path
from collections import Counter
import hashlib
import heapq
import random
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trnews_2025_final_real_pool.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_pilot"
)

ANALYSIS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)


PAIR_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "trnews_counterfactual_pilot_pairs.csv"
)

REAL_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "trnews_counterfactual_pilot_real.csv"
)

SYNTHETIC_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "trnews_counterfactual_pilot_fake.csv"
)

AUDIT_OUTPUT_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_counterfactual_pilot_audit.csv"
)

REVIEW_SAMPLE_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_counterfactual_pilot_review_sample.csv"
)

SUMMARY_OUTPUT_PATH = (
    ANALYSIS_DIRECTORY
    / "trnews_counterfactual_pilot_summary.csv"
)


CHUNK_SIZE = 20_000

PILOT_PAIR_COUNT = 5_000
PILOT_SELECTION_COUNT = 6_000

# Ön seçimde bellekte tutulacak aday sayısı.
CANDIDATE_POOL_SIZE = 30_000

# Tek bir haber kaynağının pilotu domine
# etmesini önlemek için başlangıç üst sınırı.
MAXIMUM_PAIRS_PER_SOURCE = 250

REVIEW_SAMPLE_SIZE = 200
RANDOM_STATE = 42


MONTH_NAMES = (
    "Ocak",
    "Şubat",
    "Mart",
    "Nisan",
    "Mayıs",
    "Haziran",
    "Temmuz",
    "Ağustos",
    "Eylül",
    "Ekim",
    "Kasım",
    "Aralık",
)

CITY_NAMES = [
    "Adana",
    "Ankara",
    "Antalya",
    "Bursa",
    "Diyarbakır",
    "Erzurum",
    "Eskişehir",
    "Gaziantep",
    "Hatay",
    "İstanbul",
    "İzmir",
    "Kayseri",
    "Kocaeli",
    "Konya",
    "Malatya",
    "Mardin",
    "Mersin",
    "Samsun",
    "Şanlıurfa",
    "Trabzon",
    "Van",
]


# Pilot aşamasında yüksek riskli konu alanlarını
# otomatik dönüşümden çıkarıyoruz.
SENSITIVE_TERMS = [
    "seçim",
    "oy pusulası",
    "referandum",
    "cumhurbaşkanlığı seçimi",
    "deprem",
    "sel felaketi",
    "orman yangını",
    "ölü sayısı",
    "hayatını kaybetti",
    "yaralı sayısı",
    "saldırı",
    "terör",
    "bomba",
    "savaş",
    "intihar",
    "çocuk istismarı",
    "aşı",
    "ilaç tedavisi",
    "kanser tedavisi",
]


PERCENT_WORD_PATTERN = re.compile(
    r"\b(yüzde)\s+"
    r"(\d{1,3}(?:[.,]\d+)?)",
    flags=re.IGNORECASE,
)

PERCENT_PREFIX_PATTERN = re.compile(
    r"(?<!\w)%\s*"
    r"(\d{1,3}(?:[.,]\d+)?)"
)

PERCENT_SUFFIX_PATTERN = re.compile(
    r"\b(\d{1,3}(?:[.,]\d+)?)\s*%"
)

MONEY_PATTERN = re.compile(
    r"\b"
    r"(\d+(?:[.,]\d+)?)"
    r"\s*"
    r"(bin|milyon|milyar)?"
    r"\s*"
    r"(TL|Türk lirası|lira|₺|"
    r"dolar|euro|avro)"
    r"\b",
    flags=re.IGNORECASE,
)

DATE_PATTERN = re.compile(
    r"\b"
    r"([0-3]?\d)"
    r"\s+"
    r"("
    + "|".join(MONTH_NAMES)
    + r")"
    r"(?:\s+((?:19|20)\d{2}))?"
    r"\b",
    flags=re.IGNORECASE,
)

TIME_PATTERN = re.compile(
    r"\b"
    r"([01]?\d|2[0-3])"
    r"[:.]"
    r"([0-5]\d)"
    r"\b"
)

YEAR_PATTERN = re.compile(
    r"\b"
    r"((?:19|20)\d{2})"
    r"\b"
)

CITY_PATTERN = re.compile(
    r"\b("
    + "|".join(
        re.escape(city)
        for city in CITY_NAMES
    )
    + r")\b",
    flags=re.IGNORECASE,
)

GENERAL_NUMBER_PATTERN = re.compile(
    r"\b"
    r"(\d{2,7})"
    r"\b"
)


OUTPUT_COLUMNS = [
    "record_id",
    "pair_id",
    "split_group_id",
    "parent_record_id",
    "pair_role",
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
    "is_synthetic",
    "generation_method",
    "edit_type",
    "original_fact",
    "modified_fact",
    "edit_context",
    "quality_status",
    "manual_review_status",
    "split_role",
]


def deterministic_rank(
    record_id: str,
) -> int:
    digest = hashlib.blake2b(
        record_id.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="big",
        signed=False,
    )


def contains_sensitive_topic(
    text: str,
) -> bool:
    lowered = text.casefold()

    return any(
        term.casefold() in lowered
        for term in SENSITIVE_TERMS
    )


def has_editable_fact(
    text: str,
) -> bool:
    patterns = [
        PERCENT_WORD_PATTERN,
        PERCENT_PREFIX_PATTERN,
        PERCENT_SUFFIX_PATTERN,
        MONEY_PATTERN,
        DATE_PATTERN,
        TIME_PATTERN,
        YEAR_PATTERN,
        CITY_PATTERN,
        GENERAL_NUMBER_PATTERN,
    ]

    return any(
        pattern.search(text)
        for pattern in patterns
    )


def format_number(
    value: float,
    original_text: str,
) -> str:
    uses_comma = "," in original_text
    has_decimal = (
        "," in original_text
        or "." in original_text
    )

    if not has_decimal:
        result = str(
            max(
                1,
                int(round(value)),
            )
        )

    else:
        result = f"{max(0.1, value):.1f}"

    if uses_comma:
        result = result.replace(
            ".",
            ",",
        )

    return result


def replace_span(
    text: str,
    start: int,
    end: int,
    replacement: str,
) -> str:
    return (
        text[:start]
        + replacement
        + text[end:]
    )


def create_edit_context(
    text: str,
    start: int,
    end: int,
) -> str:
    context_start = max(
        0,
        start - 100,
    )

    context_end = min(
        len(text),
        end + 100,
    )

    return text[
        context_start:context_end
    ].replace(
        "\n",
        " ",
    )


def modify_percentage(
    text: str,
    random_generator: random.Random,
):
    pattern_options = [
        (
            PERCENT_WORD_PATTERN,
            2,
        ),
        (
            PERCENT_PREFIX_PATTERN,
            1,
        ),
        (
            PERCENT_SUFFIX_PATTERN,
            1,
        ),
    ]

    random_generator.shuffle(
        pattern_options
    )

    for pattern, number_group in pattern_options:
        match = pattern.search(text)

        if match is None:
            continue

        original_number_text = match.group(
            number_group
        )

        original_number = float(
            original_number_text.replace(
                ",",
                ".",
            )
        )

        differences = [
            -25,
            -15,
            -10,
            10,
            15,
            25,
        ]

        random_generator.shuffle(
            differences
        )

        modified_number = None

        for difference in differences:
            candidate = (
                original_number
                + difference
            )

            if 0 <= candidate <= 100:
                modified_number = candidate
                break

        if modified_number is None:
            continue

        replacement_number = (
            format_number(
                modified_number,
                original_number_text,
            )
        )

        number_start = match.start(
            number_group
        )

        number_end = match.end(
            number_group
        )

        modified_text = replace_span(
            text,
            number_start,
            number_end,
            replacement_number,
        )

        return {
            "text": modified_text,
            "edit_type": "percentage_change",
            "original_fact": (
                original_number_text
            ),
            "modified_fact": (
                replacement_number
            ),
            "edit_context": (
                create_edit_context(
                    text,
                    match.start(),
                    match.end(),
                )
            ),
        }

    return None


def modify_money(
    text: str,
    random_generator: random.Random,
):
    match = MONEY_PATTERN.search(
        text
    )

    if match is None:
        return None

    original_number_text = match.group(1)

    original_number = float(
        original_number_text.replace(
            ",",
            ".",
        )
    )

    factor = random_generator.choice(
        [
            0.5,
            2,
            3,
            5,
        ]
    )

    modified_number = (
        original_number * factor
    )

    replacement_number = format_number(
        modified_number,
        original_number_text,
    )

    modified_text = replace_span(
        text,
        match.start(1),
        match.end(1),
        replacement_number,
    )

    return {
        "text": modified_text,
        "edit_type": "money_change",
        "original_fact": (
            match.group(0)
        ),
        "modified_fact": (
            modified_text[
                match.start():
                match.end()
                + len(replacement_number)
                - len(original_number_text)
            ]
        ),
        "edit_context": (
            create_edit_context(
                text,
                match.start(),
                match.end(),
            )
        ),
    }


def modify_date(
    text: str,
    random_generator: random.Random,
):
    match = DATE_PATTERN.search(
        text
    )

    if match is None:
        return None

    original_day = int(
        match.group(1)
    )

    day_shift = random_generator.choice(
        [
            -10,
            -7,
            -3,
            3,
            7,
            10,
        ]
    )

    modified_day = (
        (
            original_day
            - 1
            + day_shift
        )
        % 28
    ) + 1

    replacement_day = str(
        modified_day
    )

    modified_text = replace_span(
        text,
        match.start(1),
        match.end(1),
        replacement_day,
    )

    modified_fact_end = (
        match.end()
        + len(replacement_day)
        - len(match.group(1))
    )

    return {
        "text": modified_text,
        "edit_type": "date_change",
        "original_fact": (
            match.group(0)
        ),
        "modified_fact": (
            modified_text[
                match.start():
                modified_fact_end
            ]
        ),
        "edit_context": (
            create_edit_context(
                text,
                match.start(),
                match.end(),
            )
        ),
    }


def modify_time(
    text: str,
    random_generator: random.Random,
):
    match = TIME_PATTERN.search(
        text
    )

    if match is None:
        return None

    original_hour = int(
        match.group(1)
    )

    hour_shift = random_generator.choice(
        [
            2,
            3,
            5,
            7,
        ]
    )

    modified_hour = (
        original_hour
        + hour_shift
    ) % 24

    replacement_hour = (
        f"{modified_hour:02d}"
    )

    modified_text = replace_span(
        text,
        match.start(1),
        match.end(1),
        replacement_hour,
    )

    return {
        "text": modified_text,
        "edit_type": "time_change",
        "original_fact": (
            match.group(0)
        ),
        "modified_fact": (
            replacement_hour
            + ":"
            + match.group(2)
        ),
        "edit_context": (
            create_edit_context(
                text,
                match.start(),
                match.end(),
            )
        ),
    }


def modify_year(
    text: str,
    random_generator: random.Random,
):
    matches = list(
        YEAR_PATTERN.finditer(text)
    )

    if not matches:
        return None

    match = random_generator.choice(
        matches
    )

    original_year = int(
        match.group(1)
    )

    shift = random_generator.choice(
        [
            -5,
            -3,
            -2,
            -1,
            1,
            2,
            3,
            5,
        ]
    )

    modified_year = (
        original_year
        + shift
    )

    if not 1900 <= modified_year <= 2026:
        return None

    replacement = str(
        modified_year
    )

    modified_text = replace_span(
        text,
        match.start(1),
        match.end(1),
        replacement,
    )

    return {
        "text": modified_text,
        "edit_type": "year_change",
        "original_fact": (
            match.group(1)
        ),
        "modified_fact": replacement,
        "edit_context": (
            create_edit_context(
                text,
                match.start(),
                match.end(),
            )
        ),
    }


def modify_city(
    text: str,
    random_generator: random.Random,
):
    matches = list(
        CITY_PATTERN.finditer(text)
    )

    if not matches:
        return None

    match = random_generator.choice(
        matches
    )

    original_city = match.group(1)

    replacement_options = [
        city
        for city in CITY_NAMES
        if (
            city.casefold()
            != original_city.casefold()
        )
    ]

    replacement_city = (
        random_generator.choice(
            replacement_options
        )
    )

    modified_text = replace_span(
        text,
        match.start(1),
        match.end(1),
        replacement_city,
    )

    return {
        "text": modified_text,
        "edit_type": "location_change",
        "original_fact": original_city,
        "modified_fact": replacement_city,
        "edit_context": (
            create_edit_context(
                text,
                match.start(),
                match.end(),
            )
        ),
    }


def modify_general_number(
    text: str,
    random_generator: random.Random,
):
    matches = []

    for match in GENERAL_NUMBER_PATTERN.finditer(
        text
    ):
        number = int(
            match.group(1)
        )

        # Yıllar ayrı edit türünde işleniyor.
        if 1900 <= number <= 2026:
            continue

        matches.append(
            match
        )

    if not matches:
        return None

    match = random_generator.choice(
        matches
    )

    original_number = int(
        match.group(1)
    )

    factor = random_generator.choice(
        [
            0.5,
            2,
            3,
            5,
        ]
    )

    modified_number = max(
        1,
        int(round(
            original_number * factor
        )),
    )

    if modified_number == original_number:
        modified_number += 7

    replacement = str(
        modified_number
    )

    modified_text = replace_span(
        text,
        match.start(1),
        match.end(1),
        replacement,
    )

    return {
        "text": modified_text,
        "edit_type": "number_change",
        "original_fact": (
            match.group(1)
        ),
        "modified_fact": replacement,
        "edit_context": (
            create_edit_context(
                text,
                match.start(),
                match.end(),
            )
        ),
    }


def create_counterfactual(
    text: str,
    record_id: str,
):
    seed = deterministic_rank(
        record_id
    )

    random_generator = random.Random(
        seed
    )

    edit_functions = [
        modify_percentage,
        modify_money,
        modify_date,
        modify_time,
        modify_city,
        modify_year,
        modify_general_number,
    ]

    random_generator.shuffle(
        edit_functions
    )

    for edit_function in edit_functions:
        result = edit_function(
            text,
            random_generator,
        )

        if (
            result is not None
            and result["text"] != text
        ):
            return result

    return None


def collect_candidate_pool() -> list[dict]:
    candidate_heap = []

    required_columns = {
        "record_id",
        "pair_id",
        "text_basic",
        "source",
        "url",
        "access_date",
        "original_class",
        "word_count",
    }

    reader = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8-sig",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
        on_bad_lines="skip",
        low_memory=False,
    )

    total_read = 0
    eligible_count = 0

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
                "Final TRNews dosyasında eksik "
                f"sütunlar var: "
                f"{sorted(missing_columns)}"
            )

        for row in chunk[
            list(required_columns)
        ].to_dict(
            orient="records"
        ):
            total_read += 1

            text = str(
                row["text_basic"]
            ).strip()

            if not text:
                continue

            if contains_sensitive_topic(
                text
            ):
                continue

            if not has_editable_fact(
                text
            ):
                continue

            eligible_count += 1

            rank = deterministic_rank(
                row["record_id"]
            )

            candidate = (
                -rank,
                str(row["record_id"]),
                {
                    **row,
                    "selection_rank": rank,
                },
            )

            if (
                len(candidate_heap)
                < CANDIDATE_POOL_SIZE
            ):
                heapq.heappush(
                    candidate_heap,
                    candidate,
                )

            elif rank < -candidate_heap[0][0]:
                heapq.heapreplace(
                    candidate_heap,
                    candidate,
                )

        print(
            f"Parça {chunk_number}: "
            f"okunan={total_read:,}, "
            f"uygun={eligible_count:,}"
        )

    candidates = [
        item[2]
        for item in candidate_heap
    ]

    candidates.sort(
        key=lambda row: (
            row["selection_rank"],
            row["record_id"],
        )
    )

    print(
        "\nPilot ön seçim havuzu: "
        f"{len(candidates):,}"
    )

    return candidates


def select_source_balanced_candidates(
    candidates: list[dict],
) -> list[dict]:
    selected = []
    selected_ids = set()
    source_counts = Counter()

    for row in candidates:
        if len(selected) >= PILOT_SELECTION_COUNT:
            break

        source = (
            str(row["source"]).strip()
            or "unknown"
        )

        if (
            source_counts[source]
            >= MAXIMUM_PAIRS_PER_SOURCE
        ):
            continue

        selected.append(
            row
        )

        selected_ids.add(
            row["record_id"]
        )

        source_counts[source] += 1

    # Kaynak üst sınırı nedeniyle 5.000'e
    # ulaşılamazsa kalan kayıtlar doldurulur.
    if len(selected) < PILOT_PAIR_COUNT:
        for row in candidates:
            if len(selected) >= PILOT_PAIR_COUNT:
                break

            if row["record_id"] in selected_ids:
                continue

            selected.append(
                row
            )

            selected_ids.add(
                row["record_id"]
            )

            source_counts[
                str(row["source"])
            ] += 1

    return selected


def build_pair_rows(
    selected_candidates: list[dict],
):
    real_rows = []
    synthetic_rows = []
    audit_rows = []

    edit_counter = Counter()
    failed_records = []

    pair_index = 0

    for parent in selected_candidates:
        if pair_index >= PILOT_PAIR_COUNT:
            break

        parent_record_id = str(
            parent["record_id"]
        )

        original_text = str(
            parent["text_basic"]
        )

        edit_result = create_counterfactual(
            text=original_text,
            record_id=parent_record_id,
        )

        if edit_result is None:
            failed_records.append(
                parent_record_id
            )

            continue

        pair_index += 1

        pair_id = (
            f"trnews_pilot_pair_"
            f"{pair_index:05d}"
        )

        synthetic_record_id = (
            f"trnews_pilot_fake_"
            f"{pair_index:05d}"
        )

        shared_data = {
            "pair_id": pair_id,
            "split_group_id": pair_id,
            "dataset_name": (
                "TRNews-2025-counterfactual-pilot"
            ),
            "task_name": (
                "article_fake_news_binary"
            ),
            "text_type": (
                "full_news_article"
            ),
            "source": parent["source"],
            "access_date": (
                parent["access_date"]
            ),
            "original_class": (
                parent["original_class"]
            ),
            "word_count": (
                parent["word_count"]
            ),
            "generation_method": (
                "controlled_fact_perturbation_v1"
            ),
            "split_role": (
                "synthetic_pilot_train"
            ),
        }

        real_row = {
            "record_id": parent_record_id,
            **shared_data,
            "parent_record_id": "",
            "pair_role": "real_parent",
            "text_basic": original_text,
            "label_standard": "real",
            "label_confidence": "weak",
            "verification_status": (
                "published_news_not_fact_checked"
            ),
            "url": parent["url"],
            "is_synthetic": False,
            "edit_type": "",
            "original_fact": "",
            "modified_fact": "",
            "edit_context": "",
            "quality_status": (
                "pilot_parent_selected"
            ),
            "manual_review_status": (
                "not_required_for_parent"
            ),
        }

        synthetic_row = {
            "record_id": synthetic_record_id,
            **shared_data,
            "parent_record_id": (
                parent_record_id
            ),
            "pair_role": (
                "synthetic_counterfactual"
            ),
            "text_basic": (
                edit_result["text"]
            ),
            "label_standard": "fake",
            "label_confidence": (
                "synthetic_counterfactual"
            ),
            "verification_status": (
                "counterfactual_edit_from_parent"
            ),
            "url": "",
            "is_synthetic": True,
            "edit_type": (
                edit_result["edit_type"]
            ),
            "original_fact": (
                edit_result["original_fact"]
            ),
            "modified_fact": (
                edit_result["modified_fact"]
            ),
            "edit_context": (
                edit_result["edit_context"]
            ),
            "quality_status": (
                "pilot_pending_manual_review"
            ),
            "manual_review_status": (
                "pending"
            ),
        }

        real_rows.append(
            real_row
        )

        synthetic_rows.append(
            synthetic_row
        )

        audit_rows.append(
            {
                "pair_id": pair_id,
                "parent_record_id": (
                    parent_record_id
                ),
                "synthetic_record_id": (
                    synthetic_record_id
                ),
                "source": parent["source"],
                "edit_type": (
                    edit_result["edit_type"]
                ),
                "original_fact": (
                    edit_result[
                        "original_fact"
                    ]
                ),
                "modified_fact": (
                    edit_result[
                        "modified_fact"
                    ]
                ),
                "edit_context": (
                    edit_result[
                        "edit_context"
                    ]
                ),
                "original_text_preview": (
                    original_text[:500]
                ),
                "synthetic_text_preview": (
                    edit_result["text"][:500]
                ),
                "manual_review_status": (
                    "pending"
                ),
                "review_notes": "",
            }
        )

        edit_counter[
            edit_result["edit_type"]
        ] += 1

    return (
        real_rows,
        synthetic_rows,
        audit_rows,
        edit_counter,
        failed_records,
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Final TRNews havuzu bulunamadı: "
            f"{INPUT_PATH}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANALYSIS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TRNews karşı-olgusal pilot "
        "verisi hazırlanıyor..."
    )

    candidates = (
        collect_candidate_pool()
    )

    selected_candidates = (
        select_source_balanced_candidates(
            candidates
        )
    )

    (
        real_rows,
        synthetic_rows,
        audit_rows,
        edit_counter,
        failed_records,
    ) = build_pair_rows(
        selected_candidates
    )

    if len(synthetic_rows) < PILOT_PAIR_COUNT:
        raise ValueError(
            "5.000 başarılı sentetik kayıt "
            "oluşturulamadı. Oluşan kayıt: "
            f"{len(synthetic_rows)}"
        )

    real_dataframe = pd.DataFrame(
        real_rows,
        columns=OUTPUT_COLUMNS,
    )

    synthetic_dataframe = pd.DataFrame(
        synthetic_rows,
        columns=OUTPUT_COLUMNS,
    )

    pair_dataframe = pd.concat(
        [
            real_dataframe,
            synthetic_dataframe,
        ],
        ignore_index=True,
    )

    # Model eğitimi sırasında sıralı çiftlerin
    # etkisini azaltmak için karıştırılır.
    pair_dataframe = (
        pair_dataframe
        .sample(
            frac=1.0,
            random_state=RANDOM_STATE,
        )
        .reset_index(
            drop=True
        )
    )

    audit_dataframe = pd.DataFrame(
        audit_rows
    )

    review_sample_size = min(
        REVIEW_SAMPLE_SIZE,
        len(audit_dataframe),
    )

    review_sample = (
        audit_dataframe.sample(
            n=review_sample_size,
            random_state=RANDOM_STATE,
        )
        .sort_values("pair_id")
    )

    summary_rows = [
        {
            "metric": (
                "final_real_pool_records"
            ),
            "value": 655_357,
        },
        {
            "metric": (
                "final_planned_synthetic_records"
            ),
            "value": 655_357,
        },
        {
            "metric": (
                "pilot_real_records"
            ),
            "value": len(
                real_dataframe
            ),
        },
        {
            "metric": (
                "pilot_synthetic_records"
            ),
            "value": len(
                synthetic_dataframe
            ),
        },
        {
            "metric": (
                "pilot_total_records"
            ),
            "value": len(
                pair_dataframe
            ),
        },
        {
            "metric": (
                "pilot_unique_sources"
            ),
            "value": int(
                real_dataframe[
                    "source"
                ].nunique()
            ),
        },
        {
            "metric": (
                "failed_generation_records"
            ),
            "value": len(
                failed_records
            ),
        },
    ]

    for edit_type, count in (
        edit_counter.most_common()
    ):
        summary_rows.append(
            {
                "metric": (
                    f"edit_type_{edit_type}"
                ),
                "value": count,
            }
        )

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    pair_dataframe.to_csv(
        PAIR_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    real_dataframe.to_csv(
        REAL_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    synthetic_dataframe.to_csv(
        SYNTHETIC_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    audit_dataframe.to_csv(
        AUDIT_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    review_sample.to_csv(
        REVIEW_SAMPLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary_dataframe.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 78)
    print("TRNEWS KARŞI-OLGUSAL PİLOT SONUCU")
    print("=" * 78)

    print(
        f"Pilot gerçek haber: "
        f"{len(real_dataframe):,}"
    )

    print(
        f"Pilot sentetik asılsız: "
        f"{len(synthetic_dataframe):,}"
    )

    print(
        f"Pilot toplam kayıt: "
        f"{len(pair_dataframe):,}"
    )

    print(
        "Kullanılan benzersiz kaynak: "
        f"{real_dataframe['source'].nunique():,}"
    )

    print("\nDeğişiklik türleri:")

    for edit_type, count in (
        edit_counter.most_common()
    ):
        print(
            f"- {edit_type}: {count:,}"
        )

    print("\nOluşturulan dosyalar:")

    print(
        f"- {PAIR_OUTPUT_PATH}"
    )

    print(
        f"- {REAL_OUTPUT_PATH}"
    )

    print(
        f"- {SYNTHETIC_OUTPUT_PATH}"
    )

    print(
        f"- {AUDIT_OUTPUT_PATH}"
    )

    print(
        f"- {REVIEW_SAMPLE_PATH}"
    )

    print(
        f"- {SUMMARY_OUTPUT_PATH}"
    )

    print(
        "\nSentetik kayıtlar yalnızca "
        "proje içi eğitim amacıyla "
        "oluşturulmuştur."
    )


if __name__ == "__main__":
    main()