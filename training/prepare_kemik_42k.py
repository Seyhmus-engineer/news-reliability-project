from pathlib import Path
import hashlib
import re
import unicodedata

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kemik_42k"
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

STANDARDIZED_PATH = (
    PROCESSED_DIRECTORY
    / "kemik_42k_standardized.csv"
)

CATEGORY_SUMMARY_PATH = (
    ANALYSIS_DIRECTORY
    / "kemik_42k_category_summary.csv"
)

DUPLICATE_REPORT_PATH = (
    ANALYSIS_DIRECTORY
    / "kemik_42k_exact_duplicates.csv"
)


SOURCE_ALIASES = {
    "aa": "Anadolu Ajansı",
    "anadolu ajansi": "Anadolu Ajansı",
    "dha": "DHA",
    "iha": "İHA",
    "cihan": "CİHAN",
    "anka": "ANKA",
    "posta": "Posta",
    "fanatik": "Fanatik",
    "ntv": "NTV",
    "ntv spor": "NTV Spor",
    "ntvmsnbc": "ntvmsnbc",
    "reuters": "Reuters",
    "afp": "AFP",
    "hurriyet": "Hürriyet",
    "milliyet": "Milliyet",
    "sabah": "Sabah",
    "vatan": "Vatan",
    "star": "Star",
    "cumhuriyet": "Cumhuriyet",
    "aksam": "Akşam",
    "zaman": "Zaman",
    "radikal": "Radikal",
    "takvim": "Takvim",
    "cnn turk": "CNN Türk",
    "trt": "TRT",
}


def normalize_key(text: str) -> str:
    normalized = unicodedata.normalize(
        "NFKD",
        text,
    )

    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )

    normalized = normalized.casefold()

    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        normalized,
    )

    return re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()


def normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = "".join(
        character
        for character in text
        if unicodedata.category(character) != "Cf"
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def normalize_for_exact_duplicate(
    text: str,
) -> str:
    text = normalize_whitespace(
        text
    ).casefold()

    text = re.sub(
        r"[^0-9a-zçğıöşü]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def read_text_file(
    file_path: Path,
) -> str:
    encodings = (
        "utf-8-sig",
        "utf-8",
        "cp1254",
        "latin-1",
    )

    for encoding in encodings:
        try:
            return file_path.read_text(
                encoding=encoding
            )
        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"Dosya okunamadı: {file_path}"
    )


def find_news_root() -> Path:
    if not RAW_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Klasör bulunamadı: {RAW_DIRECTORY}\n"
            "42 Bin Haber arşivini bu klasöre çıkar."
        )

    candidates = [
        path
        for path in RAW_DIRECTORY.rglob("news")
        if path.is_dir()
    ]

    for candidate in candidates:
        txt_count = sum(
            1
            for _ in candidate.rglob("*.txt")
        )

        if txt_count > 10000:
            return candidate

    raise FileNotFoundError(
        "42 Bin Haber içindeki news klasörü "
        "bulunamadı. Arşivin tamamen "
        "çıkarıldığını kontrol et."
    )


def detect_and_remove_source(
    lines: list[str],
) -> tuple[str, list[str]]:
    cleaned_lines = list(lines)

    while (
        cleaned_lines
        and not cleaned_lines[-1].strip()
    ):
        cleaned_lines.pop()

    if not cleaned_lines:
        return "unknown", cleaned_lines

    last_line = cleaned_lines[-1].strip()

    last_line = re.sub(
        r"^kaynak\s*:\s*",
        "",
        last_line,
        flags=re.IGNORECASE,
    )

    source_key = normalize_key(
        last_line
    )

    detected_source = (
        SOURCE_ALIASES.get(source_key)
    )

    if detected_source is None:
        return "unknown", cleaned_lines

    return (
        detected_source,
        cleaned_lines[:-1],
    )


def parse_news_file(
    file_path: Path,
    news_root: Path,
) -> dict:
    raw_text = read_text_file(
        file_path
    )

    raw_text = raw_text.replace(
        "\r\n",
        "\n",
    )

    raw_text = raw_text.replace(
        "\r",
        "\n",
    ).strip()

    lines = [
        line.strip()
        for line in raw_text.split("\n")
    ]

    nonempty_indexes = [
        index
        for index, line in enumerate(lines)
        if line
    ]

    if not nonempty_indexes:
        title_raw = ""
        content_lines = []
    else:
        title_index = nonempty_indexes[0]

        title_raw = lines[
            title_index
        ]

        content_lines = lines[
            title_index + 1:
        ]

    (
        detected_source,
        content_without_signature,
    ) = detect_and_remove_source(
        content_lines
    )

    content_raw = "\n".join(
        content_lines
    ).strip()

    content_basic = "\n".join(
        content_without_signature
    ).strip()

    text_raw = normalize_whitespace(
        "\n".join(
            part
            for part in (
                title_raw,
                content_raw,
            )
            if part
        )
    )

    text_basic = normalize_whitespace(
        "\n".join(
            part
            for part in (
                title_raw,
                content_basic,
            )
            if part
        )
    )

    relative_path = file_path.relative_to(
        news_root
    )

    category = relative_path.parts[0]

    record_id = (
        f"kemik42k_{category}_"
        f"{file_path.stem}"
    )

    word_count = len(
        re.findall(
            r"\b\w+\b",
            text_basic,
            flags=re.UNICODE,
        )
    )

    cleaning_flags = [
        "unicode_normalized",
        "whitespace_normalized",
    ]

    if detected_source != "unknown":
        cleaning_flags.append(
            "trailing_source_signature_removed"
        )

    quality_status = (
        "review_required"
        if (
            len(text_basic) < 50
            or word_count < 8
        )
        else "quality_candidate"
    )

    return {
        "record_id": record_id,
        "dataset_name": "kemik_42k",
        "task_name": (
            "article_fake_news_binary"
        ),
        "text_type": "full_article",
        "category_original": category,
        "title_raw": title_raw,
        "content_raw": content_raw,
        "text_raw": text_raw,
        "text_basic": text_basic,
        "text_destyled": text_basic,
        "label_original": (
            "published_news"
        ),
        "label_standard": "real",
        "label_confidence": "weak",
        "verification_status": (
            "not_fact_checked"
        ),
        "source": detected_source,
        "url": "",
        "published_date": "",
        "cleaning_flags": ";".join(
            cleaning_flags
        ),
        "duplicate_group_id": "",
        "split_role": (
            "real_training_candidate"
        ),
        "is_synthetic": False,
        "text_character_count": len(
            text_basic
        ),
        "word_count": word_count,
        "quality_status": quality_status,
        "original_file": str(
            relative_path
        ).replace("\\", "/"),
    }


def main() -> None:
    print(
        "42 Bin Haber ortak şemaya "
        "dönüştürülüyor..."
    )

    news_root = find_news_root()

    text_files = sorted(
        news_root.rglob("*.txt")
    )

    print(
        f"Bulunan news klasörü: "
        f"{news_root}"
    )

    print(
        f"Bulunan TXT dosyası: "
        f"{len(text_files)}"
    )

    rows = [
        parse_news_file(
            file_path=file_path,
            news_root=news_root,
        )
        for file_path in text_files
    ]

    dataframe = pd.DataFrame(
        rows
    )

    dataframe["_duplicate_text"] = (
        dataframe["text_basic"]
        .apply(
            normalize_for_exact_duplicate
        )
    )

    dataframe["exact_text_hash"] = (
        dataframe["_duplicate_text"]
        .apply(
            lambda text: hashlib.sha1(
                text.encode("utf-8")
            ).hexdigest()
        )
    )

    group_sizes = (
        dataframe
        .groupby("_duplicate_text")[
            "record_id"
        ]
        .transform("size")
    )

    dataframe[
        "exact_duplicate_group_size"
    ] = group_sizes

    dataframe["is_exact_duplicate"] = (
        group_sizes > 1
    )

    duplicate_values = sorted(
        dataframe.loc[
            dataframe[
                "is_exact_duplicate"
            ],
            "_duplicate_text",
        ].unique()
    )

    duplicate_group_map = {
        value: (
            f"kemik42k_exact_{index:05d}"
        )
        for index, value in enumerate(
            duplicate_values,
            start=1,
        )
    }

    dataframe["duplicate_group_id"] = (
        dataframe["_duplicate_text"]
        .map(duplicate_group_map)
        .fillna("")
    )

    duplicate_report = dataframe[
        dataframe["is_exact_duplicate"]
    ].copy()

    duplicate_report["text_preview"] = (
        duplicate_report["text_basic"]
        .str.slice(0, 300)
    )

    duplicate_report = duplicate_report[
        [
            "duplicate_group_id",
            "exact_duplicate_group_size",
            "record_id",
            "category_original",
            "source",
            "label_standard",
            "title_raw",
            "text_preview",
            "original_file",
        ]
    ].sort_values(
        [
            "duplicate_group_id",
            "record_id",
        ]
    )

    category_summary = (
        dataframe
        .groupby(
            "category_original",
            dropna=False,
        )
        .agg(
            record_count=(
                "record_id",
                "size",
            ),
            quality_candidate_count=(
                "quality_status",
                lambda values: int(
                    (
                        values
                        == "quality_candidate"
                    ).sum()
                ),
            ),
            review_required_count=(
                "quality_status",
                lambda values: int(
                    (
                        values
                        == "review_required"
                    ).sum()
                ),
            ),
            source_detected_count=(
                "source",
                lambda values: int(
                    (
                        values
                        != "unknown"
                    ).sum()
                ),
            ),
            exact_duplicate_record_count=(
                "is_exact_duplicate",
                "sum",
            ),
            average_word_count=(
                "word_count",
                "mean",
            ),
        )
        .reset_index()
    )

    category_summary[
        "average_word_count"
    ] = (
        category_summary[
            "average_word_count"
        ].round(2)
    )

    dataframe.drop(
        columns=["_duplicate_text"],
        inplace=True,
    )

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANALYSIS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        STANDARDIZED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    category_summary.to_csv(
        CATEGORY_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    duplicate_report.to_csv(
        DUPLICATE_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    duplicate_group_count = (
        dataframe.loc[
            dataframe[
                "is_exact_duplicate"
            ],
            "duplicate_group_id",
        ]
        .nunique()
    )

    duplicate_record_count = int(
        dataframe[
            "is_exact_duplicate"
        ].sum()
    )

    review_required_count = int(
        (
            dataframe["quality_status"]
            == "review_required"
        ).sum()
    )

    detected_source_count = int(
        (
            dataframe["source"]
            != "unknown"
        ).sum()
    )

    print("\n" + "=" * 65)
    print("SONUÇ")
    print("=" * 65)

    print(
        f"Toplam kayıt: "
        f"{len(dataframe)}"
    )

    print(
        f"Kategori sayısı: "
        f"{dataframe['category_original'].nunique()}"
    )

    print(
        "Exact duplicate grubu: "
        f"{duplicate_group_count}"
    )

    print(
        "Exact duplicate içindeki kayıt: "
        f"{duplicate_record_count}"
    )

    print(
        "Kısa/düşük bilgi inceleme adayı: "
        f"{review_required_count}"
    )

    print(
        "Kaynağı son satırdan belirlenen: "
        f"{detected_source_count}"
    )

    print("\nKategori dağılımı:")

    print(
        category_summary[
            [
                "category_original",
                "record_count",
                "review_required_count",
                "exact_duplicate_record_count",
                "average_word_count",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nOluşturulan dosyalar:")

    print(
        f"- {STANDARDIZED_PATH}"
    )

    print(
        f"- {CATEGORY_SUMMARY_PATH}"
    )

    print(
        f"- {DUPLICATE_REPORT_PATH}"
    )

    print(
        "\nHiçbir kayıt silinmedi."
    )


if __name__ == "__main__":
    main()