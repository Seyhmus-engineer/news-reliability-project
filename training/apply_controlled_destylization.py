from pathlib import Path
import re
import unicodedata

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "destyled"
)

ANALYSIS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)


DATASET_FILES = {
    "article_main": (
        "article_main_unified.csv",
        "article_main_destyled.csv",
    ),
    "facturk": (
        "facturk_training_candidates.csv",
        "facturk_training_candidates_destyled.csv",
    ),
    "fctr": (
        "fctr_unified.csv",
        "fctr_destyled.csv",
    ),
    "mide22": (
        "mide22_unified.csv",
        "mide22_destyled.csv",
    ),
    "satiretr": (
        "satiretr_unified.csv",
        "satiretr_destyled.csv",
    ),
}


SOURCE_BRAND_PATTERN = re.compile(
    r"\b(?:"
    r"teyit(?:\.org)?|"
    r"doğruluk\s+payı|"
    r"dogrulukpayi\.com|"
    r"doğrula(?:\.org)?|"
    r"malumatfuruş|"
    r"trt\s+haber|"
    r"anadolu\s+ajansı"
    r")\b",
    flags=re.IGNORECASE,
)

AGENCY_CITY_SIGNATURE_PATTERN = re.compile(
    r"(?:^|\n)\s*"
    r"(?:[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü/-]*"
    r"(?:\s+[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü/-]*)*)"
    r"\s*\((?:AA|DHA|İHA)\)\s*[-–—:]\s*",
    flags=re.MULTILINE,
)

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+"
    r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

MENTION_PATTERN = re.compile(
    r"(?<!\w)@[A-Za-z0-9_]+"
)

HASHTAG_PATTERN = re.compile(
    r"(?<!\w)#([^\s#]+)"
)

AA_NEWS_PATTERN = re.compile(
    r"\bAA(?:'nın|’nın|nın)?\s+haberine\s+göre\b",
    flags=re.IGNORECASE,
)
REPEATED_USER_PATTERN = re.compile(
    r"(?:<USER>\s*){2,}"
)

REPEATED_URL_PATTERN = re.compile(
    r"(?:<URL>\s*){2,}"
)

REPORTER_PATTERNS = [
    (
        re.compile(
            r"\b(?:AA|TRT\s+Haber)\s+muhabirine\b",
            flags=re.IGNORECASE,
        ),
        "muhabire",
    ),
    (
        re.compile(
            r"\b(?:AA|TRT\s+Haber)\s+muhabirinin\b",
            flags=re.IGNORECASE,
        ),
        "muhabirin",
    ),
    (
        re.compile(
            r"\b(?:AA|TRT\s+Haber)\s+muhabiri\b",
            flags=re.IGNORECASE,
        ),
        "muhabir",
    ),
]

SOCIAL_MEDIA_CLAIM_PATTERNS = [
    (
        re.compile(
            r"\bsosyal\s+medyada\s+"
            r"(?:paylaşılan|yayılan|yer\s+alan|"
            r"dolaşıma\s+giren)\s+"
            r"iddiaya\s+göre\b",
            flags=re.IGNORECASE,
        ),
        "bir sosyal medya iddiasına göre",
    ),
    (
        re.compile(
            r"\bsosyal\s+medyada\s+"
            r"(?:paylaşılan|yayılan|yer\s+alan|"
            r"dolaşıma\s+giren)\s+"
            r"(bir\s+)?"
            r"(iddia|video|fotoğraf|görüntü|paylaşım)\b",
            flags=re.IGNORECASE,
        ),
        r"sosyal medyadaki \1\2",
    ),
]

CLAIM_STATEMENT_PATTERNS = [
    (
        re.compile(
            r"\biddia\s+edildi\b",
            flags=re.IGNORECASE,
        ),
        "iddia olarak aktarıldı",
    ),
    (
        re.compile(
            r"\biddia\s+ediliyor\b",
            flags=re.IGNORECASE,
        ),
        "iddia olarak aktarılıyor",
    ),
    (
        re.compile(
            r"\biddia\s+edilmiş\b",
            flags=re.IGNORECASE,
        ),
        "iddia olarak aktarılmış",
    ),
]

HEADING_PATTERN = re.compile(
    r"(?:^|\n)\s*"
    r"(?:analiz|bulgularımız|bulgular)\s*:?\s*",
    flags=re.IGNORECASE | re.MULTILINE,
)

VERIFICATION_RESULT_PATTERN = re.compile(
    r"\b(?:"
    r"teyit\s+sonucu|"
    r"teyit\s+edildi|"
    r"teyit\s+edilemedi"
    r")\b",
    flags=re.IGNORECASE,
)

ENGAGEMENT_PATTERN = re.compile(
    r"\b\d[\d.,]*\s*"
    r"(?:beğeni|retweet|görüntülenme|takipçi)"
    r"(?:\s+sayısı)?\b",
    flags=re.IGNORECASE,
)


def normalize_unicode(text: object) -> str:
    if pd.isna(text):
        return ""

    normalized = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    # U+2066, U+2069, zero-width space ve benzeri
    # görünmeyen Unicode biçimlendirme karakterlerini kaldırır.
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )

    return normalized


def normalize_whitespace(text: str) -> str:
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s*\n\s*",
        "\n",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def apply_substitution(
    text: str,
    pattern: re.Pattern,
    replacement: str,
) -> tuple[str, int]:
    return pattern.subn(
        replacement,
        text,
    )


def create_masked_text(
    text: str,
    dataset_key: str,
) -> tuple[str, dict[str, int]]:
    counts = {
        "url_masked": 0,
        "email_masked": 0,
        "mention_masked": 0,
        "hashtag_normalized": 0,
        "source_brand_masked": 0,
        "agency_signature_removed": 0,
        "reporter_phrase_normalized": 0,
        "agency_news_phrase_normalized": 0,
        "engagement_masked": 0,
        "repeated_users_collapsed": 0,
        "repeated_urls_collapsed": 0,
    }

    masked = normalize_unicode(text)

    masked, count = apply_substitution(
        masked,
        URL_PATTERN,
        "<URL>",
    )
    counts["url_masked"] += count

    masked, count = apply_substitution(
        masked,
        EMAIL_PATTERN,
        "<EMAIL>",
    )
    counts["email_masked"] += count

    if dataset_key == "mide22":
        masked, count = apply_substitution(
            masked,
            MENTION_PATTERN,
            "<USER>",

        )
        counts["mention_masked"] += count

        masked, count = HASHTAG_PATTERN.subn(
            lambda match: match.group(1),
            masked,
        )
        counts["hashtag_normalized"] += count

        masked, count = apply_substitution(
            masked,
            ENGAGEMENT_PATTERN,
            "<ENGAGEMENT>",
        )
        counts["engagement_masked"] += count

    masked, count = apply_substitution(
        masked,
        SOURCE_BRAND_PATTERN,
        "<SOURCE>",
    )
    counts["source_brand_masked"] += count

    masked, count = apply_substitution(
        masked,
        AGENCY_CITY_SIGNATURE_PATTERN,
        "\n",
    )
    counts["agency_signature_removed"] += count

    masked, count = apply_substitution(
        masked,
        AA_NEWS_PATTERN,
        "habere göre",
    )
    counts["agency_news_phrase_normalized"] += count
    masked, count = apply_substitution(
        masked,
        REPEATED_USER_PATTERN,
        "<USER> ",
    )

    counts["repeated_users_collapsed"] += count

    masked, count = apply_substitution(
        masked,
        REPEATED_URL_PATTERN,
        "<URL> ",
    )

    counts["repeated_urls_collapsed"] += count

    for reporter_pattern, replacement in REPORTER_PATTERNS:
        masked, count = apply_substitution(
            masked,
            reporter_pattern,
            replacement,
        )

        counts["reporter_phrase_normalized"] += count

    return normalize_whitespace(masked), counts


def create_destyled_text(
    masked_text: str,
    dataset_key: str,
) -> tuple[str, dict[str, int]]:
    counts = {
        "social_media_claim_normalized": 0,
        "claim_statement_normalized": 0,
        "heading_removed": 0,
        "verification_phrase_normalized": 0,
    }

    destyled = masked_text

    if dataset_key in {
        "article_main",
        "facturk",
        "fctr",
    }:
        for social_pattern, replacement in (
                SOCIAL_MEDIA_CLAIM_PATTERNS
        ):
            destyled, count = apply_substitution(
                destyled,
                social_pattern,
                replacement,
            )

            counts[
                "social_media_claim_normalized"
            ] += count

        for claim_pattern, replacement in (
            CLAIM_STATEMENT_PATTERNS
        ):
            destyled, count = apply_substitution(
                destyled,
                claim_pattern,
                replacement,
            )

            counts[
                "claim_statement_normalized"
            ] += count

        destyled, count = apply_substitution(
            destyled,
            HEADING_PATTERN,
            "\n",
        )

        counts["heading_removed"] += count

        destyled, count = apply_substitution(
            destyled,
            VERIFICATION_RESULT_PATTERN,
            "değerlendirme sonucu",
        )

        counts[
            "verification_phrase_normalized"
        ] += count

    return normalize_whitespace(destyled), counts


def process_dataset(
    dataset_key: str,
    input_name: str,
    output_name: str,
) -> tuple[pd.DataFrame, dict]:
    input_path = INPUT_DIRECTORY / input_name
    output_path = OUTPUT_DIRECTORY / output_name

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dosya bulunamadı: {input_path}"
        )

    dataframe = pd.read_csv(input_path)

    required_columns = {
        "record_id",
        "text_basic",
        "text_destyled",
        "cleaning_flags",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{input_name} dosyasında eksik sütunlar: "
            f"{sorted(missing_columns)}"
        )

    output = dataframe.copy()

    masked_texts = []
    destyled_texts = []
    replacement_counts = []

    for text_value in output["text_basic"]:
        masked_text, masking_counts = (
            create_masked_text(
                text=str(text_value),
                dataset_key=dataset_key,
            )
        )

        destyled_text, destyle_counts = (
            create_destyled_text(
                masked_text=masked_text,
                dataset_key=dataset_key,
            )
        )

        combined_counts = {
            **masking_counts,
            **destyle_counts,
        }

        masked_texts.append(masked_text)
        destyled_texts.append(destyled_text)
        replacement_counts.append(
            sum(combined_counts.values())
        )

    output["text_masked"] = masked_texts
    output["text_destyled"] = destyled_texts

    output["destyle_change_count"] = (
        replacement_counts
    )

    output["text_length_before"] = (
        output["text_basic"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    output["text_length_after"] = (
        output["text_destyled"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    output["cleaning_flags"] = (
        output["cleaning_flags"]
        .fillna("")
        .astype(str)
        .str.rstrip(";")
        + ";controlled_metadata_masking"
        + ";controlled_style_normalization"
    )

    empty_after = (
        output["text_destyled"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_after > 0:
        raise ValueError(
            f"{dataset_key}: Temizlikten sonra "
            f"{empty_after} boş metin oluştu."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    changed_rows = int(
        output["destyle_change_count"]
        .gt(0)
        .sum()
    )

    summary = {
        "dataset": dataset_key,
        "input_records": len(output),
        "changed_records": changed_rows,
        "unchanged_records": (
            len(output) - changed_rows
        ),
        "total_changes": int(
            output["destyle_change_count"].sum()
        ),
        "average_changes_per_record": round(
            output["destyle_change_count"].mean(),
            4,
        ),
        "average_length_before": round(
            output["text_length_before"].mean(),
            2,
        ),
        "average_length_after": round(
            output["text_length_after"].mean(),
            2,
        ),
        "empty_after_cleaning": int(empty_after),
        "output_file": str(output_path),
    }

    return output, summary


def main() -> None:
    print(
        "Kontrollü kaynak izi temizliği uygulanıyor..."
    )

    summaries = []

    for (
        dataset_key,
        (
            input_name,
            output_name,
        ),
    ) in DATASET_FILES.items():
        print("\n" + "=" * 60)
        print(dataset_key.upper())
        print("=" * 60)

        output, summary = process_dataset(
            dataset_key=dataset_key,
            input_name=input_name,
            output_name=output_name,
        )

        summaries.append(summary)

        print(f"Toplam kayıt: {len(output)}")
        print(
            "Değişiklik yapılan kayıt: "
            f"{summary['changed_records']}"
        )
        print(
            "Toplam maskeleme/normalizasyon: "
            f"{summary['total_changes']}"
        )
        print(
            "Ortalama uzunluk önce/sonra: "
            f"{summary['average_length_before']} / "
            f"{summary['average_length_after']}"
        )
        print(
            f"Çıktı: {summary['output_file']}"
        )

    report = pd.DataFrame(summaries)

    ANALYSIS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        ANALYSIS_DIRECTORY
        / "controlled_destylization_summary.csv"
    )

    report.to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("GENEL ÖZET")
    print("=" * 60)

    print(
        report[
            [
                "dataset",
                "input_records",
                "changed_records",
                "total_changes",
                "empty_after_cleaning",
            ]
        ].to_string(index=False)
    )

    print(f"\nRapor: {report_path}")
    print(
        "\nOrijinal ortak şema dosyaları "
        "değiştirilmedi."
    )


if __name__ == "__main__":
    main()