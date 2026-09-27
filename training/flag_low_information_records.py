from pathlib import Path
import re
import unicodedata

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "destyled"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "quality_flagged"
)

ANALYSIS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)


DATASET_FILES = {
    "article_main": (
        "article_main_destyled.csv",
        "article_main_quality_flagged.csv",
    ),
    "facturk": (
        "facturk_training_candidates_destyled.csv",
        "facturk_training_candidates_quality_flagged.csv",
    ),
    "fctr": (
        "fctr_destyled.csv",
        "fctr_quality_flagged.csv",
    ),
    "mide22": (
        "mide22_destyled.csv",
        "mide22_quality_flagged.csv",
    ),
    "satiretr": (
        "satiretr_destyled.csv",
        "satiretr_quality_flagged.csv",
    ),
}


# Her veri türünün doğal metin uzunluğu farklıdır.
QUALITY_THRESHOLDS = {
    "article_main": {
        "minimum_characters": 40,
        "minimum_meaningful_tokens": 6,
    },
    "facturk": {
        # Kısa iddialar anlamlı olabildiği için daha esnek.
        "minimum_characters": 10,
        "minimum_meaningful_tokens": 2,
    },
    "fctr": {
        "minimum_characters": 15,
        "minimum_meaningful_tokens": 2,
    },
    "mide22": {
        # Sosyal medya metinleri doğal olarak kısadır.
        "minimum_characters": 12,
        "minimum_meaningful_tokens": 2,
    },
    "satiretr": {
        "minimum_characters": 70,
        "minimum_meaningful_tokens": 8,
    },
}

PLACEHOLDER_PATTERN = re.compile(
    r"<(?:URL|USER|EMAIL|SOURCE|ENGAGEMENT)>",
    flags=re.IGNORECASE,
)

TOKEN_PATTERN = re.compile(
    r"[A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+"
)

PUNCTUATION_PATTERN = re.compile(
    r"[^\wÇĞİÖŞÜçğıöşü]+",
    flags=re.UNICODE,
)


# Tek başına bulunduğunda sınıflandırma için çok az bilgi
# taşıyan ifadeler. Yalnızca tam eşleşmeler işaretlenir.
GENERIC_LOW_INFORMATION_PHRASES = {
    "iddia",
    "iddia edildi",
    "iddia ediliyor",
    "iddia edilmiş",
    "iddia olarak aktarıldı",
    "iddia olarak aktarılıyor",
    "iddia olarak aktarılmış",
    "doğru",
    "yanlış",
    "gerçek",
    "yalan",
    "doğru değil",
    "yanlış bilgi",
}


REFERENCE_ONLY_PATTERNS = [
    re.compile(
        r"^(?:<USER>\s*)?"
        r"(?:ilgili|detaylı)?\s*"
        r"(?:haberimiz|yazımız|açıklama|kaynak|link)"
        r"\s*:?\s*(?:<URL>)?$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^(?:<USER>\s*)?<URL>$",
        flags=re.IGNORECASE,
    ),
]


def normalize_for_comparison(text: object) -> str:
    if pd.isna(text):
        return ""

    normalized = unicodedata.normalize(
        "NFKC",
        str(text),
    ).lower()

    normalized = PLACEHOLDER_PATTERN.sub(
        " ",
        normalized,
    )

    normalized = PUNCTUATION_PATTERN.sub(
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def calculate_text_metrics(text: object) -> dict:
    if pd.isna(text):
        text_value = ""
    else:
        text_value = str(text).strip()

    placeholders = PLACEHOLDER_PATTERN.findall(
        text_value
    )

    text_without_placeholders = PLACEHOLDER_PATTERN.sub(
        " ",
        text_value,
    )

    meaningful_tokens = TOKEN_PATTERN.findall(
        text_without_placeholders
    )

    meaningful_tokens = [
        token
        for token in meaningful_tokens
        if len(token) >= 2
    ]

    alphabetic_character_count = sum(
        character.isalpha()
        for character in text_without_placeholders
    )

    total_units = (
        len(meaningful_tokens)
        + len(placeholders)
    )

    placeholder_ratio = (
        len(placeholders) / total_units
        if total_units > 0
        else 0.0
    )

    return {
        "text_character_count": len(text_value),
        "meaningful_token_count": len(
            meaningful_tokens
        ),
        "alphabetic_character_count": (
            alphabetic_character_count
        ),
        "placeholder_count": len(placeholders),
        "placeholder_ratio": round(
            placeholder_ratio,
            4,
        ),
    }


def determine_low_information_reasons(
    text: object,
    dataset_key: str,
    metrics: dict,
) -> list[str]:
    reasons = []

    thresholds = QUALITY_THRESHOLDS[
        dataset_key
    ]

    text_value = (
        ""
        if pd.isna(text)
        else str(text).strip()
    )

    normalized_text = normalize_for_comparison(
        text_value
    )

    if (
        metrics["text_character_count"]
        < thresholds["minimum_characters"]
    ):
        reasons.append(
            "below_minimum_character_count"
        )

    if (
        metrics["meaningful_token_count"]
        < thresholds["minimum_meaningful_tokens"]
    ):
        reasons.append(
            "too_few_meaningful_tokens"
        )

    if (
        metrics["placeholder_count"] > 0
        and metrics["placeholder_ratio"] >= 0.50
    ):
        reasons.append(
            "placeholder_dominated"
        )

    if (
        dataset_key in {"facturk", "fctr"}
        and normalized_text
        in GENERIC_LOW_INFORMATION_PHRASES
    ):
        reasons.append(
            "generic_claim_phrase"
        )

    if dataset_key == "mide22":
        if any(
            pattern.fullmatch(text_value)
            for pattern in REFERENCE_ONLY_PATTERNS
        ):
            reasons.append(
                "reference_or_link_only"
            )

    if (
        metrics["alphabetic_character_count"] == 0
    ):
        reasons.append(
            "no_alphabetic_content"
        )

    return reasons


def process_dataset(
    dataset_key: str,
    input_name: str,
    output_name: str,
) -> tuple[pd.DataFrame, dict]:
    input_path = (
        INPUT_DIRECTORY
        / input_name
    )

    output_path = (
        OUTPUT_DIRECTORY
        / output_name
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dosya bulunamadı: {input_path}"
        )

    dataframe = pd.read_csv(
        input_path
    )

    required_columns = {
        "record_id",
        "dataset_name",
        "text_type",
        "text_destyled",
        "label_standard",
        "source",
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

    metric_rows = []
    reason_rows = []

    for text_value in output["text_destyled"]:
        metrics = calculate_text_metrics(
            text_value
        )

        reasons = determine_low_information_reasons(
            text=text_value,
            dataset_key=dataset_key,
            metrics=metrics,
        )

        metric_rows.append(metrics)
        reason_rows.append(reasons)

    metrics_dataframe = pd.DataFrame(
        metric_rows
    )

    for column in metrics_dataframe.columns:
        output[column] = metrics_dataframe[
            column
        ].values

    output["low_information_reasons"] = [
        ";".join(reasons)
        for reasons in reason_rows
    ]

    output["low_information"] = [
        len(reasons) > 0
        for reasons in reason_rows
    ]

    output["quality_status"] = output[
        "low_information"
    ].map(
        {
            True: "review_required",
            False: "quality_candidate",
        }
    )

    output["cleaning_flags"] = (
        output["cleaning_flags"]
        .fillna("")
        .astype(str)
        .str.rstrip(";")
        + ";low_information_checked"
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

    low_information_count = int(
        output["low_information"].sum()
    )

    summary = {
        "dataset": dataset_key,
        "total_records": len(output),
        "low_information_records": (
            low_information_count
        ),
        "quality_candidate_records": (
            len(output)
            - low_information_count
        ),
        "low_information_percent": round(
            low_information_count
            / len(output)
            * 100,
            2,
        ),
        "output_file": str(output_path),
    }

    return output, summary


def main() -> None:
    print(
        "Düşük bilgi taşıyan kayıtlar "
        "işaretleniyor..."
    )

    summary_rows = []
    flagged_reports = []

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

        summary_rows.append(summary)

        flagged = output[
            output["low_information"]
        ].copy()

        if not flagged.empty:
            flagged["analysis_dataset"] = (
                dataset_key
            )

            flagged_reports.append(
                flagged[
                    [
                        "analysis_dataset",
                        "record_id",
                        "dataset_name",
                        "text_type",
                        "source",
                        "label_standard",
                        "text_character_count",
                        "meaningful_token_count",
                        "placeholder_count",
                        "placeholder_ratio",
                        "low_information_reasons",
                        "text_basic",
                        "text_destyled",
                    ]
                ]
            )

        print(
            f"Toplam kayıt: {summary['total_records']}"
        )

        print(
            "Düşük bilgi olarak işaretlenen: "
            f"{summary['low_information_records']}"
        )

        print(
            "Kalite adayı: "
            f"{summary['quality_candidate_records']}"
        )

        print(
            "Düşük bilgi oranı: "
            f"%{summary['low_information_percent']}"
        )

        if not flagged.empty:
            print("\nNeden dağılımı:")

            reason_distribution = (
                flagged["low_information_reasons"]
                .str.split(";")
                .explode()
                .value_counts()
            )

            print(reason_distribution)

    summary_report = pd.DataFrame(
        summary_rows
    )

    flagged_report = (
        pd.concat(
            flagged_reports,
            ignore_index=True,
        )
        if flagged_reports
        else pd.DataFrame()
    )

    ANALYSIS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        ANALYSIS_DIRECTORY
        / "low_information_summary.csv"
    )

    flagged_path = (
        ANALYSIS_DIRECTORY
        / "low_information_records.csv"
    )

    summary_report.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    flagged_report.to_csv(
        flagged_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("GENEL ÖZET")
    print("=" * 60)

    print(
        summary_report[
            [
                "dataset",
                "total_records",
                "low_information_records",
                "quality_candidate_records",
                "low_information_percent",
            ]
        ].to_string(index=False)
    )

    print("\nOluşturulan raporlar:")
    print(f"- {summary_path}")
    print(f"- {flagged_path}")

    print(
        "\nHiçbir kayıt silinmedi. "
        "Yalnızca kalite bayrakları eklendi."
    )


if __name__ == "__main__":
    main()