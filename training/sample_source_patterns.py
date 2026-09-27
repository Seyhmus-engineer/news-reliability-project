from pathlib import Path
import re
import unicodedata

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

UNIFIED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "source_pattern_examples.csv"
)


DATASET_FILES = {
    "article_main": "article_main_unified.csv",
    "facturk": "facturk_training_candidates.csv",
    "fctr": "fctr_unified.csv",
    "mide22": "mide22_unified.csv",
    "satiretr": "satiretr_unified.csv",
}


# Bu kalıplar şu anda sadece örnek bulmak için kullanılıyor.
# Henüz metinlerden hiçbir şey silinmeyecek.
PATTERNS = {
    "social_media_claim": (
        r"\bsosyal medyada\s+"
        r"(?:paylaşılan|yayılan|yer alan|dolaşıma giren)"
    ),

    "claim_statement": (
        r"\biddia\s+edildi\b|"
        r"\biddia\s+ediliyor\b|"
        r"\biddia\s+edilmiş\b|"
        r"\biddiası\b"
    ),

    "claim_review": (
        r"\biddia\s+incelendiğinde\b|"
        r"\biddianın\s+incelenmesi\b|"
        r"\biddia\s+incelemesi\b"
    ),

    "analysis_heading": (
        r"(?:^|\n)\s*analiz\s*:?"
    ),

    "findings_heading": (
        r"(?:^|\n)\s*bulgular(?:ımız)?\s*:?"
    ),

    "verification_result": (
        r"\bteyit\s+sonucu\b|"
        r"\bteyit\s+edildi\b|"
        r"\bteyit\s+edilemedi\b"
    ),

    "teyit_brand": (
        r"\bteyit(?:\.org)?\b"
    ),

    "dogruluk_payi_brand": (
        r"\bdoğruluk\s+payı\b|"
        r"\bdogrulukpayi\.com\b"
    ),

    "dogrula_brand": (
        r"\bdoğrula(?:\.org)?\b"
    ),

    "malumatfurus_brand": (
        r"\bmalumatfuruş\b"
    ),

    "trt_haber_brand": (
        r"\btrt\s+haber\b"
    ),

    "anadolu_ajansi_brand": (
        r"\banadolu\s+ajansı\b"
    ),

    "agency_city_signature": (
        r"\b[A-Za-zÇĞİÖŞÜçğıöşü]+\s*"
        r"\((?:AA|DHA|İHA)\)\s*[-–—:]"
    ),

    "aa_news_phrase": (
        r"\bAA(?:'nın|’nın|nın)?\s+haberine\s+göre\b"
    ),

    "aa_reporter_phrase": (
        r"\bAA\s+muhabir(?:ine|inin|i)\b"
    ),

    "reporter_phrase": (
        r"\b(?:TRT\s+Haber\s+)?"
        r"muhabir(?:inin|i|lerince|lerden)?\b"
    ),

    "twitter_mention": (
        r"(?<!\w)@[A-Za-z0-9_]+"
    ),

    "hashtag": (
        r"(?<!\w)#[^\s#]+"
    ),

    "web_url": (
        r"https?://\S+|www\.\S+"
    ),

    "numeric_date": (
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"
    ),

    "platform_metadata": (
        r"\b(?:beğeni|retweet|görüntülenme|"
        r"paylaşım sayısı|takipçi)\b"
    ),
}


EXAMPLES_PER_DATASET_PATTERN = 8
CONTEXT_CHARACTER_COUNT = 140


def normalize_unicode(text: object) -> str:
    if pd.isna(text):
        return ""

    return unicodedata.normalize(
        "NFKC",
        str(text),
    ).strip()


def extract_match_context(
    text: str,
    pattern: str,
) -> tuple[str, str]:
    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if match is None:
        return "", ""

    start = max(
        0,
        match.start() - CONTEXT_CHARACTER_COUNT,
    )

    end = min(
        len(text),
        match.end() + CONTEXT_CHARACTER_COUNT,
    )

    context = text[start:end]

    context = re.sub(
        r"\s+",
        " ",
        context,
    ).strip()

    if start > 0:
        context = "... " + context

    if end < len(text):
        context = context + " ..."

    return match.group(0), context


def load_datasets() -> pd.DataFrame:
    loaded_dataframes = []

    for dataset_key, file_name in DATASET_FILES.items():
        file_path = UNIFIED_DIRECTORY / file_name

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dosya bulunamadı: {file_path}"
            )

        dataframe = pd.read_csv(file_path)

        required_columns = {
            "record_id",
            "dataset_name",
            "source",
            "label_standard",
            "text_basic",
        }

        missing_columns = required_columns.difference(
            dataframe.columns
        )

        if missing_columns:
            raise ValueError(
                f"{file_name} dosyasında eksik sütunlar: "
                f"{sorted(missing_columns)}"
            )

        selected = dataframe[
            [
                "record_id",
                "dataset_name",
                "source",
                "label_standard",
                "text_basic",
            ]
        ].copy()

        selected["analysis_dataset"] = dataset_key

        loaded_dataframes.append(selected)

    combined = pd.concat(
        loaded_dataframes,
        ignore_index=True,
    )

    combined["text_analysis"] = (
        combined["text_basic"]
        .apply(normalize_unicode)
    )

    return combined


def create_pattern_examples(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    example_rows = []

    for pattern_name, pattern in PATTERNS.items():
        matched_mask = (
            dataframe["text_analysis"]
            .str.contains(
                pattern,
                regex=True,
                na=False,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        )

        matched_data = dataframe[
            matched_mask
        ].copy()

        if matched_data.empty:
            continue

        for dataset_name, dataset_group in matched_data.groupby(
            "analysis_dataset"
        ):
            sample_size = min(
                EXAMPLES_PER_DATASET_PATTERN,
                len(dataset_group),
            )

            sampled = dataset_group.sample(
                n=sample_size,
                random_state=42,
            )

            for _, row in sampled.iterrows():
                matched_text, context = extract_match_context(
                    text=row["text_analysis"],
                    pattern=pattern,
                )

                example_rows.append(
                    {
                        "pattern_name": pattern_name,
                        "analysis_dataset": dataset_name,
                        "record_id": row["record_id"],
                        "dataset_name": row["dataset_name"],
                        "source": row["source"],
                        "label_standard": row[
                            "label_standard"
                        ],
                        "matched_text": matched_text,
                        "context": context,
                        "text_length": len(
                            row["text_analysis"]
                        ),
                    }
                )

    examples = pd.DataFrame(example_rows)

    if examples.empty:
        return examples

    return examples.sort_values(
        by=[
            "pattern_name",
            "analysis_dataset",
            "record_id",
        ]
    ).reset_index(drop=True)


def main() -> None:
    print("Kaynak kalıpları için gerçek örnekler çıkarılıyor...")

    dataframe = load_datasets()

    print(f"\nToplam incelenen kayıt: {len(dataframe)}")

    examples = create_pattern_examples(
        dataframe
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    examples.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 65)
    print("SONUÇ")
    print("=" * 65)

    print(f"Çıkarılan örnek sayısı: {len(examples)}")
    print(f"Dosya yolu: {OUTPUT_PATH}")

    if examples.empty:
        print("\nHiçbir kalıp örneği bulunamadı.")
        return

    print("\nKalıplara göre örnek sayıları:")
    print(
        examples["pattern_name"]
        .value_counts()
    )

    print("\nVeri setlerine göre örnek sayıları:")
    print(
        examples["analysis_dataset"]
        .value_counts()
    )

    print("\nİlk örnekler:")
    print(
        examples[
            [
                "pattern_name",
                "analysis_dataset",
                "source",
                "label_standard",
                "matched_text",
                "context",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print(
        "\nBu işlem yalnızca örnek çıkardı; "
        "hiçbir veri değiştirilmedi."
    )


if __name__ == "__main__":
    main()