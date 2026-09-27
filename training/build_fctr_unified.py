from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fctr_standardized.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified"
    / "fctr_unified.csv"
)


DATASET_NAME = "fctr"
TASK_NAME = "claim_veracity_multiclass"
TEXT_TYPE = "fact_checked_claim"


def normalize_whitespace(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def extract_domain(url_value: object) -> str:
    if pd.isna(url_value):
        return "unknown"

    url_text = str(url_value).strip()

    if not url_text:
        return "unknown"

    try:
        parsed_url = urlparse(url_text)

        domain = parsed_url.netloc.lower()

        if not domain and "://" not in url_text:
            parsed_url = urlparse(
                f"https://{url_text}"
            )
            domain = parsed_url.netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain or "unknown"

    except ValueError:
        return "unknown"


def combine_reference_text(
    evidence: pd.Series,
    summary: pd.Series,
) -> pd.Series:
    evidence_clean = normalize_whitespace(evidence)
    summary_clean = normalize_whitespace(summary)

    return (
        "Özet: "
        + summary_clean
        + "\nKanıt: "
        + evidence_clean
    ).str.strip()


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"FCTR dosyası bulunamadı: {INPUT_PATH}"
        )

    print("FCTR ortak şemaya dönüştürülüyor...")

    dataframe = pd.read_csv(INPUT_PATH)

    required_columns = {
        "claim_id",
        "claim",
        "evidence",
        "summary",
        "standard_label",
        "label",
        "url",
        "date",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Eksik sütunlar: {sorted(missing_columns)}"
        )

    print(f"\nGirdi kayıt sayısı: {len(dataframe)}")

    dataframe = dataframe.dropna(
        subset=[
            "claim",
            "standard_label",
        ]
    ).copy()

    claim_raw = (
        dataframe["claim"]
        .astype(str)
        .str.strip()
    )

    claim_basic = normalize_whitespace(
        dataframe["claim"]
    )

    label_standard = (
        dataframe["standard_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    allowed_labels = {
        "fake",
        "real",
        "misleading",
        "uncertain",
    }

    invalid_labels = set(
        label_standard.unique()
    ).difference(allowed_labels)

    if invalid_labels:
        raise ValueError(
            f"Geçersiz etiketler: {sorted(invalid_labels)}"
        )

    source_domains = (
        dataframe["url"]
        .apply(extract_domain)
    )

    reference_content = combine_reference_text(
        evidence=dataframe["evidence"],
        summary=dataframe["summary"],
    )

    unified = pd.DataFrame()

    unified["record_id"] = [
        f"fctr_{index:06d}"
        for index in range(1, len(dataframe) + 1)
    ]

    unified["dataset_name"] = DATASET_NAME
    unified["task_name"] = TASK_NAME
    unified["text_type"] = TEXT_TYPE

    # FCTR'de ayrı bir haber başlığı bulunmuyor.
    unified["title_raw"] = pd.NA

    # Model girdisi yalnızca doğrulanan iddiadır.
    unified["content_raw"] = claim_raw.values
    unified["text_raw"] = claim_raw.values
    unified["text_basic"] = claim_basic.values

    # Kaynak izi temizliği henüz uygulanmadı.
    unified["text_destyled"] = claim_basic.values

    unified["label_original"] = (
        dataframe["label"]
        .astype(str)
        .values
    )

    unified["label_standard"] = (
        label_standard.values
    )

    # URL içindeki alan adından kaynak bilgisi çıkarılır.
    unified["source"] = source_domains.values

    unified["url"] = normalize_whitespace(
        dataframe["url"]
    ).replace("", pd.NA).values

    unified["published_date"] = normalize_whitespace(
        dataframe["date"]
    ).replace("", pd.NA).values

    unified["cleaning_flags"] = (
        "whitespace_normalized;"
        "claim_only_model_text;"
        "source_extracted_from_url"
    )

    unified["duplicate_group_id"] = pd.NA
    unified["split_role"] = "external_test_candidate"

    # Özet ve kanıt denetim için saklanır,
    # fakat model girdisine dahil edilmez.
    unified["reference_content_raw"] = (
        reference_content.values
    )

    # Orijinal FCTR kimliği de kaybolmasın.
    unified["original_record_id"] = (
        dataframe["claim_id"]
        .astype(str)
        .values
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    unified.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFCTR ortak veri dosyası oluşturuldu.")
    print(f"Dosya yolu: {OUTPUT_PATH}")
    print(f"Toplam kayıt: {len(unified)}")

    print("\nEtiket dağılımı:")
    print(
        unified["label_standard"]
        .value_counts(dropna=False)
    )

    print("\nMetin türü dağılımı:")
    print(
        unified["text_type"]
        .value_counts(dropna=False)
    )

    print("\nURL'den çıkarılan kaynak dağılımı:")
    print(
        unified["source"]
        .value_counts(dropna=False)
    )

    print("\nZorunlu alanlardaki boş değerler:")
    print(
        unified[
            [
                "record_id",
                "content_raw",
                "text_basic",
                "label_standard",
                "source",
            ]
        ]
        .isnull()
        .sum()
    )

    print("\nOrtalama iddia uzunluğu:")
    print(
        round(
            unified["text_basic"]
            .str.len()
            .mean(),
            2,
        )
    )

    print("\nİlk iki kayıt:")
    print(
        unified[
            [
                "record_id",
                "dataset_name",
                "text_type",
                "label_standard",
                "source",
                "split_role",
            ]
        ].head(2)
    )


if __name__ == "__main__":
    main()