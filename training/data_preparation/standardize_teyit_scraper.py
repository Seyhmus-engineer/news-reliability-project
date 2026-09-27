from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd


LABEL_MAPPING = {
    "YANLIŞ": "fake",
    "DOĞRU": "real",
    "KARMA": "misleading",
    "BELİRSİZ": "uncertain",
}


def find_project_root() -> Path:
    """
    Dosya konumu:
    training/data_preparation/standardize_teyit_scraper.py
    """
    return Path(__file__).resolve().parents[2]


def normalize_whitespace(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value)

    # NBSP ve benzeri görünmeyen boşlukları normal boşluğa çevir
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_claim(value: object) -> str:
    text = normalize_whitespace(value)

    # Örnekler:
    # İDDİA:
    # İDDİA / Fotoğraf:
    # İDDİA / VİDEO:
    text = re.sub(
        r"^\s*İDDİA"
        r"(?:\s*/\s*[^:]{1,60})?"
        r"\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Bazı kayıtlarda iki nokta üst üste kalabilir
    text = re.sub(r"^\s*[:\-–—]\s*", "", text)

    return text.strip()


def normalize_for_duplicate_check(value: object) -> str:
    text = clean_claim(value).lower()

    # Noktalama işaretlerini boşluğa dönüştür
    text = re.sub(
        r"[^a-zçğıöşü0-9]+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_hash(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def create_record_id(url: str, row_number: int) -> str:
    normalized_url = normalize_whitespace(url)

    if normalized_url:
        short_hash = hashlib.sha1(
            normalized_url.encode("utf-8")
        ).hexdigest()[:12]

        return f"teyit_scraper_{short_hash}"

    return f"teyit_scraper_{row_number:06d}"


def is_empty(value: object) -> bool:
    return normalize_whitespace(value) == ""


def standardize_claims(dataframe: pd.DataFrame) -> pd.DataFrame:
    claims = dataframe[
        dataframe["is_claim"].eq(True)
    ].copy()

    claims["claim_raw"] = claims["claim"].apply(
        normalize_whitespace
    )

    claims["text"] = claims["claim"].apply(
        clean_claim
    )

    claims["text_normalized"] = claims["text"].apply(
        normalize_for_duplicate_check
    )

    claims["text_hash"] = claims["text_normalized"].apply(
        create_hash
    )

    claims["label_original"] = claims["verdict"].apply(
        normalize_whitespace
    )

    claims["label"] = claims["label_original"].map(
        LABEL_MAPPING
    )

    unmapped_labels = sorted(
        claims.loc[
            claims["label"].isna(),
            "label_original",
        ].dropna().unique()
    )

    if unmapped_labels:
        raise ValueError(
            "Eşleştirilemeyen etiketler bulundu: "
            + ", ".join(unmapped_labels)
        )

    claims["title"] = claims["title"].apply(
        normalize_whitespace
    )

    claims["author"] = claims["author"].apply(
        normalize_whitespace
    )

    claims["url"] = claims["url"].apply(
        normalize_whitespace
    )

    claims["analysis_text"] = claims["text_all"].apply(
        normalize_whitespace
    )

    claims["tags"] = claims["tags"].apply(
        normalize_whitespace
    )

    claims["evidence_links"] = claims["link_list"].apply(
        normalize_whitespace
    )

    claims["image_links"] = claims["img_link_list"].apply(
        normalize_whitespace
    )

    claims["date"] = pd.to_datetime(
        claims["date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    claims["record_id"] = [
        create_record_id(url, index)
        for index, url in enumerate(
            claims["url"],
            start=1,
        )
    ]

    claims["dataset"] = "teyit_scraper_2019"
    claims["source"] = "teyit.org"
    claims["task"] = "claim_classification"
    claims["text_type"] = "claim"
    claims["split_role"] = "candidate"

    claims["text_char_count"] = claims["text"].str.len()

    claims["text_word_count"] = claims["text"].apply(
        lambda text: len(text.split())
    )

    claims["has_analysis_text"] = (
        claims["analysis_text"].str.len() > 0
    )

    claims["low_information_flag"] = (
        claims["text_char_count"].lt(25)
        | claims["text_word_count"].lt(4)
    )

    output_columns = [
        "record_id",
        "dataset",
        "source",
        "task",
        "text_type",
        "text",
        "claim_raw",
        "text_normalized",
        "text_hash",
        "label",
        "label_original",
        "title",
        "analysis_text",
        "author",
        "date",
        "url",
        "tags",
        "evidence_links",
        "image_links",
        "text_char_count",
        "text_word_count",
        "has_analysis_text",
        "low_information_flag",
        "split_role",
    ]

    return claims[output_columns].reset_index(drop=True)


def standardize_non_claim_articles(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    non_claims = dataframe[
        dataframe["is_claim"].eq(False)
    ].copy()

    for column in non_claims.columns:
        if column != "date":
            non_claims[column] = non_claims[column].apply(
                normalize_whitespace
            )

    non_claims["date"] = pd.to_datetime(
        non_claims["date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    return non_claims.reset_index(drop=True)


def print_profile(claims: pd.DataFrame) -> None:
    exact_duplicate_rows = claims.duplicated(
        subset=["text_hash"],
        keep=False,
    ).sum()

    exact_duplicate_groups = (
        claims.loc[
            claims.duplicated(
                subset=["text_hash"],
                keep=False,
            ),
            "text_hash",
        ].nunique()
    )

    print("=" * 65)
    print("TEYİT SCRAPER STANDARTLAŞTIRMA SONUCU")
    print("=" * 65)

    print(f"Standartlaştırılan iddia: {len(claims):,}")

    print("\nStandart etiket dağılımı:")
    print(
        claims["label"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nOrijinal etiket dağılımı:")
    print(
        claims["label_original"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nMetin uzunlukları:")
    print(
        claims["text_word_count"]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nKalite kontrolleri:")
    print(
        "Boş temiz metin:",
        int(claims["text"].apply(is_empty).sum()),
    )
    print(
        "Düşük bilgi işaretli:",
        int(claims["low_information_flag"].sum()),
    )
    print(
        "Analiz metni bulunan:",
        int(claims["has_analysis_text"].sum()),
    )
    print(
        "Analiz metni bulunmayan:",
        int((~claims["has_analysis_text"]).sum()),
    )
    print(
        "Exact tekrar içindeki kayıt:",
        int(exact_duplicate_rows),
    )
    print(
        "Exact tekrar grubu:",
        int(exact_duplicate_groups),
    )

    print("\nİlk beş temiz iddia:")

    preview_columns = [
        "date",
        "text",
        "label",
        "title",
    ]

    print(
        claims[preview_columns]
        .head(5)
        .to_string(index=False)
    )


def main() -> None:
    project_root = find_project_root()

    input_path = (
        project_root
        / "data"
        / "raw"
        / "teyit_scraper_2019"
        / "articles.csv"
    )

    processed_directory = (
        project_root
        / "data"
        / "processed"
    )

    processed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    claims_output_path = (
        processed_directory
        / "teyit_claims_standardized.csv"
    )

    non_claims_output_path = (
        processed_directory
        / "teyit_non_claim_articles.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Ham veri bulunamadı: {input_path}"
        )

    print("Ham veri okunuyor...")

    dataframe = pd.read_csv(
        input_path,
        encoding="utf-8",
        low_memory=False,
    )

    print("İddia kayıtları standartlaştırılıyor...")

    claims = standardize_claims(dataframe)

    non_claim_articles = standardize_non_claim_articles(
        dataframe
    )

    claims.to_csv(
        claims_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    non_claim_articles.to_csv(
        non_claims_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print_profile(claims)

    print("\nDosyalar kaydedildi:")
    print(f"- {claims_output_path}")
    print(f"- {non_claims_output_path}")


if __name__ == "__main__":
    main()