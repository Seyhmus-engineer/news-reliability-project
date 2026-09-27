from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd


REFERENCE_FILE_CANDIDATES = {
    "article_main": [
        "article_main_standardized.csv",
        "article_main_unified.csv",
        "article_main_quality_cleaned.csv",
    ],
    "facturk": [
        "facturk_standardized.csv",
    ],
    "fctr": [
        "fctr_standardized.csv",
    ],
}


REFERENCE_TEXT_FIELDS = {
    "article_main": [
        "text",
        "title",
        "content",
        "headline",
        "summary",
    ],
    "facturk": [
        "claim",
        "text",
        "title",
        "content",
        "summary",
    ],
    "fctr": [
        "claim",
        "summary",
        "text",
        "title",
    ],
}


ID_FIELD_CANDIDATES = [
    "record_id",
    "claim_id",
    "id",
    "ID",
]


LABEL_FIELD_CANDIDATES = [
    "label",
    "standard_label",
    "target",
    "class",
    "Sınıf",
]


URL_FIELD_CANDIDATES = [
    "url",
    "URL",
    "link",
    "source_url",
]


LABEL_MAPPING = {
    "fake": "fake",
    "false": "fake",
    "yanlış": "fake",
    "yanlis": "fake",
    "0": "fake",

    "real": "real",
    "true": "real",
    "doğru": "real",
    "dogru": "real",
    "1": "real",

    "misleading": "misleading",
    "karma": "misleading",
    "kısmen yanlış": "misleading",
    "kismen yanlis": "misleading",
    "partly false": "misleading",

    "uncertain": "uncertain",
    "belirsiz": "uncertain",
    "sonuçlandırılamadı": "uncertain",
    "sonuclandirilamadi": "uncertain",
    "other": "uncertain",
}


def find_project_root() -> Path:
    """
    Dosyanın konumu:
    training/data_preparation/audit_teyit_exact_overlaps.py
    """
    return Path(__file__).resolve().parents[2]


def turkish_lower(text: str) -> str:
    translation = str.maketrans({
        "I": "ı",
        "İ": "i",
    })

    return text.translate(translation).lower()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value)

    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    text = unicodedata.normalize("NFKC", text)
    text = turkish_lower(text)

    # Teyit kayıtlarında kalabilen başlangıç kalıpları
    text = re.sub(
        r"^\s*iddia"
        r"(?:\s*/\s*[^:]{1,60})?"
        r"\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"[^a-zçğıöşü0-9]+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_url(value: object) -> str:
    if pd.isna(value):
        return ""

    url = str(value).strip()

    if not url:
        return ""

    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = "https://" + url

    try:
        parsed = urlsplit(url)

        hostname = parsed.netloc.lower()
        hostname = re.sub(r"^www\.", "", hostname)

        path = re.sub(r"/+$", "", parsed.path)

        normalized = urlunsplit((
            "",
            hostname,
            path,
            "",
            "",
        ))

        return normalized.lstrip("//")

    except ValueError:
        return turkish_lower(url).rstrip("/")


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""

    label = turkish_lower(
        unicodedata.normalize(
            "NFKC",
            str(value).strip(),
        )
    )

    return LABEL_MAPPING.get(label, label)


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def locate_reference_file(
    processed_directory: Path,
    candidates: list[str],
) -> Path:
    for filename in candidates:
        path = processed_directory / filename

        if path.exists():
            return path

    expected = "\n".join(
        f"- {processed_directory / filename}"
        for filename in candidates
    )

    raise FileNotFoundError(
        "Referans veri dosyası bulunamadı. "
        "Kontrol edilen yollar:\n"
        + expected
    )


def read_csv_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )


def build_reference_tables(
    dataset_name: str,
    file_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataframe = read_csv_file(file_path)

    id_column = first_existing_column(
        dataframe,
        ID_FIELD_CANDIDATES,
    )

    label_column = first_existing_column(
        dataframe,
        LABEL_FIELD_CANDIDATES,
    )

    url_column = first_existing_column(
        dataframe,
        URL_FIELD_CANDIDATES,
    )

    available_text_fields = [
        field
        for field in REFERENCE_TEXT_FIELDS[dataset_name]
        if field in dataframe.columns
    ]

    if not available_text_fields:
        raise ValueError(
            f"{dataset_name} içinde kullanılabilir "
            f"metin sütunu bulunamadı.\n"
            f"Mevcut sütunlar: {list(dataframe.columns)}"
        )

    print(f"\n{dataset_name}:")
    print(f"- Dosya: {file_path}")
    print(f"- Kayıt: {len(dataframe):,}")
    print(f"- Metin sütunları: {available_text_fields}")
    print(f"- Kimlik sütunu: {id_column}")
    print(f"- Etiket sütunu: {label_column}")
    print(f"- URL sütunu: {url_column}")

    row_records = []
    text_records = []

    for row_number, row in dataframe.iterrows():
        if id_column:
            reference_id = str(row[id_column]).strip()
        else:
            reference_id = f"{dataset_name}_{row_number:07d}"

        if not reference_id or reference_id.lower() == "nan":
            reference_id = f"{dataset_name}_{row_number:07d}"

        if label_column:
            reference_label = normalize_label(
                row[label_column]
            )
        else:
            reference_label = ""

        if url_column:
            reference_url = str(row[url_column])
            reference_url_normalized = normalize_url(
                row[url_column]
            )
        else:
            reference_url = ""
            reference_url_normalized = ""

        row_records.append({
            "reference_dataset": dataset_name,
            "reference_row_number": row_number,
            "reference_id": reference_id,
            "reference_label": reference_label,
            "reference_url": reference_url,
            "reference_url_normalized": reference_url_normalized,
        })

        for text_field in available_text_fields:
            original_text = row[text_field]
            normalized = normalize_text(original_text)

            if not normalized:
                continue

            text_records.append({
                "reference_dataset": dataset_name,
                "reference_row_number": row_number,
                "reference_id": reference_id,
                "reference_label": reference_label,
                "reference_url": reference_url,
                "reference_url_normalized": reference_url_normalized,
                "reference_text_field": text_field,
                "reference_text": str(original_text),
                "reference_text_normalized": normalized,
            })

    row_table = pd.DataFrame(row_records)
    text_table = pd.DataFrame(text_records)

    text_table = text_table.drop_duplicates(
        subset=[
            "reference_dataset",
            "reference_id",
            "reference_text_normalized",
        ],
        keep="first",
    ).reset_index(drop=True)

    row_table = row_table.drop_duplicates(
        subset=[
            "reference_dataset",
            "reference_id",
        ],
        keep="first",
    ).reset_index(drop=True)

    return text_table, row_table


def prepare_teyit_claims(
    file_path: Path,
) -> pd.DataFrame:
    dataframe = read_csv_file(file_path)

    required_columns = [
        "record_id",
        "text",
        "label",
        "url",
        "low_information_flag",
    ]

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Teyit dosyasında eksik sütunlar var: "
            + ", ".join(missing)
        )

    dataframe["text_normalized_audit"] = (
        dataframe["text"].apply(normalize_text)
    )

    dataframe["url_normalized_audit"] = (
        dataframe["url"].apply(normalize_url)
    )

    dataframe["label"] = dataframe["label"].apply(
        normalize_label
    )

    dataframe["low_information_flag"] = (
        dataframe["low_information_flag"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    dataframe["empty_text_flag"] = (
        dataframe["text_normalized_audit"].eq("")
    )

    return dataframe


def find_internal_duplicates(
    teyit: pd.DataFrame,
) -> pd.DataFrame:
    duplicate_mask = (
        teyit["text_normalized_audit"].ne("")
        & teyit.duplicated(
            subset=["text_normalized_audit"],
            keep=False,
        )
    )

    duplicates = teyit[
        duplicate_mask
    ].copy()

    duplicates["duplicate_group_size"] = (
        duplicates.groupby(
            "text_normalized_audit"
        )["record_id"].transform("size")
    )

    duplicates["duplicate_rank"] = (
        duplicates.groupby(
            "text_normalized_audit"
        ).cumcount() + 1
    )

    return duplicates.sort_values(
        [
            "text_normalized_audit",
            "record_id",
        ]
    )


def prepare_internal_unique_candidates(
    teyit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality_candidates = teyit[
        ~teyit["empty_text_flag"]
        & ~teyit["low_information_flag"]
    ].copy()

    quality_candidates = quality_candidates.sort_values(
        by=[
            "date",
            "record_id",
        ],
        ascending=[
            False,
            True,
        ],
        na_position="last",
    )

    quality_candidates["internal_duplicate_rank"] = (
        quality_candidates.groupby(
            "text_normalized_audit"
        ).cumcount() + 1
    )

    internal_unique = quality_candidates[
        quality_candidates["internal_duplicate_rank"].eq(1)
    ].copy()

    internal_excluded = quality_candidates[
        quality_candidates["internal_duplicate_rank"].gt(1)
    ].copy()

    return (
        internal_unique.reset_index(drop=True),
        internal_excluded.reset_index(drop=True),
    )


def find_text_overlaps(
    teyit: pd.DataFrame,
    reference_texts: pd.DataFrame,
) -> pd.DataFrame:
    overlaps = teyit.merge(
        reference_texts,
        left_on="text_normalized_audit",
        right_on="reference_text_normalized",
        how="inner",
    )

    if overlaps.empty:
        return overlaps

    overlaps["match_type"] = "exact_text"

    return overlaps


def find_url_overlaps(
    teyit: pd.DataFrame,
    reference_rows: pd.DataFrame,
) -> pd.DataFrame:
    left = teyit[
        teyit["url_normalized_audit"].ne("")
    ].copy()

    right = reference_rows[
        reference_rows[
            "reference_url_normalized"
        ].ne("")
    ].copy()

    overlaps = left.merge(
        right,
        left_on="url_normalized_audit",
        right_on="reference_url_normalized",
        how="inner",
    )

    if overlaps.empty:
        return overlaps

    overlaps["reference_text_field"] = ""
    overlaps["reference_text"] = ""
    overlaps["reference_text_normalized"] = ""
    overlaps["match_type"] = "exact_url"

    return overlaps


def combine_overlap_tables(
    text_overlaps: pd.DataFrame,
    url_overlaps: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = [
        "record_id",
        "text",
        "label",
        "url",
        "reference_dataset",
        "reference_id",
        "reference_label",
        "reference_url",
        "reference_text_field",
        "reference_text",
        "match_type",
    ]

    available_tables = []

    for table in [text_overlaps, url_overlaps]:
        if table.empty:
            continue

        table = table.copy()

        for column in required_columns:
            if column not in table.columns:
                table[column] = ""

        available_tables.append(
            table[required_columns]
        )

    if not available_tables:
        return pd.DataFrame(
            columns=required_columns
        )

    combined = pd.concat(
        available_tables,
        ignore_index=True,
    )

    combined["label_conflict"] = (
        combined["label"].ne("")
        & combined["reference_label"].ne("")
        & combined["label"].ne(
            combined["reference_label"]
        )
    )

    combined = combined.drop_duplicates(
        subset=[
            "record_id",
            "reference_dataset",
            "reference_id",
            "reference_text_field",
            "match_type",
        ],
        keep="first",
    )

    return combined.sort_values(
        [
            "reference_dataset",
            "record_id",
            "match_type",
        ]
    ).reset_index(drop=True)


def print_summary(
    teyit: pd.DataFrame,
    internal_duplicates: pd.DataFrame,
    internal_unique: pd.DataFrame,
    internal_excluded: pd.DataFrame,
    overlaps: pd.DataFrame,
    new_candidates: pd.DataFrame,
) -> None:
    print("\n" + "=" * 70)
    print("TEYİT EXACT ÇAKIŞMA DENETİMİ")
    print("=" * 70)

    print(f"Toplam standart kayıt: {len(teyit):,}")
    print(
        "Boş metin:",
        int(teyit["empty_text_flag"].sum()),
    )
    print(
        "Düşük bilgi işaretli:",
        int(teyit["low_information_flag"].sum()),
    )

    print(
        "İç exact tekrardaki kayıt:",
        len(internal_duplicates),
    )

    print(
        "İç exact tekrar grubu:",
        internal_duplicates[
            "text_normalized_audit"
        ].nunique(),
    )

    print(
        "Kalite + iç tekilleştirme sonrası:",
        len(internal_unique),
    )

    print(
        "İç tekrar nedeniyle dışlanan:",
        len(internal_excluded),
    )

    if overlaps.empty:
        print("\nMevcut veri setleriyle exact çakışma bulunmadı.")
    else:
        print("\nExact çakışan benzersiz Teyit kayıtları:")

        overlap_counts = (
            overlaps.groupby(
                "reference_dataset"
            )["record_id"]
            .nunique()
            .sort_values(ascending=False)
        )

        print(overlap_counts.to_string())

        print("\nEşleşme türleri:")

        print(
            overlaps["match_type"]
            .value_counts()
            .to_string()
        )

        print("\nEtiket çatışmaları:")

        conflict_summary = (
            overlaps.groupby(
                "reference_dataset"
            )["label_conflict"]
            .sum()
            .astype(int)
        )

        print(conflict_summary.to_string())

        total_overlap_records = (
            overlaps["record_id"].nunique()
        )

        total_conflict_records = (
            overlaps.loc[
                overlaps["label_conflict"],
                "record_id",
            ].nunique()
        )

        print(
            "\nToplam çakışan benzersiz Teyit kaydı:",
            total_overlap_records,
        )

        print(
            "Etiket çatışması bulunan benzersiz Teyit kaydı:",
            total_conflict_records,
        )

    print(
        "\nExact çakışması olmayan yeni aday:",
        len(new_candidates),
    )

    print("\nYeni aday etiket dağılımı:")

    print(
        new_candidates["label"]
        .value_counts()
        .to_string()
    )


def main() -> None:
    project_root = find_project_root()

    processed_directory = (
        project_root
        / "data"
        / "processed"
    )

    audit_directory = (
        project_root
        / "data"
        / "audits"
        / "teyit_scraper_2019"
    )

    audit_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    teyit_path = (
        processed_directory
        / "teyit_claims_standardized.csv"
    )

    if not teyit_path.exists():
        raise FileNotFoundError(
            f"Teyit dosyası bulunamadı: {teyit_path}"
        )

    print("Teyit verisi hazırlanıyor...")

    teyit = prepare_teyit_claims(
        teyit_path
    )

    internal_duplicates = find_internal_duplicates(
        teyit
    )

    (
        internal_unique,
        internal_excluded,
    ) = prepare_internal_unique_candidates(
        teyit
    )

    all_reference_texts = []
    all_reference_rows = []

    print("\nReferans veri setleri okunuyor...")

    for dataset_name, filenames in (
        REFERENCE_FILE_CANDIDATES.items()
    ):
        file_path = locate_reference_file(
            processed_directory,
            filenames,
        )

        text_table, row_table = build_reference_tables(
            dataset_name,
            file_path,
        )

        all_reference_texts.append(text_table)
        all_reference_rows.append(row_table)

    reference_texts = pd.concat(
        all_reference_texts,
        ignore_index=True,
    )

    reference_rows = pd.concat(
        all_reference_rows,
        ignore_index=True,
    )

    print("\nExact metin eşleşmeleri aranıyor...")

    text_overlaps = find_text_overlaps(
        internal_unique,
        reference_texts,
    )

    print("Exact URL eşleşmeleri aranıyor...")

    url_overlaps = find_url_overlaps(
        internal_unique,
        reference_rows,
    )

    overlaps = combine_overlap_tables(
        text_overlaps,
        url_overlaps,
    )

    overlapping_record_ids = set(
        overlaps["record_id"].tolist()
    )

    new_candidates = internal_unique[
        ~internal_unique["record_id"].isin(
            overlapping_record_ids
        )
    ].copy()

    internal_duplicates.to_csv(
        audit_directory
        / "teyit_internal_exact_duplicates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    internal_excluded.to_csv(
        audit_directory
        / "teyit_internal_duplicate_excluded.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overlaps.to_csv(
        audit_directory
        / "teyit_cross_dataset_exact_overlaps.csv",
        index=False,
        encoding="utf-8-sig",
    )

    new_candidates.to_csv(
        audit_directory
        / "teyit_exact_new_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(
        teyit=teyit,
        internal_duplicates=internal_duplicates,
        internal_unique=internal_unique,
        internal_excluded=internal_excluded,
        overlaps=overlaps,
        new_candidates=new_candidates,
    )

    print("\nDenetim dosyaları:")

    for filename in [
        "teyit_internal_exact_duplicates.csv",
        "teyit_internal_duplicate_excluded.csv",
        "teyit_cross_dataset_exact_overlaps.csv",
        "teyit_exact_new_candidates.csv",
    ]:
        print(f"- {audit_directory / filename}")


if __name__ == "__main__":
    main()