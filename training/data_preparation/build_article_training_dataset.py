from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

REAL_ID_CANDIDATES = [
    "record_id",
    "final_pair_id",
    "pair_id",
    "original_id",
    "news_id",
    "article_id",
    "id",
    "ID",
]

REAL_TEXT_CANDIDATES = [
    "text_basic",
    "clean_text",
    "cleaned_text",
    "text_clean",
    "haber_govdesi_clean",
    "haber_gövdesi_clean",
    "text",
    "content",
    "article_text",
    "news_text",
    "body",
    "Haber Gövdesi",
    "haber_govdesi",
    "haber_gövdesi",
]


def find_project_root() -> Path:
    """
    Dosyanın konumu:

    training/data_preparation/build_article_training_dataset.py
    """
    return Path(__file__).resolve().parents[2]


def normalize_column_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value))

    text = text.translate(
        str.maketrans(
            {
                "İ": "i",
                "I": "ı",
            }
        )
    ).lower()

    text = re.sub(r"[^a-z0-9çğıöşü]+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    column_type: str,
) -> str:
    normalized_columns = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized_candidate = normalize_column_name(candidate)

        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]

    raise ValueError(
        f"{column_type} sütunu bulunamadı.\n"
        f"Mevcut sütunlar:\n{list(dataframe.columns)}"
    )


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    # CSV okumasında bazı sayısal kimlikler 123.0 olabilir.
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    return text


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))

    text = (
        text.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def count_words(text: str) -> int:
    return len(
        re.findall(
            r"[A-Za-zÇĞİÖŞÜçğıöşüÂâÎîÛû]+"
            r"(?:['’\-][A-Za-zÇĞİÖŞÜçğıöşüÂâÎîÛû]+)*",
            text,
        )
    )


def load_real_data(
    path: Path,
) -> tuple[pd.DataFrame, str, str]:
    dataframe = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    id_column = find_column(
        dataframe,
        REAL_ID_CANDIDATES,
        "TRNews kimlik",
    )

    text_column = find_column(
        dataframe,
        REAL_TEXT_CANDIDATES,
        "TRNews metin",
    )

    dataframe["pair_id"] = dataframe[id_column].apply(
        normalize_id
    )

    dataframe["real_text"] = dataframe[text_column].apply(
        clean_text
    )

    return dataframe, id_column, text_column


def load_synthetic_data(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_columns = [
        "original_id",
        "synthetic_text",
        "method",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Sentetik dosyada eksik sütunlar bulundu: "
            + ", ".join(missing_columns)
        )

    dataframe["pair_id"] = dataframe["original_id"].apply(
        normalize_id
    )

    dataframe["synthetic_text"] = dataframe[
        "synthetic_text"
    ].apply(clean_text)

    return dataframe


def validate_input_data(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> None:
    real_empty_id = real_data["pair_id"].eq("").sum()
    synthetic_empty_id = synthetic_data["pair_id"].eq("").sum()

    real_empty_text = real_data["real_text"].eq("").sum()
    synthetic_empty_text = synthetic_data[
        "synthetic_text"
    ].eq("").sum()

    real_duplicate_ids = real_data["pair_id"].duplicated(
        keep=False
    ).sum()

    synthetic_duplicate_ids = synthetic_data[
        "pair_id"
    ].duplicated(
        keep=False
    ).sum()

    print("\nGirdi kontrolleri:")
    print("- TRNews boş kimlik:", real_empty_id)
    print("- Sentetik boş kimlik:", synthetic_empty_id)
    print("- TRNews boş metin:", real_empty_text)
    print("- Sentetik boş metin:", synthetic_empty_text)
    print("- TRNews tekrar eden kimlik:", real_duplicate_ids)
    print(
        "- Sentetik tekrar eden kimlik:",
        synthetic_duplicate_ids,
    )

    if synthetic_duplicate_ids:
        raise ValueError(
            "Sentetik dosyada tekrar eden original_id bulundu."
        )


def match_pairs(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    unmatched_path: Path,
) -> pd.DataFrame:
    real_lookup = real_data[
        [
            "pair_id",
            "real_text",
        ]
    ].copy()

    real_lookup = real_lookup[
        real_lookup["pair_id"].ne("")
        & real_lookup["real_text"].ne("")
    ]

    # Aynı kimlik varsa ilk kaliteli kayıt tutulur.
    real_lookup = real_lookup.drop_duplicates(
        subset=["pair_id"],
        keep="first",
    )

    synthetic_lookup = synthetic_data[
        [
            "pair_id",
            "synthetic_text",
            "method",
        ]
    ].copy()

    synthetic_lookup = synthetic_lookup[
        synthetic_lookup["pair_id"].ne("")
        & synthetic_lookup["synthetic_text"].ne("")
    ]

    merged = synthetic_lookup.merge(
        real_lookup,
        on="pair_id",
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    unmatched = merged[
        merged["_merge"].ne("both")
    ].copy()

    unmatched.to_csv(
        unmatched_path,
        index=False,
        encoding="utf-8-sig",
    )

    matched = merged[
        merged["_merge"].eq("both")
    ].drop(columns=["_merge"])

    return matched.reset_index(drop=True)


def create_pair_splits(
    matched_pairs: pd.DataFrame,
) -> pd.DataFrame:
    pair_table = matched_pairs[
        [
            "pair_id",
            "method",
        ]
    ].drop_duplicates(
        subset=["pair_id"]
    )

    train_pairs, temporary_pairs = train_test_split(
        pair_table,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=pair_table["method"],
    )

    validation_pairs, test_pairs = train_test_split(
        temporary_pairs,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temporary_pairs["method"],
    )

    train_pairs = train_pairs.copy()
    validation_pairs = validation_pairs.copy()
    test_pairs = test_pairs.copy()

    train_pairs["split"] = "train"
    validation_pairs["split"] = "validation"
    test_pairs["split"] = "test"

    split_table = pd.concat(
        [
            train_pairs,
            validation_pairs,
            test_pairs,
        ],
        ignore_index=True,
    )

    return split_table


def build_long_format_dataset(
    matched_pairs: pd.DataFrame,
    split_table: pd.DataFrame,
) -> pd.DataFrame:
    pairs = matched_pairs.merge(
        split_table[
            [
                "pair_id",
                "split",
            ]
        ],
        on="pair_id",
        how="inner",
        validate="one_to_one",
    )

    real_rows = pd.DataFrame(
        {
            "pair_id": pairs["pair_id"],
            "text": pairs["real_text"],
            "label": "real",
            "label_id": 0,
            "source": "trnews_2025",
            "method": "original",
            "is_synthetic": False,
            "split": pairs["split"],
        }
    )

    fake_rows = pd.DataFrame(
        {
            "pair_id": pairs["pair_id"],
            "text": pairs["synthetic_text"],
            "label": "fake",
            "label_id": 1,
            "source": "synthetic_50k",
            "method": pairs["method"],
            "is_synthetic": True,
            "split": pairs["split"],
        }
    )

    dataset = pd.concat(
        [
            real_rows,
            fake_rows,
        ],
        ignore_index=True,
    )

    dataset["word_count"] = dataset["text"].apply(
        count_words
    )

    dataset = dataset.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return dataset


def validate_final_dataset(
    dataset: pd.DataFrame,
) -> None:
    print("\n" + "=" * 65)
    print("EĞİTİM VERİ SETİ KONTROLÜ")
    print("=" * 65)

    print("Toplam kayıt:", f"{len(dataset):,}")
    print(
        "Benzersiz haber çifti:",
        f"{dataset['pair_id'].nunique():,}",
    )

    print("\nEtiket dağılımı:")
    print(
        dataset["label"]
        .value_counts()
        .to_string()
    )

    print("\nSplit ve etiket dağılımı:")
    print(
        pd.crosstab(
            dataset["split"],
            dataset["label"],
        ).to_string()
    )

    print("\nÜretim yöntemi dağılımı:")
    print(
        dataset[
            dataset["label"].eq("fake")
        ]["method"]
        .value_counts()
        .to_string()
    )

    print("\nKelime uzunluğu özeti:")
    print(
        dataset.groupby("label")["word_count"]
        .describe()[
            [
                "count",
                "mean",
                "50%",
                "min",
                "max",
            ]
        ]
        .round(2)
        .to_string()
    )

    train_ids = set(
        dataset.loc[
            dataset["split"].eq("train"),
            "pair_id",
        ]
    )

    validation_ids = set(
        dataset.loc[
            dataset["split"].eq("validation"),
            "pair_id",
        ]
    )

    test_ids = set(
        dataset.loc[
            dataset["split"].eq("test"),
            "pair_id",
        ]
    )

    pair_leakage = (
        bool(train_ids & validation_ids)
        or bool(train_ids & test_ids)
        or bool(validation_ids & test_ids)
    )

    print("\nÇiftler arasında split sızıntısı:", pair_leakage)

    rows_per_pair = dataset.groupby(
        "pair_id"
    ).size()

    invalid_pairs = rows_per_pair[
        rows_per_pair.ne(2)
    ]

    print(
        "İki kayıttan oluşmayan çift:",
        len(invalid_pairs),
    )

    if pair_leakage:
        raise RuntimeError(
            "Train/validation/test arasında pair_id sızıntısı bulundu."
        )

    if len(invalid_pairs):
        raise RuntimeError(
            "Bazı pair_id değerlerinde gerçek-sentetik çift eksik."
        )


def save_splits(
    dataset: pd.DataFrame,
    output_directory: Path,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_path = (
        output_directory
        / "article_pairs_full_60k.csv"
    )

    train_path = (
        output_directory
        / "article_train_48k.csv"
    )

    validation_path = (
        output_directory
        / "article_validation_6k.csv"
    )

    test_path = (
        output_directory
        / "article_test_6k.csv"
    )

    dataset.to_csv(
        full_path,
        index=False,
        encoding="utf-8-sig",
    )

    dataset[
        dataset["split"].eq("train")
    ].to_csv(
        train_path,
        index=False,
        encoding="utf-8-sig",
    )

    dataset[
        dataset["split"].eq("validation")
    ].to_csv(
        validation_path,
        index=False,
        encoding="utf-8-sig",
    )

    dataset[
        dataset["split"].eq("test")
    ].to_csv(
        test_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nOluşturulan dosyalar:")
    print("-", full_path)
    print("-", train_path)
    print("-", validation_path)
    print("-", test_path)


def main() -> None:
    project_root = find_project_root()

    real_path = (
        project_root
        / "data"
        / "processed"
        / "trnews_2025_final_real_pool.csv"
    )

    synthetic_path = (
        project_root
        / "data"
        / "processed"
        / "synthetic_50k_training_balanced_30k.csv"
    )

    output_directory = (
        project_root
        / "data"
        / "training"
        / "article_60k"
    )

    unmatched_path = (
        output_directory
        / "unmatched_synthetic_ids.csv"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not real_path.exists():
        raise FileNotFoundError(
            f"TRNews dosyası bulunamadı: {real_path}"
        )

    if not synthetic_path.exists():
        raise FileNotFoundError(
            f"Sentetik dosya bulunamadı: {synthetic_path}"
        )

    print("TRNews gerçek haberleri okunuyor...")

    real_data, real_id_column, real_text_column = (
        load_real_data(real_path)
    )

    print("Kullanılan TRNews kimlik sütunu:", real_id_column)
    print("Kullanılan TRNews metin sütunu:", real_text_column)
    print("TRNews kayıt sayısı:", f"{len(real_data):,}")

    print("\nSentetik veriler okunuyor...")

    synthetic_data = load_synthetic_data(
        synthetic_path
    )

    print(
        "Sentetik kayıt sayısı:",
        f"{len(synthetic_data):,}",
    )

    validate_input_data(
        real_data,
        synthetic_data,
    )

    print("\nGerçek ve sentetik kayıtlar eşleştiriliyor...")

    matched_pairs = match_pairs(
        real_data,
        synthetic_data,
        unmatched_path,
    )

    print(
        "Eşleşen haber çifti:",
        f"{len(matched_pairs):,}",
    )

    unmatched_count = (
        len(synthetic_data)
        - len(matched_pairs)
    )

    print(
        "Eşleşmeyen sentetik kayıt:",
        f"{unmatched_count:,}",
    )

    if matched_pairs.empty:
        raise RuntimeError(
            "Hiçbir sentetik kayıt TRNews ile eşleşmedi. "
            "Kimlik sütunları kontrol edilmeli."
        )

    split_table = create_pair_splits(
        matched_pairs
    )

    dataset = build_long_format_dataset(
        matched_pairs,
        split_table,
    )

    validate_final_dataset(dataset)

    save_splits(
        dataset,
        output_directory,
    )


if __name__ == "__main__":
    main()