from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

UNIFIED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)

DATASET_FILES = {
    "article_main": "article_main_unified.csv",
    "facturk": "facturk_training_candidates.csv",
    "fctr": "fctr_unified.csv",
    "mide22": "mide22_unified.csv",
    "satiretr": "satiretr_unified.csv",
}


# Şimdilik yalnızca aradığımız kaynak ve platform izleri.
# Bu kalıplar henüz metinden silinmeyecek.
PATTERNS = {
    "social_media_claim": (
        r"\bsosyal medyada\s+"
        r"(?:paylaşılan|yayılan|yer alan|dolaşıma giren)"
    ),
    "claim_according_to": (
        r"\biddia(?:ya|sına|sına göre|ya göre)\b"
    ),
    "claim_reviewed": (
        r"\biddia(?:nın)?\s+incelenmesi\b|"
        r"\biddia\s+incelendiğinde\b"
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
        r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s*"
        r"\((?:AA|DHA|İHA)\)\s*[-–—:]"
    ),
    "aa_news_phrase": (
        r"\bAA(?:'nın|’nın|nın)?\s+haberine\s+göre\b"
    ),
    "reporter_phrase": (
        r"\bmuhabir(?:inin|i|lerince|lerden)?\b"
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
}


def normalize_unicode(text: object) -> str:
    if pd.isna(text):
        return ""

    return unicodedata.normalize(
        "NFKC",
        str(text),
    ).strip()


def load_datasets() -> pd.DataFrame:
    loaded = []

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

        loaded.append(selected)

    combined = pd.concat(
        loaded,
        ignore_index=True,
    )

    combined["text_analysis"] = (
        combined["text_basic"]
        .apply(normalize_unicode)
    )

    return combined


def create_pattern_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    reports = []

    group_columns = [
        "analysis_dataset",
        "source",
        "label_standard",
    ]

    for pattern_name, pattern in PATTERNS.items():
        matched = dataframe["text_analysis"].str.contains(
            pattern,
            regex=True,
            na=False,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        temporary = dataframe[
            group_columns
        ].copy()

        temporary["matched"] = matched.astype(int)

        grouped = (
            temporary
            .groupby(
                group_columns,
                dropna=False,
            )
            .agg(
                record_count=("matched", "size"),
                match_count=("matched", "sum"),
            )
            .reset_index()
        )

        grouped = grouped[
            grouped["match_count"] > 0
        ].copy()

        if grouped.empty:
            continue

        grouped["pattern_name"] = pattern_name

        grouped["match_rate_percent"] = (
            grouped["match_count"]
            .div(grouped["record_count"])
            .mul(100)
            .round(2)
        )

        reports.append(grouped)

    if not reports:
        return pd.DataFrame(
            columns=[
                "analysis_dataset",
                "source",
                "label_standard",
                "record_count",
                "match_count",
                "pattern_name",
                "match_rate_percent",
            ]
        )

    result = pd.concat(
        reports,
        ignore_index=True,
    )

    return result.sort_values(
        by=[
            "analysis_dataset",
            "match_rate_percent",
            "match_count",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )


def calculate_discriminative_ngrams(
    dataframe: pd.DataFrame,
    group_column: str,
    top_n: int = 30,
) -> pd.DataFrame:
    results = []

    for dataset_name, dataset in dataframe.groupby(
        "analysis_dataset"
    ):
        dataset = dataset.reset_index(drop=True)

        groups = (
            dataset[group_column]
            .fillna("unknown")
            .astype(str)
        )

        if groups.nunique() < 2:
            continue

        document_count = len(dataset)

        min_df = max(
            2,
            min(
                20,
                round(document_count * 0.001),
            ),
        )

        vectorizer = CountVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            min_df=min_df,
            max_df=0.98,
            max_features=30000,
            binary=True,
            token_pattern=r"(?u)\b\w\w+\b",
        )

        try:
            matrix = vectorizer.fit_transform(
                dataset["text_analysis"]
            )
        except ValueError:
            continue

        features = np.array(
            vectorizer.get_feature_names_out()
        )

        for group_value in sorted(groups.unique()):
            group_mask = groups.eq(
                group_value
            ).to_numpy()

            group_document_count = int(
                group_mask.sum()
            )

            other_document_count = int(
                (~group_mask).sum()
            )

            if (
                group_document_count < 20
                or other_document_count < 20
            ):
                continue

            group_rate = np.asarray(
                matrix[group_mask]
                .mean(axis=0)
            ).ravel()

            other_rate = np.asarray(
                matrix[~group_mask]
                .mean(axis=0)
            ).ravel()

            difference = group_rate - other_rate

            candidate_indexes = np.where(
                (group_rate >= 0.02)
                & (difference > 0)
            )[0]

            if len(candidate_indexes) == 0:
                continue

            sorted_indexes = candidate_indexes[
                np.argsort(
                    difference[candidate_indexes]
                )[::-1]
            ][:top_n]

            for rank, feature_index in enumerate(
                sorted_indexes,
                start=1,
            ):
                results.append(
                    {
                        "analysis_dataset": dataset_name,
                        "group_column": group_column,
                        "group_value": group_value,
                        "rank": rank,
                        "ngram": features[feature_index],
                        "group_rate_percent": round(
                            group_rate[feature_index] * 100,
                            2,
                        ),
                        "other_rate_percent": round(
                            other_rate[feature_index] * 100,
                            2,
                        ),
                        "difference_percent": round(
                            difference[feature_index] * 100,
                            2,
                        ),
                        "group_document_count": (
                            group_document_count
                        ),
                        "other_document_count": (
                            other_document_count
                        ),
                    }
                )

    return pd.DataFrame(results)


def main() -> None:
    print("Kaynak ve tarz parmak izleri analiz ediliyor...")

    dataframe = load_datasets()

    print(f"\nToplam incelenen kayıt: {len(dataframe)}")

    pattern_report = create_pattern_report(
        dataframe
    )

    source_ngram_report = (
        calculate_discriminative_ngrams(
            dataframe=dataframe,
            group_column="source",
        )
    )

    label_ngram_report = (
        calculate_discriminative_ngrams(
            dataframe=dataframe,
            group_column="label_standard",
        )
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    pattern_path = (
        OUTPUT_DIRECTORY
        / "source_pattern_report.csv"
    )

    source_ngram_path = (
        OUTPUT_DIRECTORY
        / "source_discriminative_ngrams.csv"
    )

    label_ngram_path = (
        OUTPUT_DIRECTORY
        / "label_discriminative_ngrams.csv"
    )

    pattern_report.to_csv(
        pattern_path,
        index=False,
        encoding="utf-8-sig",
    )

    source_ngram_report.to_csv(
        source_ngram_path,
        index=False,
        encoding="utf-8-sig",
    )

    label_ngram_report.to_csv(
        label_ngram_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("KALIP ANALİZİ")
    print("=" * 60)

    if pattern_report.empty:
        print("Tanımlanan kalıplardan hiçbiri bulunmadı.")
    else:
        print(
            pattern_report[
                [
                    "analysis_dataset",
                    "source",
                    "label_standard",
                    "pattern_name",
                    "match_count",
                    "match_rate_percent",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    print("\n" + "=" * 60)
    print("ANA HABER KAYNAK PARMAK İZLERİ")
    print("=" * 60)

    article_source_ngrams = source_ngram_report[
        source_ngram_report[
            "analysis_dataset"
        ].eq("article_main")
    ]

    if article_source_ngrams.empty:
        print("Ana haber için ayırt edici n-gram bulunamadı.")
    else:
        print(
            article_source_ngrams[
                article_source_ngrams["rank"] <= 10
            ][
                [
                    "group_value",
                    "rank",
                    "ngram",
                    "group_rate_percent",
                    "other_rate_percent",
                    "difference_percent",
                ]
            ].to_string(index=False)
        )

    print("\nOluşturulan raporlar:")
    print(f"- {pattern_path}")
    print(f"- {source_ngram_path}")
    print(f"- {label_ngram_path}")

    print("\nParmak izi analizi tamamlandı.")


if __name__ == "__main__":
    main()