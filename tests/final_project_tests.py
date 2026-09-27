# ============================================================
# FINAL PROJE TESTİ VE GÜVENLİ TEMİZLİK
#
# Dosya önerisi:
# tests/run_final_project_tests.py
#
# PyCharm:
#   Dosyaya sağ tık -> Run 'run_final_project_tests'
#
# Bu script:
#   1. Proje yapısını ve dosyaları kontrol eder.
#   2. Python / JSON / JavaScript sözdizimini doğrular.
#   3. Chrome eklentisi manifest ve sınırlandırmalarını kontrol eder.
#   4. Final ELECTRA modelini yükler.
#   5. Model servisinin kısa, normal ve uzun metin davranışını test eder.
#   6. 4 REAL + 4 FAKE canlı demonstrasyon örneğini test eder.
#   7. FastAPI endpointlerini test eder.
#   8. Kısa / orta / uzun metin performansını ölçer.
#   9. Varsa veri splitlerini ve kayıt sayılarını kontrol eder.
#  10. Raporları reports/ klasörüne yazar.
#  11. Tüm kritik testler geçerse yalnızca güvenli geçici dosyaları temizler
#      ve eski kullanılmayan dosyaları archive/ klasörüne taşır.
# ============================================================

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable


# ============================================================
# AYARLAR
# ============================================================

# Tüm kritik testler başarılı olursa güvenli temizlik yapılır.
# Eski dosyalar silinmez; archive/ altına taşınır.
ENABLE_SAFE_CLEANUP = True

# Performans ölçümünde her metin kaç kez çalıştırılacak?
PERFORMANCE_REPEAT_COUNT = 3

# Arka arkaya kararlılık testi istek sayısı.
STRESS_REQUEST_COUNT = 4

# Büyük veri dosyaları bulunursa split bütünlük kontrolü yapılır.
ENABLE_DATASET_CHECKS = True

# JavaScript sözdizimi için Node.js yoksa test SKIP olur.
CHECK_JAVASCRIPT_WITH_NODE = True


# ============================================================
# PROJE YOLU
# ============================================================

THIS_FILE = Path(__file__).resolve()

# Script tests/ içinde veya proje kökünde çalışabilir.
if THIS_FILE.parent.name == "tests":
    PROJECT_ROOT = THIS_FILE.parents[1]
else:
    PROJECT_ROOT = THIS_FILE.parent

REPORTS_DIR = PROJECT_ROOT / "reports"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

TEXT_REPORT_PATH = (
    REPORTS_DIR
    / "final_system_test_report.txt"
)

PERFORMANCE_CSV_PATH = (
    REPORTS_DIR
    / "final_performance_results.csv"
)

CLEANUP_REPORT_PATH = (
    REPORTS_DIR
    / "final_cleanup_report.txt"
)

DEMO_RESULTS_PATH = (
    REPORTS_DIR
    / "final_demo_test_results.csv"
)


# Proje modüllerinin import edilebilmesi için kökü ekle.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# SONUÇ YAPILARI
# ============================================================

@dataclass
class TestResult:
    category: str
    name: str
    status: str
    duration_seconds: float
    detail: str = ""
    critical: bool = True


@dataclass
class PerformanceResult:
    name: str
    character_count: int
    token_count: int
    chunk_count: int
    prediction: str
    confidence: float
    repeat_count: int
    first_seconds: float
    average_seconds: float
    minimum_seconds: float
    maximum_seconds: float


@dataclass
class Context:
    results: list[TestResult] = field(default_factory=list)
    performance_results: list[PerformanceResult] = field(
        default_factory=list
    )
    cleanup_lines: list[str] = field(default_factory=list)
    demo_results: list[dict[str, Any]] = field(default_factory=list)
    model_service: Any | None = None
    torch_module: Any | None = None
    test_articles: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )


CONTEXT = Context()


# ============================================================
# GENEL YARDIMCILAR
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def format_seconds(value: float) -> str:
    if value < 1:
        return f"{value * 1000:.1f} ms"
    return f"{value:.3f} sn"


def run_test(
    category: str,
    name: str,
    function: Callable[[], str | None],
    *,
    critical: bool = True,
) -> None:
    start_time = time.perf_counter()

    try:
        detail = function() or "Başarılı."
        status = "PASS"

    except SkipTest as error:
        detail = str(error)
        status = "SKIP"

    except Exception as error:
        detail = (
            f"{error.__class__.__name__}: {error}\n"
            f"{traceback.format_exc()}"
        )
        status = "FAIL"

    duration = time.perf_counter() - start_time

    result = TestResult(
        category=category,
        name=name,
        status=status,
        duration_seconds=duration,
        detail=detail,
        critical=critical,
    )

    CONTEXT.results.append(result)

    symbol = {
        "PASS": "✓",
        "FAIL": "✗",
        "SKIP": "-",
    }[status]

    print(
        f"{symbol} [{status:<4}] "
        f"{category} / {name} "
        f"({format_seconds(duration)})"
    )

    if status != "PASS":
        first_line = detail.splitlines()[0] if detail else ""
        print(f"        {first_line}")


class SkipTest(RuntimeError):
    """Eksik isteğe bağlı bileşen nedeniyle atlanan test."""


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_close(
    first: float,
    second: float,
    tolerance: float = 1e-4,
    message: str = "Değerler birbirine yakın değil.",
) -> None:
    if not math.isclose(
        first,
        second,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise AssertionError(
            f"{message} Bulunan: {first} ve {second}"
        )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize_text_for_hash(value: str) -> str:
    return " ".join(value.split()).strip().lower()


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        normalize_text_for_hash(value).encode("utf-8")
    ).hexdigest()


def get_directory_size(path: Path) -> int:
    total = 0

    if not path.exists():
        return 0

    if path.is_file():
        return path.stat().st_size

    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue

    return total


def format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{value} B"


# ============================================================
# 1. DOSYA VE SÖZDİZİMİ TESTLERİ
# ============================================================

def test_required_project_files() -> str:
    required_paths = [
        PROJECT_ROOT / "backend" / "app" / "main.py",
        PROJECT_ROOT / "backend" / "app" / "api" / "routes.py",
        PROJECT_ROOT
        / "backend"
        / "app"
        / "services"
        / "model_service.py",
        PROJECT_ROOT
        / "backend"
        / "app"
        / "schemas"
        / "prediction.py",
        PROJECT_ROOT / "extension" / "manifest.json",
        PROJECT_ROOT / "extension" / "background.js",
        PROJECT_ROOT / "extension" / "content.js",
        PROJECT_ROOT / "extension" / "site_rules.js",
        PROJECT_ROOT
        / "models"
        / "electra_turkish_article_60k_final"
        / "config.json",
        PROJECT_ROOT
        / "models"
        / "electra_turkish_article_60k_final"
        / "model.safetensors",
        PROJECT_ROOT
        / "models"
        / "electra_turkish_article_60k_final"
        / "tokenizer.json",
        PROJECT_ROOT / "tests" / "fake_news_test_site.py",
        PROJECT_ROOT / "requirements.txt",
    ]

    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in required_paths
        if not path.exists()
    ]

    assert_true(
        not missing,
        "Eksik zorunlu dosyalar:\n" + "\n".join(missing),
    )

    return f"{len(required_paths)} zorunlu dosya bulundu."


def iter_source_files(suffix: str) -> Iterable[Path]:
    excluded_parts = {
        ".git",
        ".idea",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        "archive",
    }

    for path in PROJECT_ROOT.rglob(f"*{suffix}"):
        if any(part in excluded_parts for part in path.parts):
            continue
        if path.is_file():
            yield path


def test_python_syntax() -> str:
    files = list(iter_source_files(".py"))
    errors: list[str] = []

    for path in files:
        try:
            source = path.read_text(encoding="utf-8-sig")
            compile(source, str(path), "exec")
        except Exception as error:
            errors.append(
                f"{path.relative_to(PROJECT_ROOT)}: {error}"
            )

    assert_true(
        not errors,
        "Python sözdizimi hataları:\n" + "\n".join(errors),
    )

    return f"{len(files)} Python dosyası derleme kontrolünden geçti."


def test_json_files() -> str:
    files = list(iter_source_files(".json"))
    errors: list[str] = []

    for path in files:
        try:
            with path.open("r", encoding="utf-8-sig") as file:
                json.load(file)
        except Exception as error:
            errors.append(
                f"{path.relative_to(PROJECT_ROOT)}: {error}"
            )

    assert_true(
        not errors,
        "JSON hataları:\n" + "\n".join(errors),
    )

    return f"{len(files)} JSON dosyası doğrulandı."


def test_javascript_syntax() -> str:
    files = list(iter_source_files(".js"))

    if not CHECK_JAVASCRIPT_WITH_NODE:
        raise SkipTest("Node.js sözdizimi kontrolü kapalı.")

    node_path = shutil.which("node")

    if node_path is None:
        raise SkipTest(
            "Node.js bulunamadı; JavaScript sözdizimi testi atlandı."
        )

    errors: list[str] = []

    for path in files:
        process = subprocess.run(
            [node_path, "--check", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

        if process.returncode != 0:
            errors.append(
                f"{path.relative_to(PROJECT_ROOT)}:\n"
                f"{process.stderr.strip()}"
            )

    assert_true(
        not errors,
        "JavaScript sözdizimi hataları:\n" + "\n".join(errors),
    )

    return f"{len(files)} JavaScript dosyası doğrulandı."


def test_model_metrics_file() -> str:
    metrics_path = (
        PROJECT_ROOT
        / "models"
        / "electra_turkish_article_60k_final"
        / "metrics.json"
    )

    if not metrics_path.exists():
        raise SkipTest("metrics.json bulunamadı.")

    metrics = json.loads(read_text(metrics_path))
    test_metrics = metrics.get("test_metrics", {})

    assert_true(
        int(metrics.get("train_records", 0)) == 48_000,
        "Train kayıt sayısı 48.000 değil.",
    )
    assert_true(
        int(metrics.get("validation_records", 0)) == 6_000,
        "Validation kayıt sayısı 6.000 değil.",
    )
    assert_true(
        int(metrics.get("test_records", 0)) == 6_000,
        "Test kayıt sayısı 6.000 değil.",
    )

    macro_f1 = float(test_metrics.get("macro_f1", 0.0))
    fake_recall = float(test_metrics.get("fake_recall", 0.0))
    roc_auc = float(test_metrics.get("roc_auc", 0.0))

    assert_true(macro_f1 >= 0.95, "Test Macro F1 %95 altında.")
    assert_true(fake_recall >= 0.94, "Fake recall %94 altında.")
    assert_true(roc_auc >= 0.99, "ROC-AUC %99 altında.")

    mapping = metrics.get("label_mapping", {})
    assert_true(
        str(mapping.get("0")).lower() == "real",
        "metrics.json içinde 0=real değil.",
    )
    assert_true(
        str(mapping.get("1")).lower() == "fake",
        "metrics.json içinde 1=fake değil.",
    )

    return (
        f"Macro F1={macro_f1:.6f}, "
        f"fake recall={fake_recall:.6f}, "
        f"ROC-AUC={roc_auc:.6f}."
    )


# ============================================================
# 2. CHROME EKLENTİSİ TESTLERİ
# ============================================================

def test_extension_manifest() -> str:
    manifest_path = PROJECT_ROOT / "extension" / "manifest.json"

    with manifest_path.open("r", encoding="utf-8-sig") as file:
        manifest = json.load(file)

    assert_true(
        manifest.get("manifest_version") == 3,
        "Manifest V3 kullanılmıyor.",
    )

    background_file = (
        manifest.get("background", {})
        .get("service_worker")
    )

    assert_true(
        background_file == "background.js",
        "Service worker background.js değil.",
    )

    scripts = manifest.get("content_scripts", [])
    assert_true(bool(scripts), "Content script tanımı yok.")

    loaded_js = scripts[0].get("js", [])
    assert_true(
        loaded_js == ["site_rules.js", "content.js"],
        "Content script sırası site_rules.js -> content.js değil.",
    )

    action = manifest.get("action", {})
    assert_true(
        "default_popup" not in action,
        "Eski popup hâlâ manifestte aktif.",
    )

    for file_name in [background_file, *loaded_js]:
        assert_true(
            (PROJECT_ROOT / "extension" / str(file_name)).exists(),
            f"Manifestte tanımlı dosya yok: {file_name}",
        )

    return "Manifest V3, service worker ve content script sırası doğru."


def test_extension_backend_connection() -> str:
    background = read_text(
        PROJECT_ROOT / "extension" / "background.js"
    )

    required_markers = [
        'http://127.0.0.1:8000',
        "/health",
        "/predict",
        "ANALYZE_ARTICLE",
    ]

    missing = [
        marker
        for marker in required_markers
        if marker not in background
    ]

    assert_true(
        not missing,
        "background.js içinde eksik işaretler: "
        + ", ".join(missing),
    )

    return "Backend URL, /health, /predict ve ANALYZE_ARTICLE bulundu."


def test_extension_page_restrictions() -> str:
    content = read_text(
        PROJECT_ROOT / "extension" / "content.js"
    )

    required_markers = [
        "wikipedia",
        "EXCLUDED_PATH_PATTERNS",
        "getPageExclusionReason",
        "/arama",
        "Kategori, arama, etiket, yazar",
        "Site ana sayfası haber detay",
    ]

    missing = [
        marker
        for marker in required_markers
        if marker not in content
    ]

    assert_true(
        not missing,
        "Sayfa sınırlandırmalarında eksik işaretler: "
        + ", ".join(missing),
    )

    return "Wikipedia, ana sayfa, arama ve liste sayfası kontrolleri mevcut."


def test_site_specific_rules() -> str:
    rules = read_text(
        PROJECT_ROOT / "extension" / "site_rules.js"
    ).lower()

    expected_sites = [
        "trthaber.com",
        "aa.com.tr",
        "sozcu.com.tr",
        "cumhuriyet.com.tr",
        "hurriyet.com.tr",
        "milliyet.com.tr",
    ]

    missing = [
        site
        for site in expected_sites
        if site not in rules
    ]

    assert_true(
        not missing,
        "Site kuralları eksik: " + ", ".join(missing),
    )

    return f"{len(expected_sites)} haber sitesi için kural bulundu."


# ============================================================
# 3. MODEL YÜKLEME VE SERVİS TESTLERİ
# ============================================================

def load_demo_articles() -> dict[str, dict[str, Any]]:
    module_path = PROJECT_ROOT / "tests" / "fake_news_test_site.py"

    spec = importlib.util.spec_from_file_location(
        "final_fake_news_test_site",
        module_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("fake_news_test_site.py yüklenemedi.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    articles = getattr(module, "TEST_ARTICLES", None)

    if not isinstance(articles, dict):
        raise RuntimeError("TEST_ARTICLES sözlüğü bulunamadı.")

    return articles


def article_to_text(article: dict[str, Any]) -> str:
    parts = [
        str(article.get("title", "")),
        str(article.get("description", "")),
        *[
            str(paragraph)
            for paragraph in article.get("paragraphs", [])
        ],
    ]

    normalized_parts: list[str] = []
    seen: set[str] = set()

    for part in parts:
        normalized = " ".join(part.split()).strip()
        key = normalized.lower()

        if not normalized or key in seen:
            continue

        seen.add(key)
        normalized_parts.append(normalized)

    return "\n\n".join(normalized_parts)


def test_model_import_and_load() -> str:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch kurulu değil. requirements içine torch eklenmeli."
        ) from error

    try:
        from backend.app.services.model_service import model_service
    except ImportError as error:
        raise RuntimeError(
            "ModelService import edilemedi. transformers ve torch "
            "bağımlılıklarını kontrol et."
        ) from error

    CONTEXT.torch_module = torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model_service.load_model()
    load_seconds = time.perf_counter() - start

    CONTEXT.model_service = model_service
    CONTEXT.test_articles = load_demo_articles()

    assert_true(model_service.is_loaded, "Model yüklenmiş görünmüyor.")

    status = model_service.get_status()
    assert_true(status.get("loaded") is True, "Health model loaded=false.")
    assert_true(status.get("maximum_token_length") == 512, "MAX_LENGTH 512 değil.")
    assert_true(status.get("minimum_token_length") == 80, "MIN_TOKEN_COUNT 80 değil.")
    assert_true(status.get("chunk_overlap_tokens") == 50, "Overlap 50 değil.")

    labels = status.get("labels", {})
    normalized_labels = {
        str(key): str(value).lower()
        for key, value in labels.items()
    }

    assert_true(
        normalized_labels.get("0") == "real",
        "Model status içinde 0=real değil.",
    )
    assert_true(
        normalized_labels.get("1") == "fake",
        "Model status içinde 1=fake değil.",
    )

    gpu_text = "CPU"
    if torch.cuda.is_available():
        gpu_text = torch.cuda.get_device_name(0)

    return (
        f"Model {format_seconds(load_seconds)} sürede yüklendi. "
        f"Cihaz: {gpu_text}."
    )


def require_model_service() -> Any:
    if CONTEXT.model_service is None:
        raise RuntimeError("ModelService önce yüklenmelidir.")
    return CONTEXT.model_service


def validate_prediction_payload(result: dict[str, Any]) -> None:
    assert_true(
        result.get("label") in {"real", "fake"},
        "Tahmin label değeri geçersiz.",
    )

    label_id = result.get("label_id")
    assert_true(label_id in {0, 1}, "Tahmin label_id geçersiz.")

    probabilities = result.get("probabilities", {})
    real_probability = float(probabilities.get("real", -1))
    fake_probability = float(probabilities.get("fake", -1))

    assert_true(
        0.0 <= real_probability <= 1.0,
        "Real olasılığı 0-1 aralığında değil.",
    )
    assert_true(
        0.0 <= fake_probability <= 1.0,
        "Fake olasılığı 0-1 aralığında değil.",
    )

    assert_close(
        real_probability + fake_probability,
        1.0,
        tolerance=2e-5,
        message="Real + fake olasılığı 1 değil.",
    )

    confidence = float(result.get("confidence", -1))
    assert_true(0.0 <= confidence <= 1.0, "Güven değeri geçersiz.")

    chunks = result.get("chunks")
    assert_true(isinstance(chunks, list) and chunks, "Chunk listesi boş.")
    assert_true(
        len(chunks) == int(result.get("chunk_count", 0)),
        "chunk_count ile chunks uzunluğu uyuşmuyor.",
    )

    for chunk in chunks:
        assert_true(
            1 <= int(chunk.get("token_count", 0)) <= 512,
            "Bir chunk 512 token sınırını aşıyor.",
        )

        chunk_probabilities = chunk.get("probabilities", {})
        chunk_sum = (
            float(chunk_probabilities.get("real", -1))
            + float(chunk_probabilities.get("fake", -1))
        )

        assert_close(
            chunk_sum,
            1.0,
            tolerance=2e-5,
            message="Chunk olasılıklarının toplamı 1 değil.",
        )


def test_clean_text_rules() -> str:
    service = require_model_service()

    cleaned = service.clean_text(
        "  Haber\u200b   metni\n\n düzenli biçimde temizlendi.  "
    )

    assert_true(
        cleaned == "Haber metni düzenli biçimde temizlendi.",
        f"Metin temizleme sonucu beklenmedik: {cleaned}",
    )

    invalid_inputs: list[tuple[Any, type[Exception]]] = [
        (None, TypeError),
        ("", ValueError),
        ("çok kısa", ValueError),
    ]

    for value, expected_error in invalid_inputs:
        try:
            service.clean_text(value)
        except expected_error:
            continue
        raise AssertionError(
            f"Geçersiz giriş reddedilmedi: {value!r}"
        )

    try:
        service.clean_text("x" * 100_001)
    except ValueError:
        pass
    else:
        raise AssertionError("100.000 karakter sınırı çalışmıyor.")

    return "Görünmeyen karakter, boşluk, tip ve uzunluk kontrolleri doğru."


def test_short_text_rejection() -> str:
    service = require_model_service()

    short_text = (
        "Bu kısa haber metni yalnızca minimum token "
        "kontrolünü doğrulamak amacıyla hazırlanmıştır."
    )

    try:
        service.predict(short_text)
    except ValueError as error:
        message = str(error)
        assert_true(
            "80" in message and "çok kısa" in message,
            "Kısa metin doğru hata mesajını üretmedi.",
        )
    else:
        raise AssertionError("80 token altındaki metin kabul edildi.")

    return "80 token altındaki içerik doğru biçimde reddedildi."


def test_single_chunk_prediction() -> str:
    service = require_model_service()

    articles = CONTEXT.test_articles
    assert_true(bool(articles), "Demo haberleri yüklenmedi.")

    # En kısa ama 80 token üstündeki demo metnini seç.
    candidates: list[tuple[int, str]] = []

    for article in articles.values():
        text = article_to_text(article)
        token_count = len(
            service._tokenize_without_special_tokens(text)
        ) + 2
        if 80 <= token_count <= 512:
            candidates.append((token_count, text))

    assert_true(bool(candidates), "Tek parça demo metni bulunamadı.")

    token_count, text = min(candidates, key=lambda item: item[0])
    result = service.predict(text)

    validate_prediction_payload(result)

    assert_true(result["is_chunked"] is False, "Kısa metin parçalandı.")
    assert_true(result["chunk_count"] == 1, "Kısa metin 1 chunk değil.")
    assert_true(
        result["chunk_overlap_tokens"] == 0,
        "Tek chunk için overlap sıfır değil.",
    )
    assert_true(
        result["total_token_count"] == token_count,
        "Token sayısı tutarsız.",
    )
    assert_true(result["all_text_analyzed"] is True, "Tüm metin analiz edilmedi.")

    return (
        f"{token_count} tokenlık metin tek parça işlendi; "
        f"sonuç={result['label']}."
    )


def build_long_text(minimum_content_tokens: int = 1_500) -> str:
    service = require_model_service()

    articles = CONTEXT.test_articles
    assert_true(bool(articles), "Demo haberleri yüklenmedi.")

    base_article = next(iter(articles.values()))
    base_text = article_to_text(base_article)

    pieces: list[str] = []

    while True:
        pieces.append(base_text)
        long_text = "\n\n".join(pieces)

        if len(long_text) > 95_000:
            raise RuntimeError("Uzun test metni 100.000 karaktere yaklaştı.")

        token_count = len(
            service._tokenize_without_special_tokens(long_text)
        )

        if token_count >= minimum_content_tokens:
            return long_text


def test_long_text_chunking() -> str:
    service = require_model_service()
    long_text = build_long_text(1_500)
    result = service.predict(long_text)

    validate_prediction_payload(result)

    assert_true(result["is_chunked"] is True, "Uzun metin parçalanmadı.")
    assert_true(result["chunk_count"] >= 3, "Uzun metin en az 3 chunk olmadı.")
    assert_true(
        result["chunk_overlap_tokens"] == 50,
        "Uzun metin overlap değeri 50 değil.",
    )
    assert_true(result["all_text_analyzed"] is True, "Tüm uzun metin analiz edilmedi.")
    assert_true(
        result["processed_token_count"] == result["total_token_count"],
        "İşlenen ve toplam token sayısı eşit değil.",
    )

    effective_sum = sum(
        int(chunk["effective_token_count"])
        for chunk in result["chunks"]
    )

    # İçerik tokenı = toplam token - CLS - SEP
    assert_true(
        effective_sum == result["total_token_count"] - 2,
        "Chunk etkili token toplamı orijinal metni kapsamıyor.",
    )

    return (
        f"{result['total_token_count']} token, "
        f"{result['chunk_count']} chunk ve 50 token overlap ile işlendi."
    )


def test_demo_articles_8_of_8() -> str:
    service = require_model_service()
    articles = CONTEXT.test_articles

    assert_true(
        len(articles) == 8,
        f"Kontrollü demonstrasyon seti 8 değil: {len(articles)}",
    )

    correct_count = 0
    real_count = 0
    fake_count = 0
    demo_rows: list[dict[str, Any]] = []

    for slug, article in articles.items():
        expected_label = str(article.get("expected_label", "")).lower()
        assert_true(
            expected_label in {"real", "fake"},
            f"Geçersiz expected_label: {slug}",
        )

        text = article_to_text(article)
        result = service.predict(text)
        validate_prediction_payload(result)

        predicted_label = result["label"]
        is_correct = predicted_label == expected_label

        if expected_label == "real":
            real_count += 1
        else:
            fake_count += 1

        if is_correct:
            correct_count += 1

        demo_rows.append(
            {
                "slug": slug,
                "title": article.get("title", ""),
                "expected_label": expected_label,
                "predicted_label": predicted_label,
                "correct": is_correct,
                "confidence": result["confidence"],
                "real_probability": result["probabilities"]["real"],
                "fake_probability": result["probabilities"]["fake"],
                "token_count": result["total_token_count"],
                "chunk_count": result["chunk_count"],
            }
        )

        print(
            f"        {slug:<34} "
            f"beklenen={expected_label:<4} "
            f"çıktı={predicted_label:<4} "
            f"güven=%{result['confidence'] * 100:.2f}"
        )

    CONTEXT.demo_results = demo_rows

    assert_true(real_count == 4, f"Real kontrol sayısı 4 değil: {real_count}")
    assert_true(fake_count == 4, f"Fake kontrol sayısı 4 değil: {fake_count}")
    assert_true(
        correct_count == 8,
        f"Kontrollü demo sonucu {correct_count}/8; beklenen 8/8.",
    )

    return "4 REAL + 4 FAKE kontrollü testin tamamı doğru: 8/8."


def test_stability_requests() -> str:
    service = require_model_service()
    articles = list(CONTEXT.test_articles.values())
    assert_true(bool(articles), "Demo haberleri bulunamadı.")

    texts = [
        article_to_text(articles[index % len(articles)])
        for index in range(STRESS_REQUEST_COUNT)
    ]

    errors: list[str] = []
    labels: list[str] = []

    def predict_text(text: str) -> str:
        result = service.predict(text)
        validate_prediction_payload(result)
        return str(result["label"])

    with ThreadPoolExecutor(
        max_workers=min(4, STRESS_REQUEST_COUNT)
    ) as executor:
        futures = [executor.submit(predict_text, text) for text in texts]

        for future in as_completed(futures):
            try:
                labels.append(future.result())
            except Exception as error:
                errors.append(str(error))

    assert_true(
        not errors,
        "Arka arkaya tahmin hataları:\n" + "\n".join(errors),
    )
    assert_true(
        len(labels) == STRESS_REQUEST_COUNT,
        "Bütün kararlılık istekleri tamamlanmadı.",
    )

    return f"{STRESS_REQUEST_COUNT} eş zamanlı istek hatasız tamamlandı."


# ============================================================
# 4. FASTAPI ENTEGRASYON TESTLERİ
# ============================================================

def create_test_client() -> Any:
    try:
        from fastapi.testclient import TestClient
        from backend.app.main import app
    except ImportError as error:
        raise RuntimeError(
            "FastAPI TestClient kullanılamıyor. "
            "fastapi ve httpx paketlerini kontrol et."
        ) from error

    return TestClient(app)


def test_api_root_and_health() -> str:
    with create_test_client() as client:
        root_response = client.get("/")
        assert_true(root_response.status_code == 200, "GET / 200 dönmedi.")

        root_data = root_response.json()
        assert_true(
            root_data.get("status") == "running",
            "GET / status=running değil.",
        )

        health_response = client.get("/health")
        assert_true(
            health_response.status_code == 200,
            "GET /health 200 dönmedi.",
        )

        health_data = health_response.json()
        assert_true(
            health_data.get("status") == "healthy",
            "/health healthy değil.",
        )
        assert_true(
            health_data.get("model", {}).get("loaded") is True,
            "/health model loaded=true değil.",
        )

    return "GET / ve GET /health başarılı."


def test_api_predict_valid() -> str:
    article = next(iter(CONTEXT.test_articles.values()))
    text = article_to_text(article)

    with create_test_client() as client:
        response = client.post("/predict", json={"text": text})
        assert_true(
            response.status_code == 200,
            f"POST /predict 200 dönmedi: {response.text}",
        )

        data = response.json()
        validate_prediction_payload(data)

    return (
        f"POST /predict başarılı; label={data['label']}, "
        f"token={data['total_token_count']}."
    )


def test_api_validation_errors() -> str:
    short_text = (
        "Bu metin minimum token sınırının altında kalacak "
        "şekilde hazırlanmış kısa bir test haberidir."
    )

    with create_test_client() as client:
        short_response = client.post(
            "/predict",
            json={"text": short_text},
        )
        assert_true(
            short_response.status_code == 400,
            f"Kısa metin 400 dönmedi: {short_response.status_code}",
        )

        missing_response = client.post("/predict", json={})
        assert_true(
            missing_response.status_code == 422,
            "Eksik text alanı 422 dönmedi.",
        )

        tiny_response = client.post("/predict", json={"text": "kısa"})
        assert_true(
            tiny_response.status_code == 422,
            "20 karakter altı Pydantic doğrulaması 422 dönmedi.",
        )

        too_long_response = client.post(
            "/predict",
            json={"text": "x" * 100_001},
        )
        assert_true(
            too_long_response.status_code == 422,
            "100.000 karakter üstü istek 422 dönmedi.",
        )

    return "400 ve 422 doğrulama senaryoları doğru çalıştı."


def test_api_cors() -> str:
    with create_test_client() as client:
        response = client.options(
            "/predict",
            headers={
                "Origin": "chrome-extension://test-extension",
                "Access-Control-Request-Method": "POST",
            },
        )

        assert_true(
            response.status_code == 200,
            f"CORS preflight 200 dönmedi: {response.status_code}",
        )

        allow_origin = response.headers.get(
            "access-control-allow-origin"
        )

        assert_true(
            allow_origin in {"*", "chrome-extension://test-extension"},
            f"CORS origin header beklenmedik: {allow_origin}",
        )

    return "Chrome eklentisi için CORS preflight başarılı."


# ============================================================
# 5. PERFORMANS TESTLERİ
# ============================================================

def measure_prediction(
    name: str,
    text: str,
    repeat_count: int,
) -> PerformanceResult:
    service = require_model_service()

    durations: list[float] = []
    last_result: dict[str, Any] | None = None

    for _ in range(repeat_count):
        start = time.perf_counter()
        result = service.predict(text)

        # CUDA işleminin tamamen bitmesini bekle.
        torch_module = CONTEXT.torch_module
        if (
            torch_module is not None
            and torch_module.cuda.is_available()
        ):
            torch_module.cuda.synchronize()

        durations.append(time.perf_counter() - start)
        validate_prediction_payload(result)
        last_result = result

    assert_true(last_result is not None, "Performans tahmini oluşmadı.")

    return PerformanceResult(
        name=name,
        character_count=len(text),
        token_count=int(last_result["total_token_count"]),
        chunk_count=int(last_result["chunk_count"]),
        prediction=str(last_result["label"]),
        confidence=float(last_result["confidence"]),
        repeat_count=repeat_count,
        first_seconds=durations[0],
        average_seconds=statistics.mean(durations),
        minimum_seconds=min(durations),
        maximum_seconds=max(durations),
    )


def test_inference_performance() -> str:
    service = require_model_service()
    articles = list(CONTEXT.test_articles.values())
    assert_true(len(articles) >= 2, "Performans için demo haberleri eksik.")

    short_text = article_to_text(articles[0])
    medium_text = "\n\n".join(
        [article_to_text(articles[1])] * 2
    )
    long_text = build_long_text(1_500)

    # İlk ölçüm öncesi tek ısınma tahmini.
    service.predict(short_text)

    performance_results = [
        measure_prediction(
            "kısa_haber",
            short_text,
            PERFORMANCE_REPEAT_COUNT,
        ),
        measure_prediction(
            "orta_haber",
            medium_text,
            PERFORMANCE_REPEAT_COUNT,
        ),
        measure_prediction(
            "uzun_haber",
            long_text,
            PERFORMANCE_REPEAT_COUNT,
        ),
    ]

    CONTEXT.performance_results = performance_results

    for result in performance_results:
        print(
            f"        {result.name:<12} "
            f"token={result.token_count:<5} "
            f"chunk={result.chunk_count:<3} "
            f"ortalama={format_seconds(result.average_seconds)}"
        )

    assert_true(
        all(math.isfinite(item.average_seconds) for item in performance_results),
        "Performans süresi geçersiz.",
    )

    return "Kısa, orta ve uzun haber performansı ölçüldü."


def test_gpu_memory() -> str:
    torch_module = CONTEXT.torch_module

    if torch_module is None:
        raise SkipTest("PyTorch modülü yüklenmedi.")

    if not torch_module.cuda.is_available():
        raise SkipTest("CUDA aktif değil; GPU bellek testi atlandı.")

    allocated_mb = (
        torch_module.cuda.memory_allocated(0)
        / 1024**2
    )
    reserved_mb = (
        torch_module.cuda.memory_reserved(0)
        / 1024**2
    )
    peak_mb = (
        torch_module.cuda.max_memory_allocated(0)
        / 1024**2
    )

    assert_true(peak_mb > 0, "GPU tepe bellek değeri sıfır.")

    return (
        f"GPU allocated={allocated_mb:.1f} MB, "
        f"reserved={reserved_mb:.1f} MB, "
        f"peak={peak_mb:.1f} MB."
    )


# ============================================================
# 6. VERİ SETİ BÜTÜNLÜK TESTLERİ (VARSA)
# ============================================================

def find_file_by_name(file_name: str) -> Path | None:
    search_roots = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "datasets",
        PROJECT_ROOT,
    ]

    seen: set[Path] = set()

    for search_root in search_roots:
        if not search_root.exists():
            continue

        for path in search_root.rglob(file_name):
            resolved = path.resolve()

            if resolved in seen:
                continue

            seen.add(resolved)

            if path.is_file():
                return path

    return None


def detect_column(field_names: list[str], candidates: list[str]) -> str | None:
    normalized = {
        str(name).strip().lower(): name
        for name in field_names
    }

    for candidate in candidates:
        found = normalized.get(candidate.lower())
        if found is not None:
            return found

    return None


def inspect_split_csv(path: Path) -> dict[str, Any]:
    row_count = 0
    pair_ids: set[str] = set()
    text_hashes: set[str] = set()
    label_counts: dict[str, int] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        field_names = list(reader.fieldnames or [])

        pair_column = detect_column(
            field_names,
            ["pair_id", "test_pair_id", "split_group_id"],
        )
        text_column = detect_column(
            field_names,
            [
                "text",
                "article_text",
                "test_text",
                "text_basic",
                "content",
            ],
        )
        label_column = detect_column(
            field_names,
            [
                "label",
                "label_id",
                "test_label",
                "test_label_id",
                "label_standard",
            ],
        )

        for row in reader:
            row_count += 1

            if pair_column:
                value = str(row.get(pair_column, "")).strip()
                if value:
                    pair_ids.add(value)

            if text_column:
                value = str(row.get(text_column, ""))
                if value.strip():
                    text_hashes.add(sha256_text(value))

            if label_column:
                value = str(row.get(label_column, "")).strip().lower()
                label_counts[value] = label_counts.get(value, 0) + 1

    return {
        "path": path,
        "row_count": row_count,
        "pair_ids": pair_ids,
        "text_hashes": text_hashes,
        "label_counts": label_counts,
    }


def test_dataset_splits() -> str:
    if not ENABLE_DATASET_CHECKS:
        raise SkipTest("Veri seti kontrolleri kapalı.")

    expected_files = {
        "train": ("article_train_48k.csv", 48_000),
        "validation": ("article_validation_6k.csv", 6_000),
        "test": ("article_test_6k.csv", 6_000),
    }

    found_paths: dict[str, Path] = {}

    for split_name, (file_name, _) in expected_files.items():
        path = find_file_by_name(file_name)
        if path is not None:
            found_paths[split_name] = path

    if not found_paths:
        raise SkipTest(
            "data/datasets klasöründe article split CSV dosyaları bulunamadı."
        )

    missing_splits = set(expected_files) - set(found_paths)
    assert_true(
        not missing_splits,
        "Eksik split dosyaları: " + ", ".join(sorted(missing_splits)),
    )

    split_data = {
        name: inspect_split_csv(path)
        for name, path in found_paths.items()
    }

    for split_name, (_, expected_count) in expected_files.items():
        actual_count = split_data[split_name]["row_count"]
        assert_true(
            actual_count == expected_count,
            f"{split_name} kayıt sayısı {actual_count}; "
            f"beklenen {expected_count}.",
        )

    split_pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    overlap_details: list[str] = []

    for first, second in split_pairs:
        first_pairs = split_data[first]["pair_ids"]
        second_pairs = split_data[second]["pair_ids"]

        if first_pairs and second_pairs:
            pair_overlap = first_pairs & second_pairs
            assert_true(
                not pair_overlap,
                f"{first}-{second} pair_id overlap: {len(pair_overlap)}",
            )
            overlap_details.append(
                f"{first}-{second} pair=0"
            )

        first_hashes = split_data[first]["text_hashes"]
        second_hashes = split_data[second]["text_hashes"]

        if first_hashes and second_hashes:
            text_overlap = first_hashes & second_hashes
            assert_true(
                not text_overlap,
                f"{first}-{second} exact text overlap: {len(text_overlap)}",
            )
            overlap_details.append(
                f"{first}-{second} text=0"
            )

    return (
        "Train=48.000, validation=6.000, test=6.000. "
        + (", ".join(overlap_details) if overlap_details else "")
    ).strip()


def test_existing_6000_predictions() -> str:
    candidate_paths = [
        REPORTS_DIR / "local_electra_test_predictions.csv",
        PROJECT_ROOT
        / "models"
        / "electra_turkish_article_60k_final"
        / "test_predictions.csv",
    ]

    path = next((item for item in candidate_paths if item.exists()), None)

    if path is None:
        raise SkipTest("6.000 test tahmini CSV dosyası bulunamadı.")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        row_count = sum(1 for _ in reader)

    assert_true(
        row_count == 6_000,
        f"Tahmin CSV kayıt sayısı {row_count}; beklenen 6.000.",
    )

    return f"{path.name} içinde 6.000 tahmin kaydı doğrulandı."


# ============================================================
# 7. RAPORLAMA
# ============================================================

def write_demo_results() -> None:
    if not CONTEXT.demo_results:
        return

    field_names = list(CONTEXT.demo_results[0].keys())

    with DEMO_RESULTS_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(CONTEXT.demo_results)


def write_performance_results() -> None:
    if not CONTEXT.performance_results:
        return

    field_names = [
        "name",
        "character_count",
        "token_count",
        "chunk_count",
        "prediction",
        "confidence",
        "repeat_count",
        "first_seconds",
        "average_seconds",
        "minimum_seconds",
        "maximum_seconds",
    ]

    with PERFORMANCE_CSV_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()

        for item in CONTEXT.performance_results:
            writer.writerow(
                {
                    "name": item.name,
                    "character_count": item.character_count,
                    "token_count": item.token_count,
                    "chunk_count": item.chunk_count,
                    "prediction": item.prediction,
                    "confidence": item.confidence,
                    "repeat_count": item.repeat_count,
                    "first_seconds": round(item.first_seconds, 6),
                    "average_seconds": round(item.average_seconds, 6),
                    "minimum_seconds": round(item.minimum_seconds, 6),
                    "maximum_seconds": round(item.maximum_seconds, 6),
                }
            )


def critical_failures() -> list[TestResult]:
    return [
        result
        for result in CONTEXT.results
        if result.critical and result.status == "FAIL"
    ]


def write_text_report() -> None:
    pass_count = sum(result.status == "PASS" for result in CONTEXT.results)
    fail_count = sum(result.status == "FAIL" for result in CONTEXT.results)
    skip_count = sum(result.status == "SKIP" for result in CONTEXT.results)

    lines: list[str] = [
        "=" * 78,
        "NEWS RELIABILITY PROJECT — FINAL SİSTEM TEST RAPORU",
        "=" * 78,
        "",
        f"Tarih: {datetime.now().isoformat(timespec='seconds')}",
        f"Proje kökü: {PROJECT_ROOT}",
        f"Python: {sys.version.split()[0]}",
        f"Platform: {sys.platform}",
        "",
        "GENEL SONUÇ",
        "-" * 78,
        f"PASS: {pass_count}",
        f"FAIL: {fail_count}",
        f"SKIP: {skip_count}",
        f"Kritik hata: {len(critical_failures())}",
        "",
    ]

    categories: list[str] = []
    for result in CONTEXT.results:
        if result.category not in categories:
            categories.append(result.category)

    for category in categories:
        lines.extend([category.upper(), "-" * 78])

        for result in CONTEXT.results:
            if result.category != category:
                continue

            lines.append(
                f"[{result.status}] {result.name} "
                f"({result.duration_seconds:.4f} sn)"
            )

            if result.detail:
                for detail_line in result.detail.rstrip().splitlines():
                    lines.append(f"    {detail_line}")

        lines.append("")

    if CONTEXT.performance_results:
        lines.extend(["PERFORMANS ÖZETİ", "-" * 78])

        for item in CONTEXT.performance_results:
            lines.append(
                f"{item.name}: token={item.token_count}, "
                f"chunk={item.chunk_count}, "
                f"ortalama={item.average_seconds:.4f} sn, "
                f"min={item.minimum_seconds:.4f} sn, "
                f"max={item.maximum_seconds:.4f} sn"
            )

        lines.append("")

    if CONTEXT.demo_results:
        correct = sum(bool(row["correct"]) for row in CONTEXT.demo_results)
        lines.extend(
            [
                "KONTROLLÜ CANLI DEMO",
                "-" * 78,
                f"Doğru: {correct}/{len(CONTEXT.demo_results)}",
                "",
            ]
        )

    final_status = (
        "BAŞARILI — KRİTİK HATA YOK"
        if not critical_failures()
        else "BAŞARISIZ — KRİTİK HATALAR VAR"
    )

    lines.extend(["FİNAL DURUM", "-" * 78, final_status, ""])

    TEXT_REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# 8. GÜVENLİ TEMİZLİK
# ============================================================

def find_cache_items() -> list[Path]:
    items: list[Path] = []

    for path in PROJECT_ROOT.rglob("__pycache__"):
        if "archive" not in path.parts and path.is_dir():
            items.append(path)

    for path in PROJECT_ROOT.rglob(".pytest_cache"):
        if "archive" not in path.parts and path.is_dir():
            items.append(path)

    for path in PROJECT_ROOT.rglob("*.pyc"):
        if "archive" not in path.parts and path.is_file():
            items.append(path)

    # Alt öğeleri önce silmek yerine üst cache klasörünü tercih et.
    unique: list[Path] = []

    for path in sorted(set(items), key=lambda item: len(item.parts)):
        if any(parent in unique for parent in path.parents):
            continue
        unique.append(path)

    return unique


def legacy_file_candidates() -> list[Path]:
    candidates = [
        PROJECT_ROOT / "extension" / "popup.html",
        PROJECT_ROOT / "extension" / "popup.css",
        PROJECT_ROOT / "extension" / "popup.js",
        PROJECT_ROOT / "extension" / "overlay.js",
        PROJECT_ROOT / "test.py",
    ]

    return [path for path in candidates if path.exists()]


def report_manual_cleanup_candidates() -> list[str]:
    candidates = [
        PROJECT_ROOT / "models" / "baselines",
        PROJECT_ROOT / "models" / "pilot",
        REPORTS_DIR / "electra_validation_uncertainty_predictions.csv",
        REPORTS_DIR / "electra_validation_uncertainty_thresholds.csv",
    ]

    lines: list[str] = []

    for path in candidates:
        if not path.exists():
            continue

        size = get_directory_size(path)
        lines.append(
            f"MANUEL İNCELEME: {path.relative_to(PROJECT_ROOT)} "
            f"({format_bytes(size)})"
        )

    return lines


def perform_safe_cleanup() -> str:
    lines: list[str] = []

    if not ENABLE_SAFE_CLEANUP:
        lines.append("Güvenli temizlik ayarı kapalı.")
        CONTEXT.cleanup_lines = lines
        return lines[0]

    if critical_failures():
        lines.append(
            "Kritik test hatası bulunduğu için otomatik temizlik yapılmadı."
        )
        CONTEXT.cleanup_lines = lines
        return lines[0]

    # 1. Cache dosyaları gerçekten silinebilir.
    removed_cache_count = 0
    removed_cache_bytes = 0

    for path in find_cache_items():
        try:
            size = get_directory_size(path)

            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

            removed_cache_count += 1
            removed_cache_bytes += size
            lines.append(
                f"SİLİNDİ: {path.relative_to(PROJECT_ROOT)} "
                f"({format_bytes(size)})"
            )

        except OSError as error:
            lines.append(
                f"SİLİNEMEDİ: {path.relative_to(PROJECT_ROOT)} — {error}"
            )

    # 2. Manifestte kullanılmayan eski dosyalar silinmez, arşivlenir.
    legacy_files = legacy_file_candidates()
    archived_count = 0
    archived_bytes = 0

    if legacy_files:
        legacy_archive = (
            ARCHIVE_DIR
            / f"legacy_cleanup_{TIMESTAMP}"
        )
        legacy_archive.mkdir(parents=True, exist_ok=True)

        for source in legacy_files:
            relative = source.relative_to(PROJECT_ROOT)
            destination = legacy_archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)

            size = get_directory_size(source)
            shutil.move(str(source), str(destination))

            archived_count += 1
            archived_bytes += size
            lines.append(
                f"ARŞİVLENDİ: {relative} -> "
                f"{destination.relative_to(PROJECT_ROOT)} "
                f"({format_bytes(size)})"
            )

    lines.extend(report_manual_cleanup_candidates())

    summary = (
        f"Cache: {removed_cache_count} öğe / "
        f"{format_bytes(removed_cache_bytes)} silindi. "
        f"Eski dosya: {archived_count} öğe / "
        f"{format_bytes(archived_bytes)} arşivlendi."
    )

    lines.insert(0, summary)
    CONTEXT.cleanup_lines = lines

    return summary


def write_cleanup_report() -> None:
    if not CONTEXT.cleanup_lines:
        CONTEXT.cleanup_lines = report_manual_cleanup_candidates()

    lines = [
        "=" * 78,
        "FINAL GÜVENLİ TEMİZLİK RAPORU",
        "=" * 78,
        "",
        *CONTEXT.cleanup_lines,
        "",
        "Not: models/baselines, models/pilot ve analiz raporları otomatik "
        "silinmez. Bunlar geliştirme sürecinin kanıtı olabilir.",
    ]

    CLEANUP_REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# 9. ANA AKIŞ
# ============================================================

def main() -> None:
    print_header("FINAL PROJE TESTİ BAŞLIYOR")
    print("Proje kökü:", PROJECT_ROOT)
    print("Python:", sys.version.split()[0])
    print("Güvenli temizlik:", ENABLE_SAFE_CLEANUP)

    print_header("1. PROJE DOSYALARI VE SÖZDİZİMİ")
    run_test("Dosya", "Zorunlu proje dosyaları", test_required_project_files)
    run_test("Dosya", "Python sözdizimi", test_python_syntax)
    run_test("Dosya", "JSON doğrulaması", test_json_files)
    run_test(
        "Dosya",
        "JavaScript sözdizimi",
        test_javascript_syntax,
        critical=False,
    )
    run_test("Model kaydı", "metrics.json", test_model_metrics_file)

    print_header("2. CHROME EKLENTİSİ")
    run_test("Eklenti", "Manifest yapısı", test_extension_manifest)
    run_test("Eklenti", "Backend bağlantısı", test_extension_backend_connection)
    run_test("Eklenti", "Sayfa sınırlandırmaları", test_extension_page_restrictions)
    run_test("Eklenti", "Siteye özel kurallar", test_site_specific_rules)

    print_header("3. MODEL VE MODEL SERVİSİ")
    run_test("Model", "Model import ve yükleme", test_model_import_and_load)

    # Model yüklenemezse bağımlı testler anlamlı değildir; yine de rapora yazılır.
    if CONTEXT.model_service is not None:
        run_test("Model", "Metin temizleme kuralları", test_clean_text_rules)
        run_test("Model", "Kısa metin reddi", test_short_text_rejection)
        run_test("Model", "Tek parça tahmin", test_single_chunk_prediction)
        run_test("Model", "Uzun metin chunking", test_long_text_chunking)
        run_test("Model", "4 REAL + 4 FAKE demo", test_demo_articles_8_of_8)
        run_test("Model", "Arka arkaya istek kararlılığı", test_stability_requests)
    else:
        for test_name in [
            "Metin temizleme kuralları",
            "Kısa metin reddi",
            "Tek parça tahmin",
            "Uzun metin chunking",
            "4 REAL + 4 FAKE demo",
            "Arka arkaya istek kararlılığı",
        ]:
            CONTEXT.results.append(
                TestResult(
                    category="Model",
                    name=test_name,
                    status="SKIP",
                    duration_seconds=0.0,
                    detail="Model yüklenemediği için atlandı.",
                    critical=True,
                )
            )

    print_header("4. FASTAPI ENTEGRASYONU")
    if CONTEXT.model_service is not None:
        run_test("API", "GET / ve /health", test_api_root_and_health)
        run_test("API", "POST /predict", test_api_predict_valid)
        run_test("API", "İstek doğrulama hataları", test_api_validation_errors)
        run_test("API", "CORS preflight", test_api_cors)
    else:
        print("- Model yüklenemediği için API testleri atlandı.")

    print_header("5. PERFORMANS")
    if CONTEXT.model_service is not None:
        run_test("Performans", "Kısa / orta / uzun tahmin", test_inference_performance)
        run_test(
            "Performans",
            "GPU bellek kullanımı",
            test_gpu_memory,
            critical=False,
        )
    else:
        print("- Model yüklenemediği için performans testleri atlandı.")

    print_header("6. VERİ VE KAYIT KONTROLLERİ")
    run_test(
        "Veri",
        "Train / validation / test splitleri",
        test_dataset_splits,
        critical=False,
    )
    run_test(
        "Veri",
        "6.000 kayıtlık mevcut tahmin dosyası",
        test_existing_6000_predictions,
        critical=False,
    )

    # Önce test raporları yazılır.
    write_demo_results()
    write_performance_results()
    write_text_report()

    print_header("7. GÜVENLİ TEMİZLİK")
    run_test(
        "Temizlik",
        "Cache temizliği ve eski dosya arşivleme",
        perform_safe_cleanup,
        critical=False,
    )
    write_cleanup_report()

    # Temizlik sonucunu da ana rapora eklemek için tekrar yaz.
    write_text_report()

    print_header("FINAL SONUÇ")

    pass_count = sum(result.status == "PASS" for result in CONTEXT.results)
    fail_count = sum(result.status == "FAIL" for result in CONTEXT.results)
    skip_count = sum(result.status == "SKIP" for result in CONTEXT.results)
    critical_error_count = len(critical_failures())

    print("PASS:", pass_count)
    print("FAIL:", fail_count)
    print("SKIP:", skip_count)
    print("Kritik hata:", critical_error_count)

    print("\nRaporlar:")
    print(" -", TEXT_REPORT_PATH)
    print(" -", PERFORMANCE_CSV_PATH)
    print(" -", DEMO_RESULTS_PATH)
    print(" -", CLEANUP_REPORT_PATH)

    if critical_error_count == 0:
        print("\nFINAL PROJE TESTİ BAŞARILI — KRİTİK HATA YOK")
    else:
        print("\nFINAL PROJE TESTİNDE KRİTİK HATALAR BULUNDU")
        print("Detay için final_system_test_report.txt dosyasını aç.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()