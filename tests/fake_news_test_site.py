# ============================================================
# ELECTRA KONTROLLÜ HABER DAVRANIŞ TEST SİTESİ
#
# Dosya:
# testsRA KONTROLLÜ HABER DAVRANIŞ TEST SİTESİ/fake_news_test_site.py
#
# Amaç:
# - Chrome eklentisini görsel olarak test etmek.
# - Modelin önceki davranış analizinde öğrendiğimiz
#   REAL / FAKE karar eğilimlerini yeniden doğrulamak.
# - Sunum için dengeli ve açıklanabilir sonuç üretmek.
#
# Test yapısı:
# - 3 kalibrasyon fake
# - 3 yeni genelleme fake
# - 2 real kontrol
#
# ÖNEMLİ:
# Bu içeriklerin tamamı yerel model testi için hazırlanmıştır.
# Gerçek haber olarak yayımlanmamalıdır.
# ============================================================

from __future__ import annotations

import html
import json
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse


# ============================================================
# FASTAPI UYGULAMASI
# ============================================================

app = FastAPI(
    title="ELECTRA Kontrollü Haber Test Sistemi",
    docs_url=None,
    redoc_url=None,
)


# ============================================================
# YENİ KONTROLLÜ TEST HABERLERİ
#
# 4 FAKE kontrolü:
# - Önceki tam sayfa testinde doğru sonuç verdi.
#
# 4 REAL kontrolü:
# - Biri daha önce doğrulanan TCMB örneğidir.
# - Diğerleri held-out test verisindeki doğal real haber
#   yapılarına göre hazırlanmıştır.
# ============================================================

TEST_ARTICLES: dict[str, dict[str, Any]] = {
    # ========================================================
    # BEKLENEN FAKE
    # ========================================================

    "sinirsiz-enerji-cihazi": {
        "expected_label": "fake",
        "test_group": "Doğrulanmış fake kontrolü",
        "test_feature": (
            "Fiziksel olarak aşırı teknoloji iddiası"
        ),
        "previous_probe_result": (
            "Önceki tam sayfa sonucu: FAKE %99,50"
        ),
        "title": (
            "Üniversite enerji kaynağı kullanmadan sınırsız "
            "elektrik üreten cihaz geliştirdi"
        ),
        "description": (
            "Yeni cihazın herhangi bir yakıt, batarya veya dış "
            "enerji bağlantısı olmadan yıllarca elektrik "
            "üretebildiği öne sürüldü."
        ),
        "paragraphs": [
            (
                "Kuzeykent Üniversitesindeki araştırmacıların "
                "herhangi bir enerji kaynağı kullanmadan sınırsız "
                "elektrik üreten yeni bir cihaz geliştirdiği "
                "iddia edildi. Cihazın prize, bataryaya, güneş "
                "paneline veya yakıta ihtiyaç duymadığı belirtildi."
            ),
            (
                "Laboratuvarda çalıştırılan sistemin yıllarca "
                "kesintisiz enerji üretebildiği ve herhangi bir "
                "bakım gerektirmediği öne sürüldü. Cihazın tek "
                "başına bütün bir şehrin elektrik ihtiyacını "
                "karşılayabileceği ileri sürüldü."
            ),
            (
                "Araştırmacıların sistemin mevcut fizik "
                "kurallarından etkilenmediğini açıkladığı iddia "
                "edildi. Teknolojinin elektrik faturalarını "
                "tamamen ortadan kaldıracağı ve kısa süre içinde "
                "bütün konutlarda kullanılacağı öne sürüldü."
            ),
        ],
    },

    "bilim-senliginde-100-bin-robot": {
        "expected_label": "fake",
        "test_group": "Doğrulanmış fake kontrolü",
        "test_feature": (
            "Aşırı üretim miktarı ve kesin başarı iddiası"
        ),
        "previous_probe_result": (
            "Önceki tam sayfa sonucu: FAKE %98,68"
        ),
        "title": (
            "Bilim şenliğinde öğrenciler bir günde "
            "100 bin robot üretti"
        ),
        "description": (
            "Şenliğe katılan öğrencilerin tek gün içerisinde "
            "100 bin çalışan robot tamamladığı iddia edildi."
        ),
        "paragraphs": [
            (
                "Kuzeykent'te düzenlenen bilim şenliğinde "
                "öğrencilerin bir gün içerisinde 100 bin farklı "
                "robot ürettiği bildirildi. Robotların tasarım, "
                "yazılım ve montaj işlemlerinin birkaç saat "
                "içerisinde tamamlandığı öne sürüldü."
            ),
            (
                "Hazırlanan robotların tamamının sorunsuz şekilde "
                "çalıştığı ve aynı anda sergi alanında gösterildiği "
                "iddia edildi. Projelerin enerji, sağlık, ulaşım "
                "ve uzay teknolojileri için geliştirildiği "
                "belirtildi."
            ),
            (
                "Öğrencilerin üretim sırasında herhangi bir hazır "
                "parça kullanmadığı ileri sürüldü. Yüz bin robotun "
                "tamamının şenlik sona ermeden vatandaşlara "
                "ücretsiz olarak dağıtıldığı iddia edildi."
            ),
        ],
    },

    "mezunlara-is-garantisi": {
        "expected_label": "fake",
        "test_group": "Doğrulanmış fake kontrolü",
        "test_feature": (
            "Koşulsuz ve ömür boyu garanti iddiası"
        ),
        "previous_probe_result": (
            "Önceki tam sayfa sonucu: FAKE %99,94"
        ),
        "title": (
            "Kuzeykent Üniversitesi bütün mezunlara sınavsız "
            "ve görüşmesiz iş garantisi verdi"
        ),
        "description": (
            "Programa katılan her öğrencinin mezuniyet gününde "
            "doğrudan işe yerleştirileceği iddia edildi."
        ),
        "paragraphs": [
            (
                "Kuzeykent Üniversitesi tarafından düzenlenen "
                "kariyer programına katılan bütün öğrencilere "
                "kesin iş garantisi verildiği iddia edildi. "
                "Öğrencilerin sınava veya iş görüşmesine "
                "katılmasına gerek olmadığı belirtildi."
            ),
            (
                "Farklı sektörlerden şirketlerin öğrencileri "
                "mezuniyet gününde doğrudan işe alacağı öne "
                "sürüldü. Not ortalaması, bölüm, tecrübe ve "
                "yabancı dil düzeyi dikkate alınmadan bütün "
                "mezunların yerleştirileceği ileri sürüldü."
            ),
            (
                "Üniversite yönetiminin hiçbir öğrencinin işsiz "
                "kalmayacağını garanti ettiği iddia edildi. "
                "Programa katılan mezunların ömür boyunca işten "
                "çıkarılamayacağı ve maaşlarının üniversite "
                "tarafından belirleneceği öne sürüldü."
            ),
        ],
    },

    "12-bin-otobus": {
        "expected_label": "fake",
        "test_group": "Sınırda fake kontrolü",
        "test_feature": (
            "Aşırı sayı ve kısa sürede tamamlanma iddiası"
        ),
        "previous_probe_result": (
            "Önceki tam sayfa sonucu: FAKE %65,20"
        ),
        "title": (
            "Kuzeykent Belediyesi 12 bin yeni otobüsü "
            "aynı gün hizmete aldı"
        ),
        "description": (
            "On iki bin yeni aracın bir gecede kontrol edilerek "
            "bütün güzergâhlarda sefere başladığı öne sürüldü."
        ),
        "paragraphs": [
            (
                "Kuzeykent Belediyesi, kent içi ulaşımda "
                "kullanılmak üzere on iki bin yeni otobüsün aynı "
                "gün hizmete alındığını açıkladı. Araçların bütün "
                "ilçelerde ve şehir merkezindeki güzergâhlarda "
                "çalışmaya başladığı bildirildi."
            ),
            (
                "Yeni otobüslerin teknik ve güvenlik "
                "kontrollerinin tek bir gecede tamamlandığı "
                "belirtildi. Belediye ekiplerinin bütün araçları "
                "birkaç saat içinde kayıt altına aldığı iddia "
                "edildi."
            ),
            (
                "Araçların hizmete alınmasıyla toplu taşımadaki "
                "bekleme süresinin tamamen ortadan kaldırıldığı "
                "ileri sürüldü. Vatandaşların artık duraklarda "
                "beklemeyeceği ve kesintisiz sefer yapılacağı "
                "açıklandı."
            ),
        ],
    },

    # ========================================================
    # BEKLENEN REAL
    # ========================================================

    "tcmb-faiz-karari": {
        "expected_label": "real",
        "test_group": "Doğrulanmış real kontrolü",
        "test_feature": (
            "Gerçekçi kurum açıklaması ve nötr ekonomi dili"
        ),
        "previous_probe_result": (
            "Önceki tam sayfa sonucu: REAL %99,98"
        ),
        "title": (
            "TCMB faiz kararını kurul toplantısının "
            "ardından açıklayacak"
        ),
        "description": (
            "Faiz kararının Para Politikası Kurulu toplantısının "
            "ardından kamuoyuyla paylaşılacağı bildirildi."
        ),
        "paragraphs": [
            (
                "Türkiye Cumhuriyet Merkez Bankası, faiz "
                "kararının Para Politikası Kurulu toplantısının "
                "ardından kamuoyuna açıklanacağını bildirdi. "
                "Toplantının ilan edilen takvim doğrultusunda "
                "gerçekleştirileceği belirtildi."
            ),
            (
                "Toplantıda para politikası görünümü, enflasyon "
                "gelişmeleri ve ekonomik verilerin "
                "değerlendirileceği ifade edildi. Kurul üyelerinin "
                "güncel ekonomik göstergeleri ele alacağı aktarıldı."
            ),
            (
                "Kararın kurumun resmî iletişim kanalları "
                "üzerinden duyurulacağı bildirildi. Piyasa "
                "katılımcılarının toplantının ardından yayımlanacak "
                "açıklamayı takip ettiği belirtildi."
            ),
        ],
    },

    "engin-altan-belgesel": {
        "expected_label": "real",
        "test_group": "Held-out real kontrolü",
        "test_feature": (
            "Doğal magazin ve kültür haberi anlatımı"
        ),
        "previous_probe_result": (
            "Yeni real kontrol — sonuç yeniden ölçülecek"
        ),
        "title": (
            "Engin Altan Düzyatan Afrika'daki göç "
            "yolculuğunu belgesel için görüntüleyecek"
        ),
        "description": (
            "Oyuncunun doğal yaşamı görüntülemek ve çektiği "
            "fotoğrafları sergilemek için Kenya'ya gittiği "
            "bildirildi."
        ),
        "paragraphs": [
            (
                "Engin Altan Düzyatan'ın eşi Neslişah Alkoçlar "
                "ile birlikte yeniden Kenya'ya gittiği bildirildi. "
                "Oyuncunun vahşi hayvanların yiyecek ve su bulmak "
                "için gerçekleştirdiği göç yolculuğunu "
                "görüntüleyeceği belirtildi."
            ),
            (
                "Düzyatan'ın yolculuk sırasında doğal yaşam "
                "fotoğrafları da çekeceği ifade edildi. Hazırlanan "
                "görüntülerin belgesel çalışmasında, fotoğrafların "
                "ise daha sonra açılacak bir sergide kullanılması "
                "planlanıyor."
            ),
            (
                "Sergiden elde edilecek gelirin bir bölümünün "
                "yardım çalışmalarına aktarılacağı bildirildi. "
                "Oyuncunun Afrika'daki su sorununa dikkat çekmeyi "
                "amaçladığı ifade edildi."
            ),
        ],
    },

    "cfg-lommel-satin-aldi": {
        "expected_label": "real",
        "test_group": "Held-out real kontrolü",
        "test_feature": (
            "Kurum, spor kulübü ve satın alma haberi"
        ),
        "previous_probe_result": (
            "Yeni real kontrol — sonuç yeniden ölçülecek"
        ),
        "title": (
            "City Football Group Belçika ekibi "
            "Lommel'in çoğunluk hisselerini satın aldı"
        ),
        "description": (
            "Manchester City'nin de bağlı olduğu grubun "
            "Belçika ekibi Lommel'i bünyesine kattığı açıklandı."
        ),
        "paragraphs": [
            (
                "Manchester City'yi de bünyesinde bulunduran "
                "City Football Group'un Belçika takımı Lommel'in "
                "çoğunluk hisselerini satın aldığı bildirildi. "
                "Anlaşmaya ilişkin açıklamanın şirket tarafından "
                "yapıldığı belirtildi."
            ),
            (
                "Lommel'in Belçika ikinci futbol liginde mücadele "
                "ettiği ve karşılaşmalarını Soeverein Stadı'nda "
                "oynadığı aktarıldı. Kulübün yeni yönetim yapısıyla "
                "faaliyetlerini sürdüreceği ifade edildi."
            ),
            (
                "City Football Group'un daha önce farklı ülkelerde "
                "New York City, Melbourne City, Girona ve Mumbai "
                "City gibi kulüplere yatırım yaptığı belirtildi. "
                "Lommel'in grubun bünyesine katılan kulüplerden "
                "biri olduğu bildirildi."
            ),
        ],
    },

    "bensu-soral-dolandirici-uyarisi": {
        "expected_label": "real",
        "test_group": "Held-out real kontrolü",
        "test_feature": (
            "Kişisel açıklama ve dolandırıcılık uyarısı"
        ),
        "previous_probe_result": (
            "Yeni real kontrol — sonuç yeniden ölçülecek"
        ),
        "title": (
            "Bensu Soral eski telefon numarasıyla ilgili "
            "dolandırıcılık uyarısı yaptı"
        ),
        "description": (
            "Oyuncu, daha önce kullandığı telefon numarası "
            "üzerinden para istendiğine dair duyumlar aldığını "
            "açıkladı."
        ),
        "paragraphs": [
            (
                "Oyuncu Bensu Soral, eski telefon numarasının "
                "dolandırıcılık girişimlerinde kullanıldığına "
                "ilişkin duyumlar aldığını açıkladı. Sosyal medya "
                "hesabından takipçilerine ve yakınlarına uyarıda "
                "bulundu."
            ),
            (
                "Soral, yıllar önce iptal ettirdiği numarayla artık "
                "herhangi bir bağlantısının bulunmadığını belirtti. "
                "Numaranın yeni sahibinin oyuncunun adını "
                "kullanarak kişilerden para istediğinin kendisine "
                "iletildiğini ifade etti."
            ),
            (
                "Eski numaradan gönderilen mesajlara itibar "
                "edilmemesini isteyen oyuncu, yaşanan durumla "
                "ilgili hukuki sürecin başlatıldığını bildirdi. "
                "Takipçilerinden şüpheli mesajları dikkate "
                "almamalarını istedi."
            ),
        ],
    },
}




# ============================================================
# HTML YARDIMCILARI
# ============================================================

def escape(
    value: Any,
) -> str:
    return html.escape(
        str(value),
        quote=True,
    )


def get_expected_text(
    expected_label: str,
) -> str:
    if expected_label == "fake":
        return "FAKE / Şüpheli Görünüyor"

    return "REAL / Güvenilir Görünüyor"


def get_expected_class(
    expected_label: str,
) -> str:
    if expected_label == "fake":
        return "fake"

    return "real"


def create_page(
    title: str,
    body: str,
) -> str:
    return f"""
<!doctype html>
<html lang="tr">
<head>
    <meta charset="utf-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >

    <title>{escape(title)}</title>

    <style>
        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            color: #1f2937;
            background: #eef2f7;
        }}

        .test-warning {{
            padding: 12px 20px;
            text-align: center;
            font-size: 13px;
            font-weight: 800;
            color: #7f1d1d;
            background: #fee2e2;
            border-bottom: 1px solid #fecaca;
        }}

        .container {{
            width: min(960px, calc(100% - 32px));
            margin: 28px auto;
        }}

        .card {{
            padding: 30px;
            background: white;
            border: 1px solid #dce3ec;
            border-radius: 16px;
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
        }}

        .section {{
            margin-top: 30px;
        }}

        .section:first-child {{
            margin-top: 0;
        }}

        .section-title {{
            margin: 0 0 8px;
            font-size: 23px;
        }}

        .section-description {{
            margin: 0 0 16px;
            color: #64748b;
            line-height: 1.6;
        }}

        .test-list {{
            display: grid;
            gap: 14px;
        }}

        .test-link {{
            display: block;
            padding: 18px;
            color: inherit;
            text-decoration: none;
            background: white;
            border: 1px solid #dce3ec;
            border-radius: 12px;
        }}

        .test-link:hover {{
            border-color: #315fe9;
            box-shadow: 0 8px 20px rgba(49, 95, 233, 0.08);
        }}

        .test-link-top {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 15px;
        }}

        .test-link strong {{
            display: block;
            line-height: 1.4;
        }}

        .test-link small {{
            display: block;
            margin-top: 8px;
            color: #64748b;
            line-height: 1.5;
        }}

        .expected-badge {{
            flex-shrink: 0;
            padding: 6px 9px;
            font-size: 11px;
            font-weight: 800;
            border-radius: 999px;
        }}

        .expected-badge.fake {{
            color: #991b1b;
            background: #fee2e2;
        }}

        .expected-badge.real {{
            color: #166534;
            background: #dcfce7;
        }}

        .test-info {{
            margin-bottom: 24px;
            padding: 16px;
            font-size: 13px;
            line-height: 1.65;
            border-radius: 10px;
        }}

        .test-info.fake {{
            color: #7f1d1d;
            background: #fee2e2;
            border: 1px solid #fecaca;
        }}

        .test-info.real {{
            color: #14532d;
            background: #dcfce7;
            border: 1px solid #bbf7d0;
        }}

        h1 {{
            margin: 0 0 15px;
            font-size: 34px;
            line-height: 1.2;
        }}

        .spot {{
            margin: 0 0 20px;
            font-size: 18px;
            font-weight: 700;
            line-height: 1.55;
            color: #4b5563;
        }}

        .meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px 18px;
            margin-bottom: 25px;
            padding-bottom: 16px;
            font-size: 13px;
            color: #6b7280;
            border-bottom: 1px solid #e5e7eb;
        }}

        article p {{
            margin: 0 0 18px;
            font-size: 17px;
            line-height: 1.75;
        }}

        .back-link {{
            display: inline-block;
            margin-top: 20px;
            color: #315fe9;
            font-weight: 700;
            text-decoration: none;
        }}

        .back-link:hover {{
            text-decoration: underline;
        }}

        @media (max-width: 650px) {{
            .test-link-top {{
                flex-direction: column;
            }}

            h1 {{
                font-size: 28px;
            }}

            .card {{
                padding: 22px;
            }}
        }}
    </style>
</head>

<body>
    <div class="test-warning">
        KONTROLLÜ MODEL TESTİ — BU İÇERİKLER GERÇEK HABER
        OLARAK PAYLAŞILMAMALIDIR
    </div>

    {body}
</body>
</html>
"""


# ============================================================
# TEST KARTLARI
# ============================================================

def create_test_cards(
    expected_label: str,
) -> str:
    cards: list[str] = []

    for slug, article in TEST_ARTICLES.items():
        if (
            article["expected_label"]
            != expected_label
        ):
            continue

        expected_class = get_expected_class(
            expected_label
        )

        cards.append(
            f"""
            <a
                class="test-link"
                href="/haber/{escape(slug)}"
            >
                <div class="test-link-top">
                    <strong>
                        {escape(article["title"])}
                    </strong>

                    <span
                        class="expected-badge {expected_class}"
                    >
                        {escape(expected_label.upper())}
                    </span>
                </div>

                <small>
                    {escape(article["test_group"])}
                    ·
                    {escape(article["test_feature"])}
                    <br>
                    {escape(article["previous_probe_result"])}
                </small>
            </a>
            """
        )

    return "".join(
        cards
    )


# ============================================================
# ANA SAYFA
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def index() -> HTMLResponse:
    fake_cards = create_test_cards(
        "fake"
    )

    real_cards = create_test_cards(
        "real"
    )

    body = f"""
    <main class="container">
        <section class="card">
            <h1>
                ELECTRA kontrollü haber testleri
            </h1>

            <p class="section-description">
                Bu sayfa modelin davranış analizinden sonra
                hazırlanmıştır. Bir haber sayfasını açtıktan sonra
                Chrome eklentisinin otomatik analiz bildirimini
                kullan.
            </p>

            <div class="section">
                <h2 class="section-title">
                    Beklenen FAKE testleri
                </h2>

                <p class="section-description">
                    İlk üç haber daha önce gözlenen davranışların
                    kalibrasyonudur. Son üç haber aynı özelliklerin
                    farklı içeriklere aktarılmasını test eder.
                </p>

                <div class="test-list">
                    {fake_cards}
                </div>
            </div>

            <div class="section">
                <h2 class="section-title">
                    Beklenen REAL kontrol testleri
                </h2>

                <p class="section-description">
                    Modelin yalnızca fake sonucu vermediğini ve iki
                    sınıfın da çalıştığını göstermek için hazırlanmış
                    kontrol örnekleridir.
                </p>

                <div class="test-list">
                    {real_cards}
                </div>
            </div>
        </section>
    </main>
    """

    return HTMLResponse(
        create_page(
            "ELECTRA Kontrollü Haber Testleri",
            body,
        )
    )


# ============================================================
# HABER DETAY SAYFASI
# ============================================================

@app.get(
    "/haber/{slug}",
    response_class=HTMLResponse,
)
def article_page(
    slug: str,
) -> HTMLResponse:
    article = TEST_ARTICLES.get(
        slug
    )

    if article is None:
        raise HTTPException(
            status_code=404,
            detail="Test haberi bulunamadı.",
        )

    paragraphs_html = "".join(
        f"<p>{escape(paragraph)}</p>"
        for paragraph in article["paragraphs"]
    )

    expected_label = str(
        article["expected_label"]
    )

    expected_class = get_expected_class(
        expected_label
    )

    expected_text = get_expected_text(
        expected_label
    )

    # Yalnızca gerçek haber metni articleBody içerisine konur.
    # Test açıklamaları model girdisine karışmaz.
    article_body_text = " ".join(
        article["paragraphs"]
    )

    json_ld = {
        "@context": "https://schema.org",
        "@type": "NewsArticle",
        "headline": article["title"],
        "description": article["description"],
        "datePublished": "2026-08-04T15:30:00+03:00",
        "author": {
            "@type": "Organization",
            "name": "Yerel Model Test Sistemi",
        },
        "articleBody": article_body_text,
    }

    body = f"""
    <main class="container">
        <div class="test-info {expected_class}">
            <strong>
                Beklenen sonuç:
                {escape(expected_text)}
            </strong>

            <br>

            Test grubu:
            {escape(article["test_group"])}

            <br>

            Test edilen özellik:
            {escape(article["test_feature"])}

            <br>

            {escape(article["previous_probe_result"])}
        </div>

        <article
            class="news-content"
            itemprop="articleBody"
        >
            <div class="card">
                <h1>
                    {escape(article["title"])}
                </h1>

                <p class="spot">
                    {escape(article["description"])}
                </p>

                <div class="meta">
                    <span itemprop="author">
                        Yerel model test sistemi
                    </span>

                    <time datetime="2026-08-04T15:30:00+03:00">
                        4 Ağustos 2026
                    </time>
                </div>

                {paragraphs_html}
            </div>
        </article>

        <a
            class="back-link"
            href="/"
        >
            ← Bütün testlere dön
        </a>

        <script type="application/ld+json">
            {json.dumps(json_ld, ensure_ascii=False)}
        </script>
    </main>
    """

    return HTMLResponse(
        create_page(
            article["title"],
            body,
        )
    )


# ============================================================
# PYCHARM ÜZERİNDEN BAŞLAT
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        reload=False,
    )