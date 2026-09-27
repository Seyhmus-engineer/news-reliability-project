(() => {
    "use strict";


    // ========================================================
    // TEMEL AYARLAR
    // ========================================================

    const INSTALLATION_FLAG =
        "__newsReliabilityContentScriptInstalled__";

    const PROMPT_HOST_ID =
        "news-reliability-analysis-prompt";

    const PANEL_HOST_ID =
        "news-reliability-analysis-panel";

    const NOTICE_HOST_ID =
        "news-reliability-analysis-notice";

    const MAX_TEXT_CHARACTER_COUNT = 75000;
    const MIN_ARTICLE_CHARACTER_COUNT = 600;
    const MIN_PARAGRAPH_LENGTH = 40;
    const MIN_PARAGRAPH_COUNT = 3;
    const MIN_DETECTION_SCORE = 7;

    const AUTO_DETECTION_DELAYS = [
        700,
        1800,
        3500,
        6000
    ];


    // ========================================================
    // HABER OLMAYAN SAYFA KURALLARI
    // ========================================================

    const WIKI_HOST_PATTERNS = [
        /(^|\.)wikipedia\.org$/i,
        /(^|\.)wikimedia\.org$/i,
        /(^|\.)wiktionary\.org$/i,
        /(^|\.)wikiquote\.org$/i,
        /(^|\.)wikibooks\.org$/i,
        /(^|\.)wikisource\.org$/i
    ];

    const EXCLUDED_PATH_PATTERNS = [
        /\/arama(\/|$)/i,
        /\/search(\/|$)/i,

        /\/kategori(\/|$)/i,
        /\/category(\/|$)/i,
        /\/categories(\/|$)/i,

        /\/etiket(\/|$)/i,
        /\/tag(\/|$)/i,
        /\/tags(\/|$)/i,

        /\/yazar(\/|$)/i,
        /\/author(\/|$)/i,
        /\/authors(\/|$)/i,

        /\/arsiv(\/|$)/i,
        /\/arşiv(\/|$)/i,
        /\/archive(\/|$)/i,

        /\/login(\/|$)/i,
        /\/signin(\/|$)/i,
        /\/signup(\/|$)/i,
        /\/uyelik(\/|$)/i,
        /\/üyelik(\/|$)/i,

        /\/hakkimizda(\/|$)/i,
        /\/hakkımızda(\/|$)/i,
        /\/about(\/|$)/i,

        /\/iletisim(\/|$)/i,
        /\/iletişim(\/|$)/i,
        /\/contact(\/|$)/i,

        /\/gizlilik(\/|$)/i,
        /\/privacy(\/|$)/i,

        /\/kullanim-kosullari(\/|$)/i,
        /\/kullanım-koşulları(\/|$)/i,
        /\/terms(\/|$)/i
    ];

    const CATEGORY_ROOT_PATTERNS = [
        /^\/gundem\/?$/i,
        /^\/gündem\/?$/i,
        /^\/dunya\/?$/i,
        /^\/dünya\/?$/i,
        /^\/ekonomi\/?$/i,
        /^\/spor\/?$/i,
        /^\/magazin\/?$/i,
        /^\/teknoloji\/?$/i,
        /^\/saglik\/?$/i,
        /^\/sağlık\/?$/i,
        /^\/kultur-sanat\/?$/i,
        /^\/kültür-sanat\/?$/i,
        /^\/politika\/?$/i,
        /^\/yerel\/?$/i,
        /^\/haberler\/?$/i,
        /^\/news\/?$/i
    ];

    const SEARCH_QUERY_KEYS = new Set([
        "q",
        "s",
        "search",
        "query",
        "keyword",
        "ara",
        "arama"
    ]);

    const NEWS_JSON_LD_TYPES = new Set([
        "newsarticle",
        "reportagenewsarticle",
        "analysisnewsarticle",
        "backgroundnewsarticle",
        "opinionnewsarticle",
        "reviewnewsarticle"
    ]);

    const GENERIC_ARTICLE_JSON_LD_TYPES = new Set([
        "article",
        "blogposting"
    ]);


    // ========================================================
    // ÇALIŞMA KONTROLLERİ
    // ========================================================

    if (window.top !== window) {
        return;
    }

    if (globalThis[INSTALLATION_FLAG]) {
        return;
    }

    globalThis[INSTALLATION_FLAG] = true;


    // ========================================================
    // UYGULAMA DURUMU
    // ========================================================

    const state = {
        detectedArticle: null,

        backendStatus: null,
        backendError: null,

        analysisInProgress: false,
        promptDismissed: false,

        currentUrl: window.location.href
    };


    // ========================================================
    // GENEL YARDIMCI FONKSİYONLAR
    // ========================================================

    function normalizeText(value) {
        if (typeof value !== "string") {
            return "";
        }

        return value
            .replace(/[\u200B-\u200D\uFEFF]/g, "")
            .replace(/\s+/g, " ")
            .trim();
    }


    function wait(milliseconds) {
        return new Promise(resolve => {
            setTimeout(resolve, milliseconds);
        });
    }


    function removeElementById(elementId) {
        const element = document.getElementById(
            elementId
        );

        if (element) {
            element.remove();
        }
    }


    function getCurrentUrl() {
        try {
            return new URL(
                window.location.href
            );

        } catch (error) {
            return null;
        }
    }
// ========================================================
// SİTEYE ÖZEL HABER KURALLARI
// ========================================================

function getCurrentSiteRule() {
    const rules = Array.isArray(
        globalThis.HG_SITE_RULES
    )
        ? globalThis.HG_SITE_RULES
        : [];

    const hostname = window.location.hostname
        .toLocaleLowerCase("tr-TR");

    return (
        rules.find(rule => {
            return (
                Array.isArray(rule.hostnames)
                && rule.hostnames.some(ruleHostname => {
                    const normalizedRuleHostname =
                        String(ruleHostname)
                            .toLocaleLowerCase("tr-TR");

                    return (
                        hostname === normalizedRuleHostname
                        || hostname.endsWith(
                            `.${normalizedRuleHostname}`
                        )
                    );
                })
            );
        })
        || null
    );
}


function getTextFromSelectors(selectors) {
    if (!Array.isArray(selectors)) {
        return "";
    }

    for (const selector of selectors) {
        let elements = [];

        try {
            elements = Array.from(
                document.querySelectorAll(selector)
            );

        } catch (error) {
            console.warn(
                "[HG] Geçersiz site seçicisi:",
                selector
            );

            continue;
        }

        for (const element of elements) {
            if (!isElementVisible(element)) {
                continue;
            }

            const text = normalizeText(
                element.textContent || ""
            );

            if (text) {
                return text;
            }
        }
    }

    return "";
}


function isExcludedBySiteRule(element) {
    const siteRule = getCurrentSiteRule();

    if (
        !siteRule
        || !Array.isArray(
            siteRule.excludedSelectors
        )
    ) {
        return false;
    }

    for (
        const selector
        of siteRule.excludedSelectors
    ) {
        try {
            if (element.closest(selector)) {
                return true;
            }

        } catch (error) {
            console.warn(
                "[HG] Geçersiz dışlama seçicisi:",
                selector
            );
        }
    }

    return false;
}


    function clampPercentage(value) {
        const numericValue = Number(value);

        if (!Number.isFinite(numericValue)) {
            return 0;
        }

        return Math.min(
            100,
            Math.max(
                0,
                numericValue * 100
            )
        );
    }


    function formatPercentage(value) {
        return `%${clampPercentage(value).toLocaleString(
            "tr-TR",
            {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            }
        )}`;
    }


    function formatNumber(value) {
        const numericValue = Number(value);

        if (!Number.isFinite(numericValue)) {
            return "0";
        }

        return numericValue.toLocaleString(
            "tr-TR"
        );
    }


    function sendExtensionMessage(message) {
        return new Promise(
            (resolve, reject) => {
                chrome.runtime.sendMessage(
                    message,
                    response => {
                        const runtimeError =
                            chrome.runtime.lastError;

                        if (runtimeError) {
                            reject(
                                new Error(
                                    runtimeError.message
                                )
                            );

                            return;
                        }

                        resolve(response);
                    }
                );
            }
        );
    }


    // ========================================================
    // BACKEND DURUMU
    // ========================================================

    function getBackendView(
        backendStatus,
        backendError = null
    ) {
        const model = backendStatus?.model;

        const ready = (
            backendStatus?.status === "healthy"
            && model?.loaded === true
        );

        if (!ready) {
            return {
                ready: false,
                className: "offline",

                text: (
                    backendError
                    || "Backend veya model bağlantısı yok"
                )
            };
        }

        const device = String(
            model?.device || ""
        ).toLocaleLowerCase("tr-TR");

        if (
            device === "cuda"
            || device.startsWith("cuda:")
        ) {
            return {
                ready: true,
                className: "gpu",

                text: (
                    "Model hazır — GPU: "
                    + `${model?.gpu || "CUDA"}`
                )
            };
        }

        return {
            ready: true,
            className: "cpu",
            text: "Model hazır — CPU kullanılıyor"
        };
    }


    async function getBackendHealth() {
        try {
            const response =
                await sendExtensionMessage({
                    type: "CHECK_BACKEND"
                });

            if (!response?.ok) {
                state.backendStatus = null;

                state.backendError = (
                    response?.error
                    || "Backend bağlantısı kurulamadı."
                );

                return null;
            }

            state.backendStatus = response.data;
            state.backendError = null;

            return response.data;

        } catch (error) {
            state.backendStatus = null;

            state.backendError = (
                error instanceof Error
                    ? error.message
                    : "Backend bağlantısı kurulamadı."
            );

            return null;
        }
    }


    // ========================================================
    // META ETİKETLERİ
    // ========================================================

    function getMetaContent(selectors) {
        for (const selector of selectors) {
            const element =
                document.querySelector(selector);

            if (!element) {
                continue;
            }

            const content = normalizeText(
                element.getAttribute("content") || ""
            );

            if (content) {
                return content;
            }
        }

        return "";
    }


    // ========================================================
    // JSON-LD ANALİZİ
    // ========================================================

    function normalizeJsonLdType(typeValue) {
        return normalizeText(
            String(typeValue || "")
        ).toLocaleLowerCase("tr-TR");
    }


    function inspectJsonLdValue(
        value,
        information
    ) {
        if (!value) {
            return;
        }

        if (Array.isArray(value)) {
            for (const item of value) {
                inspectJsonLdValue(
                    item,
                    information
                );
            }

            return;
        }

        if (typeof value !== "object") {
            return;
        }

        const rawTypes = Array.isArray(
            value["@type"]
        )
            ? value["@type"]
            : [value["@type"]];

        for (const rawType of rawTypes) {
            const type =
                normalizeJsonLdType(rawType);

            if (NEWS_JSON_LD_TYPES.has(type)) {
                information.hasNewsArticle = true;
            }

            if (
                GENERIC_ARTICLE_JSON_LD_TYPES
                    .has(type)
            ) {
                information.hasGenericArticle = true;
            }
        }

        if (
            normalizeText(
                String(value.articleBody || "")
            ).length >= 100
        ) {
            information.hasArticleBody = true;
        }

        if (
            value.datePublished
            || value.dateCreated
            || value.uploadDate
        ) {
            information.hasPublishedDate = true;
        }

        if (
            value.author
            || value.creator
        ) {
            information.hasAuthor = true;
        }

        for (const nestedValue of Object.values(
            value
        )) {
            inspectJsonLdValue(
                nestedValue,
                information
            );
        }
    }


    function getJsonLdInformation() {
        const information = {
            hasNewsArticle: false,
            hasGenericArticle: false,
            hasArticleBody: false,
            hasPublishedDate: false,
            hasAuthor: false
        };

        const scripts = Array.from(
            document.querySelectorAll(
                "script[type='application/ld+json']"
            )
        ).slice(0, 40);

        for (const script of scripts) {
            try {
                const parsed = JSON.parse(
                    script.textContent || "{}"
                );

                inspectJsonLdValue(
                    parsed,
                    information
                );

            } catch (error) {
                // Geçersiz JSON-LD kaydı atlanır.
            }
        }

        return information;
    }


    // ========================================================
    // SAYFA TÜRÜNÜ ENGELLEME
    // ========================================================

    function isWikiPage(url) {
        if (!url) {
            return false;
        }

        const hostname = url.hostname
            .toLocaleLowerCase("tr-TR");

        if (
            WIKI_HOST_PATTERNS.some(pattern => {
                return pattern.test(hostname);
            })
        ) {
            return true;
        }

        const mediaWikiGenerator =
            getMetaContent([
                "meta[name='generator']"
            ]).toLocaleLowerCase("tr-TR");

        return mediaWikiGenerator.includes(
            "mediawiki"
        );
    }


    function hasSearchQuery(url) {
        if (!url) {
            return false;
        }

        for (const key of url.searchParams.keys()) {
            if (
                SEARCH_QUERY_KEYS.has(
                    key.toLocaleLowerCase("tr-TR")
                )
            ) {
                return true;
            }
        }

        return false;
    }


    function isExcludedPath(url) {
        if (!url) {
            return true;
        }

        const path = decodeURIComponent(
            url.pathname || "/"
        );

        if (
            EXCLUDED_PATH_PATTERNS.some(
                pattern => pattern.test(path)
            )
        ) {
            return true;
        }

        return CATEGORY_ROOT_PATTERNS.some(
            pattern => pattern.test(path)
        );
    }


    function getPageExclusionReason() {
        const url = getCurrentUrl();

        if (!url) {
            return "Sayfa adresi okunamadı.";
        }

        if (
            url.protocol !== "http:"
            && url.protocol !== "https:"
        ) {
            return (
                "Bu sayfa türü haber analizi için "
                + "desteklenmiyor."
            );
        }

        if (isWikiPage(url)) {
            return (
                "Ansiklopedi veya wiki sayfaları "
                + "haber olarak analiz edilmez."
            );
        }

        if (
            url.pathname === "/"
            || url.pathname === ""
        ) {
            return (
                "Site ana sayfası haber detay "
                + "sayfası değildir."
            );
        }

        if (hasSearchQuery(url)) {
            return (
                "Arama sonuçları haber olarak "
                + "analiz edilmez."
            );
        }

        if (isExcludedPath(url)) {
            return (
                "Kategori, arama, etiket, yazar "
                + "veya liste sayfası analiz edilmez."
            );
        }

        return null;
    }


    // ========================================================
    // HTML ELEMANI KONTROLLERİ
    // ========================================================

    function isElementVisible(element) {
        if (!(element instanceof HTMLElement)) {
            return false;
        }

        const style =
            window.getComputedStyle(element);

        if (
            style.display === "none"
            || style.visibility === "hidden"
            || Number(style.opacity) === 0
        ) {
            return false;
        }

        const rectangle =
            element.getBoundingClientRect();

        return (
            rectangle.width > 0
            && rectangle.height > 0
        );
    }


function isUnwantedElement(element) {
    if (
        element.closest(
            "nav, footer, aside, form, "
            + "[role='navigation'], "
            + "[aria-hidden='true']"
        )
    ) {
        return true;
    }

    /*
     * Aktif haber sitesi için tanımlanan özel dışlama
     * kurallarını uygular.
     */
    if (isExcludedBySiteRule(element)) {
        return true;
    }

    const classAndId = normalizeText(
        `${String(element.className)} ${element.id}`
    ).toLocaleLowerCase("tr-TR");

    const unwantedPattern =
        /cookie|advert|advertisement|reklam|yorum|comment|social|share|paylaş|footer|menu|breadcrumb|related|ilgili|recommend|newsletter|abonelik|subscription|gallery|caption|promo|banner|most-read|cok-okunan|çok-okunan/;

    return unwantedPattern.test(
        classAndId
    );
}


    // ========================================================
    // BAŞLIK VE AÇIKLAMA
    // ========================================================

 function extractTitle() {
    const siteRule =
        getCurrentSiteRule();

    const siteSpecificTitle =
        getTextFromSelectors(
            siteRule?.titleSelectors || []
        );

    const candidates = [
        siteSpecificTitle,

        normalizeText(
            document.querySelector(
                "article h1"
            )?.textContent || ""
        ),

        normalizeText(
            document.querySelector(
                "main h1"
            )?.textContent || ""
        ),

        normalizeText(
            document.querySelector("h1")
                ?.textContent || ""
        ),

        getMetaContent([
            "meta[property='og:title']",
            "meta[name='twitter:title']"
        ]),

        normalizeText(document.title)
    ];

    return (
        candidates.find(candidate => {
            return (
                candidate.length >= 10
                && candidate.length <= 500
            );
        })
        || "Başlık bulunamadı"
    );
}


function extractDescription(title) {
    const candidates = [];

    const siteRule =
        getCurrentSiteRule();

    const siteSpecificDescription =
        getTextFromSelectors(
            siteRule?.descriptionSelectors
            || []
        );

    if (
        siteSpecificDescription
        && siteSpecificDescription !== title
    ) {
        candidates.push(
            siteSpecificDescription
        );
    }

    const metaDescription =
        getMetaContent([
            "meta[property='og:description']",
            "meta[name='description']",
            "meta[name='twitter:description']"
        ]);

    if (
        metaDescription
        && metaDescription !== title
    ) {
        candidates.push(
            metaDescription
        );
    }

    const generalSelectors = [
        ".spot",
        ".article-spot",
        ".news-spot",
        ".lead",
        ".summary",
        ".article-summary",
        ".haber-spot",
        ".detail-spot",
        "[itemprop='description']"
    ];

    for (const selector of generalSelectors) {
        const elements =
            document.querySelectorAll(selector);

        for (const element of elements) {
            if (!isElementVisible(element)) {
                continue;
            }

            const text = normalizeText(
                element.textContent || ""
            );

            if (
                text.length >= 30
                && text !== title
            ) {
                candidates.push(text);
            }
        }
    }

    return (
        candidates.find(candidate => {
            return (
                candidate.length >= 30
                && candidate.length <= 2000
            );
        })
        || ""
    );
}


    // ========================================================
    // HABER KÖK ELEMANINI BUL
    // ========================================================

    function calculateRootTextLength(element) {
        const paragraphs = Array.from(
            element.querySelectorAll("p")
        );

        return paragraphs.reduce(
            (total, paragraph) => {
                if (
                    !isElementVisible(paragraph)
                    || isUnwantedElement(paragraph)
                ) {
                    return total;
                }

                return (
                    total
                    + normalizeText(
                        paragraph.textContent || ""
                    ).length
                );
            },
            0
        );
    }


function findBestArticleRoot() {
    const siteRule =
        getCurrentSiteRule();

    const siteSpecificSelectors =
        Array.isArray(
            siteRule?.bodySelectors
        )
            ? siteRule.bodySelectors
            : [];

    const genericSelectors = [
        "[itemprop='articleBody']",
        "article",

        ".article-body",
        ".article-content",
        ".article-detail",
        ".article-text",

        ".news-content",
        ".news-detail",
        ".news-body",

        ".haber-icerik",
        ".haber-içerik",
        ".haber-detay",
        ".haber-metni",

        ".post-content",
        ".entry-content",
        ".detail-content",

        "main",
        "[role='main']"
    ];

    /*
     * Siteye özel seçiciler önce denenir.
     */
    const selectors = [
        ...siteSpecificSelectors,
        ...genericSelectors
    ];

    const candidates = [];
    const seenElements = new Set();

    for (const selector of selectors) {
        let elements = [];

        try {
            elements = Array.from(
                document.querySelectorAll(selector)
            );

        } catch (error) {
            console.warn(
                "[HG] Geçersiz haber gövdesi seçicisi:",
                selector
            );

            continue;
        }

        for (const element of elements) {
            if (
                seenElements.has(element)
                || !(element instanceof HTMLElement)
            ) {
                continue;
            }

            seenElements.add(element);

            const textLength =
                calculateRootTextLength(element);

            if (textLength <= 0) {
                continue;
            }

            candidates.push({
                element,
                textLength,

                /*
                 * Siteye özel seçicilerle bulunan adaylara
                 * öncelik verilir.
                 */
                siteSpecific: (
                    siteSpecificSelectors
                        .includes(selector)
                )
            });
        }
    }

    candidates.sort(
        (first, second) => {
            if (
                first.siteSpecific
                !== second.siteSpecific
            ) {
                return (
                    first.siteSpecific
                        ? -1
                        : 1
                );
            }

            return (
                second.textLength
                - first.textLength
            );
        }
    );

    const selectedRoot =
        candidates[0]?.element
        || document.body;

    console.info(
        "[HG] İçerik çıkarma yöntemi:",
        {
            site: (
                siteRule?.name
                || "Genel çıkarıcı"
            ),

            rootTag:
                selectedRoot.tagName,

            rootClass:
                String(
                    selectedRoot.className || ""
                ),

            candidateCount:
                candidates.length
        }
    );

    return selectedRoot;
}


    // ========================================================
    // HABER PARAGRAFLARINI ÇIKAR
    // ========================================================

    function collectParagraphs(elements) {
        const paragraphs = [];
        const seenTexts = new Set();

        let totalCharacterCount = 0;

        for (const element of elements) {
            if (!isElementVisible(element)) {
                continue;
            }

            if (isUnwantedElement(element)) {
                continue;
            }

            const text = normalizeText(
                element.textContent || ""
            );

            if (
                text.length
                < MIN_PARAGRAPH_LENGTH
            ) {
                continue;
            }

            const comparisonKey =
                text.toLocaleLowerCase("tr-TR");

            if (seenTexts.has(comparisonKey)) {
                continue;
            }

            if (
                totalCharacterCount + text.length
                > MAX_TEXT_CHARACTER_COUNT
            ) {
                break;
            }

            seenTexts.add(comparisonKey);
            paragraphs.push(text);

            totalCharacterCount += text.length;

            if (paragraphs.length >= 200) {
                break;
            }
        }

        return paragraphs;
    }


    function extractParagraphs(root) {
        let paragraphElements = Array.from(
            root.querySelectorAll("p")
        );

        let paragraphs = collectParagraphs(
            paragraphElements
        );

        const totalLength = paragraphs.reduce(
            (total, paragraph) => {
                return total + paragraph.length;
            },
            0
        );

        if (
            paragraphs.length
            < MIN_PARAGRAPH_COUNT
            || totalLength < 500
        ) {
            paragraphElements = Array.from(
                document.querySelectorAll("p")
            );

            paragraphs = collectParagraphs(
                paragraphElements
            );
        }

        return paragraphs;
    }


    // ========================================================
    // HABER İŞARETLERİ
    // ========================================================

    function hasPublishedDateElement() {
        if (
            document.querySelector(
                "time[datetime]"
            )
        ) {
            return true;
        }

        const selectors = [
            "[itemprop='datePublished']",
            "meta[property='article:published_time']",
            "meta[name='date']",
            ".publish-date",
            ".published-date",
            ".article-date",
            ".news-date",
            ".haber-tarih"
        ];

        return selectors.some(selector => {
            return Boolean(
                document.querySelector(selector)
            );
        });
    }


    function hasAuthorElement() {
        const selectors = [
            "[itemprop='author']",
            "meta[name='author']",
            "meta[property='article:author']",
            ".author",
            ".article-author",
            ".news-author",
            ".haber-yazar"
        ];

        return selectors.some(selector => {
            return Boolean(
                document.querySelector(selector)
            );
        });
    }


    function getPathDepth() {
        const url = getCurrentUrl();

        if (!url) {
            return 0;
        }

        return url.pathname
            .split("/")
            .filter(Boolean)
            .length;
    }


    // ========================================================
    // HABER ALGILAMA PUANI
    // ========================================================

    function calculateDetectionInformation(
        paragraphCount,
        characterCount
    ) {
        const jsonLd =
            getJsonLdInformation();

        const openGraphType =
            getMetaContent([
                "meta[property='og:type']"
            ]).toLocaleLowerCase("tr-TR");

        const hasNewsJsonLd =
            jsonLd.hasNewsArticle;

        const hasGenericArticleJsonLd =
            jsonLd.hasGenericArticle;

        const hasArticleBody = (
            jsonLd.hasArticleBody
            || Boolean(
                document.querySelector(
                    "[itemprop='articleBody']"
                )
            )
        );

        const hasPublishedDate = (
            jsonLd.hasPublishedDate
            || hasPublishedDateElement()
        );

        const hasAuthor = (
            jsonLd.hasAuthor
            || hasAuthorElement()
        );

        const hasArticleElement = Boolean(
            document.querySelector("article")
        );

        const hasHeading = (
            normalizeText(
                document.querySelector("h1")
                    ?.textContent || ""
            ).length >= 10
        );

        let score = 0;

        if (hasNewsJsonLd) {
            score += 5;
        }

        if (
            openGraphType === "article"
        ) {
            score += 3;
        }

        if (hasArticleBody) {
            score += 3;
        }

        if (hasGenericArticleJsonLd) {
            score += 1;
        }

        if (hasArticleElement) {
            score += 1;
        }

        if (hasPublishedDate) {
            score += 2;
        }

        if (hasAuthor) {
            score += 1;
        }

        if (hasHeading) {
            score += 1;
        }

        if (paragraphCount >= 6) {
            score += 2;

        } else if (
            paragraphCount
            >= MIN_PARAGRAPH_COUNT
        ) {
            score += 1;
        }

        if (characterCount >= 1500) {
            score += 2;

        } else if (
            characterCount
            >= MIN_ARTICLE_CHARACTER_COUNT
        ) {
            score += 1;
        }

        if (getPathDepth() >= 2) {
            score += 1;
        }

        const hasStrongNewsSignal = (
            hasNewsJsonLd
            || hasArticleBody
            || (
                openGraphType === "article"
                && hasPublishedDate
            )
            || (
                hasArticleElement
                && hasPublishedDate
                && getPathDepth() >= 2
            )
        );

        return {
            score,
            hasStrongNewsSignal,
            hasNewsJsonLd,
            hasGenericArticleJsonLd,
            hasArticleBody,
            hasPublishedDate,
            hasAuthor,
            hasArticleElement,
            openGraphType
        };
    }


    // ========================================================
    // HABER METNİNİ OLUŞTUR
    // ========================================================

    function extractArticle() {
        const exclusionReason =
            getPageExclusionReason();

        if (exclusionReason) {
            const rejectedResult = {
                ok: false,
                reason: exclusionReason,
                title: "",
                text: "",
                paragraphCount: 0,
                characterCount: 0,
                score: 0,
                url: window.location.href
            };

            console.info(
                "[HG] Sayfa algılama sonucu:",
                rejectedResult
            );

            return rejectedResult;
        }

        const title = extractTitle();

        const description =
            extractDescription(title);

        const root =
            findBestArticleRoot();

        const paragraphs =
            extractParagraphs(root);

        const textParts = [];
        const seenParts = new Set();

        for (const part of [
            title,
            description,
            ...paragraphs
        ]) {
            const normalizedPart =
                normalizeText(part);

            if (!normalizedPart) {
                continue;
            }

            const comparisonKey =
                normalizedPart
                    .toLocaleLowerCase("tr-TR");

            if (seenParts.has(comparisonKey)) {
                continue;
            }

            seenParts.add(comparisonKey);
            textParts.push(normalizedPart);
        }

        const text = textParts
            .join("\n\n")
            .slice(
                0,
                MAX_TEXT_CHARACTER_COUNT
            );

        const detection =
            calculateDetectionInformation(
                paragraphs.length,
                text.length
            );

        let reason = null;

        if (
            title === "Başlık bulunamadı"
        ) {
            reason =
                "Haber başlığı bulunamadı.";

        } else if (
            paragraphs.length
            < MIN_PARAGRAPH_COUNT
        ) {
            reason =
                "Yeterli haber paragrafı bulunamadı.";

        } else if (
            text.length
            < MIN_ARTICLE_CHARACTER_COUNT
        ) {
            reason =
                "Haber metni yeterince uzun değil.";

        } else if (
            !detection.hasStrongNewsSignal
        ) {
            reason =
                "Sayfa yeterli haber işaretlerini taşımıyor.";

        } else if (
            detection.score
            < MIN_DETECTION_SCORE
        ) {
            reason =
                "Haber algılama puanı yeterli değil.";
        }

        const result = {
            ok: reason === null,
            reason,

            title,
            description,
            text,

            paragraphCount:
                paragraphs.length,

            characterCount:
                text.length,

            score:
                detection.score,

            detection,
            url: window.location.href
        };

        console.info(
            "[HG] Sayfa algılama sonucu:",
            result
        );

        return result;
    }


    // ========================================================
    // KÜÇÜK BİLGİ MESAJI
    // ========================================================

    function showNotice(message) {
        removeElementById(
            NOTICE_HOST_ID
        );

        const host =
            document.createElement("div");

        host.id = NOTICE_HOST_ID;

        const shadow = host.attachShadow({
            mode: "open"
        });

        shadow.innerHTML = `
            <style>
                :host {
                    all: initial;
                }

                .notice {
                    position: fixed;
                    right: 22px;
                    bottom: 22px;
                    z-index: 2147483647;

                    max-width: 360px;
                    padding: 13px 16px;

                    font-family:
                        Arial,
                        Helvetica,
                        sans-serif;

                    font-size: 13px;
                    line-height: 1.45;

                    color: #ffffff;
                    background: #28344b;

                    border-radius: 10px;

                    box-shadow:
                        0 12px 30px
                        rgba(15, 23, 42, 0.25);
                }
            </style>

            <div class="notice"></div>
        `;

        shadow.querySelector(
            ".notice"
        ).textContent = message;

        document.documentElement.appendChild(
            host
        );

        setTimeout(
            () => {
                host.remove();
            },
            4500
        );
    }


    // ========================================================
    // ANALİZ ONAY BİLDİRİMİ
    // ========================================================

    function showAnalysisPrompt(
        article,
        backendStatus = null,
        backendError = null
    ) {
        if (
            document.getElementById(
                PANEL_HOST_ID
            )
        ) {
            return;
        }

        removeElementById(
            PROMPT_HOST_ID
        );

        state.detectedArticle = article;
        state.backendStatus = backendStatus;
        state.backendError = backendError;

        const backendView = getBackendView(
            backendStatus,
            backendError
        );

        const host =
            document.createElement("div");

        host.id = PROMPT_HOST_ID;

        const shadow = host.attachShadow({
            mode: "open"
        });

        shadow.innerHTML = `
            <style>
                :host {
                    all: initial;
                }

                * {
                    box-sizing: border-box;
                }

                .prompt {
                    position: fixed;
                    right: 22px;
                    bottom: 22px;
                    z-index: 2147483647;

                    width: min(
                        390px,
                        calc(100vw - 32px)
                    );

                    padding: 16px;

                    font-family:
                        Arial,
                        Helvetica,
                        sans-serif;

                    color: #172033;
                    background: #ffffff;

                    border:
                        1px solid #d7dfeb;

                    border-radius: 14px;

                    box-shadow:
                        0 18px 45px
                        rgba(15, 23, 42, 0.22);

                    animation:
                        hg-prompt-open
                        0.25s ease-out;
                }

                @keyframes hg-prompt-open {
                    from {
                        opacity: 0;
                        transform:
                            translateY(12px);
                    }

                    to {
                        opacity: 1;
                        transform:
                            translateY(0);
                    }
                }

                .header {
                    display: flex;
                    align-items: center;
                    gap: 10px;

                    margin-bottom: 11px;
                }

                .logo {
                    width: 39px;
                    height: 39px;

                    display: flex;
                    align-items: center;
                    justify-content: center;

                    flex-shrink: 0;

                    font-size: 13px;
                    font-weight: 800;

                    color: #ffffff;
                    background: #315fe9;

                    border-radius: 10px;
                }

                h2 {
                    margin: 0;

                    font-size: 15px;
                    line-height: 1.35;

                    color: #172033;
                }

                .article-title {
                    margin: 0 0 10px;

                    display: -webkit-box;
                    overflow: hidden;

                    -webkit-line-clamp: 2;
                    -webkit-box-orient: vertical;

                    font-size: 12px;
                    font-weight: 700;
                    line-height: 1.5;

                    color: #445168;
                }

                .status {
                    display: flex;
                    align-items: center;
                    gap: 8px;

                    margin-bottom: 11px;
                    padding: 9px 10px;

                    font-size: 11px;
                    font-weight: 700;
                    line-height: 1.4;

                    border-radius: 8px;
                }

                .status.gpu {
                    color: #087544;
                    background: #dcf7e9;
                }

                .status.cpu {
                    color: #8a5707;
                    background: #fff1cf;
                }

                .status.offline {
                    color: #a22132;
                    background: #ffe2e6;
                }

                .status-dot {
                    width: 9px;
                    height: 9px;

                    flex-shrink: 0;

                    background: currentColor;
                    border-radius: 50%;
                }

                .message {
                    margin: 0 0 14px;

                    font-size: 12px;
                    line-height: 1.5;

                    color: #667389;
                }

                .buttons {
                    display: flex;
                    gap: 8px;
                }

                button {
                    min-height: 38px;
                    padding: 9px 13px;

                    font-family: inherit;
                    font-size: 12px;
                    font-weight: 700;

                    border: none;
                    border-radius: 9px;

                    cursor: pointer;
                }

                .analyze {
                    flex: 1;

                    color: #ffffff;
                    background: #315fe9;
                }

                .analyze:hover:not(:disabled) {
                    background: #264fcd;
                }

                .analyze:disabled {
                    opacity: 0.65;
                    cursor: not-allowed;
                }

                .dismiss {
                    color: #536078;
                    background: #edf1f6;
                }

                .dismiss:hover {
                    background: #e0e6ee;
                }
            </style>

            <section
                class="prompt"
                role="dialog"
                aria-label="Haber analizi önerisi"
            >
                <div class="header">
                    <div class="logo">
                        HG
                    </div>

                    <h2>
                        Haber içeriği algılandı
                    </h2>
                </div>

                <p class="article-title"></p>

                <div class="status">
                    <span class="status-dot"></span>
                    <span class="status-text"></span>
                </div>

                <p class="message">
                    Bu sayfadaki haber metni
                    güvenilirlik modeliyle analiz
                    edilsin mi?
                </p>

                <div class="buttons">
                    <button
                        type="button"
                        class="analyze"
                    >
                        Analiz Et
                    </button>

                    <button
                        type="button"
                        class="dismiss"
                    >
                        Şimdi Değil
                    </button>
                </div>
            </section>
        `;

        shadow.querySelector(
            ".article-title"
        ).textContent = article.title;

        const statusElement =
            shadow.querySelector(".status");

        statusElement.classList.add(
            backendView.className
        );

        shadow.querySelector(
            ".status-text"
        ).textContent = backendView.text;

        const analyzeButton =
            shadow.querySelector(".analyze");

        analyzeButton.disabled =
            !backendView.ready;

        if (!backendView.ready) {
            analyzeButton.textContent =
                "Model Kullanıma Hazır Değil";
        }

        analyzeButton.addEventListener(
            "click",
            () => {
                if (!backendView.ready) {
                    return;
                }

                state.promptDismissed = true;

                host.remove();

                startAnalysis(article);
            }
        );

        shadow.querySelector(
            ".dismiss"
        ).addEventListener(
            "click",
            () => {
                state.promptDismissed = true;
                host.remove();
            }
        );

        document.documentElement.appendChild(
            host
        );
    }


    // ========================================================
    // SAĞ SONUÇ PANELİ
    // ========================================================

    function createSidePanel(article) {
        removeElementById(
            PROMPT_HOST_ID
        );

        removeElementById(
            PANEL_HOST_ID
        );

        const backendView = getBackendView(
            state.backendStatus,
            state.backendError
        );

        const host =
            document.createElement("div");

        host.id = PANEL_HOST_ID;

        const shadow = host.attachShadow({
            mode: "open"
        });

        shadow.innerHTML = `
            <style>
                :host {
                    all: initial;
                }

                * {
                    box-sizing: border-box;
                }

                .panel {
                    position: fixed;
                    top: 0;
                    right: 0;
                    z-index: 2147483647;

                    width: min(
                        420px,
                        calc(100vw - 12px)
                    );

                    height: 100vh;
                    overflow-y: auto;

                    padding: 20px;

                    font-family:
                        Arial,
                        Helvetica,
                        sans-serif;

                    color: #172033;
                    background: #f4f7fb;

                    border-left:
                        1px solid #d5deea;

                    box-shadow:
                        -18px 0 45px
                        rgba(15, 23, 42, 0.20);

                    animation:
                        hg-panel-open
                        0.28s ease-out;
                }

                @keyframes hg-panel-open {
                    from {
                        transform:
                            translateX(100%);
                    }

                    to {
                        transform:
                            translateX(0);
                    }
                }

                .header {
                    display: flex;
                    align-items: flex-start;
                    justify-content: space-between;
                    gap: 12px;

                    margin-bottom: 17px;
                }

                .brand {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }

                .logo {
                    width: 42px;
                    height: 42px;

                    display: flex;
                    align-items: center;
                    justify-content: center;

                    flex-shrink: 0;

                    font-size: 14px;
                    font-weight: 800;

                    color: #ffffff;
                    background: #315fe9;

                    border-radius: 11px;
                }

                h2 {
                    margin: 0;

                    font-size: 17px;
                    line-height: 1.3;
                }

                .subtitle {
                    margin: 4px 0 0;

                    max-width: 275px;

                    font-size: 10px;
                    line-height: 1.4;

                    color: #748097;
                }

                .close {
                    width: 36px;
                    height: 36px;

                    display: flex;
                    align-items: center;
                    justify-content: center;

                    flex-shrink: 0;

                    padding: 0;

                    font-size: 23px;

                    color: #566379;
                    background: #e7ecf3;

                    border: none;
                    border-radius: 9px;

                    cursor: pointer;
                }

                .close:hover {
                    background: #dae1eb;
                }

                .card {
                    margin-bottom: 14px;
                    padding: 16px;

                    background: #ffffff;

                    border:
                        1px solid #d8e0eb;

                    border-radius: 13px;
                }

                .small-label {
                    display: block;

                    margin-bottom: 7px;

                    font-size: 10px;
                    font-weight: 800;
                    letter-spacing: 0.05em;
                    text-transform: uppercase;

                    color: #7d899d;
                }

                .article-title {
                    margin: 0;

                    font-size: 14px;
                    font-weight: 700;
                    line-height: 1.5;
                }

                .statistics {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px 14px;

                    margin-top: 10px;

                    font-size: 11px;
                    color: #778398;
                }

                .loading {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;

                    min-height: 220px;

                    text-align: center;
                }

                .spinner {
                    width: 38px;
                    height: 38px;

                    margin-bottom: 14px;

                    border:
                        4px solid #dce4f2;

                    border-top-color: #315fe9;
                    border-radius: 50%;

                    animation:
                        hg-spin 0.8s linear infinite;
                }

                @keyframes hg-spin {
                    to {
                        transform: rotate(360deg);
                    }
                }

                .loading p {
                    margin: 0;

                    font-size: 13px;
                    color: #647188;
                }

                .result,
                .error {
                    display: none;
                }

                .badge {
                    display: inline-block;

                    margin-bottom: 12px;
                    padding: 7px 11px;

                    font-size: 14px;
                    font-weight: 800;

                    border-radius: 9px;
                }

                .badge.real {
                    color: #087544;
                    background: #d9f6e7;
                }

                .badge.fake {
                    color: #a22132;
                    background: #ffe1e5;
                }

                .description {
                    margin: 0 0 15px;

                    font-size: 12px;
                    line-height: 1.55;

                    color: #536078;
                }

                .confidence {
                    display: flex;
                    justify-content: space-between;

                    margin-bottom: 16px;
                    padding: 12px;

                    background: #f1f4f9;
                    border-radius: 9px;
                }

                .confidence strong {
                    font-size: 17px;
                }

                .probability {
                    margin-top: 13px;
                }

                .probability-header {
                    display: flex;
                    justify-content: space-between;
                    gap: 12px;

                    margin-bottom: 6px;

                    font-size: 11px;
                    color: #5e6b81;
                }

                .track {
                    width: 100%;
                    height: 8px;

                    overflow: hidden;

                    background: #e6ebf2;
                    border-radius: 999px;
                }

                .bar {
                    width: 0;
                    height: 100%;

                    border-radius: 999px;

                    transition:
                        width 0.35s ease;
                }

                .real-bar {
                    background: #1ea766;
                }

                .fake-bar {
                    background: #df4657;
                }

                .analysis-info {
                    margin-top: 17px;
                    padding-top: 13px;

                    border-top:
                        1px solid #e3e8ef;
                }

                .info-row {
                    display: flex;
                    justify-content: space-between;
                    gap: 14px;

                    margin-top: 8px;

                    font-size: 11px;
                    color: #69768b;
                }

                .info-row:first-child {
                    margin-top: 0;
                }

                .complete {
                    margin-top: 12px;
                    padding: 9px 10px;

                    font-size: 11px;
                    font-weight: 700;
                    line-height: 1.45;

                    color: #087544;
                    background: #dcf7e9;

                    border-radius: 8px;
                }

                .warning {
                    margin: 15px 0 0;
                    padding: 11px;

                    font-size: 10px;
                    line-height: 1.5;

                    color: #6b778c;
                    background: #f5f7fa;

                    border-radius: 8px;
                }

                .error {
                    color: #8f1d2c;
                    background: #ffe7ea;

                    border:
                        1px solid #f3bdc4;
                }

                .error h3 {
                    margin: 0 0 7px;

                    font-size: 14px;
                }

                .error p {
                    margin: 0;

                    font-size: 12px;
                    line-height: 1.5;
                }

                @media (max-width: 520px) {
                    .panel {
                        width: 100vw;
                    }
                }
            </style>

            <aside
                class="panel"
                role="dialog"
                aria-label="Haber güvenilirlik analiz paneli"
            >
                <header class="header">
                    <div class="brand">
                        <div class="logo">
                            HG
                        </div>

                        <div>
                            <h2>
                                Haber Güvenilirlik Analizi
                            </h2>

                            <p class="subtitle"></p>
                        </div>
                    </div>

                    <button
                        type="button"
                        class="close"
                        aria-label="Paneli kapat"
                        title="Paneli kapat"
                    >
                        ×
                    </button>
                </header>

                <section class="card">
                    <span class="small-label">
                        Analiz edilen içerik
                    </span>

                    <p class="article-title"></p>

                    <div class="statistics">
                        <span class="paragraph-count"></span>
                        <span class="character-count"></span>
                    </div>
                </section>

                <section class="card loading">
                    <div class="spinner"></div>

                    <p>
                        Haber modeliyle analiz ediliyor...
                    </p>
                </section>

                <section class="card result">
                    <div class="badge"></div>

                    <p class="description"></p>

                    <div class="confidence">
                        <span>Model güveni</span>

                        <strong
                            class="confidence-value"
                        ></strong>
                    </div>

                    <div class="probability">
                        <div
                            class="probability-header"
                        >
                            <span>
                                Güvenilir görünme olasılığı
                            </span>

                            <strong
                                class="real-text"
                            ></strong>
                        </div>

                        <div class="track">
                            <div
                                class="bar real-bar"
                            ></div>
                        </div>
                    </div>

                    <div class="probability">
                        <div
                            class="probability-header"
                        >
                            <span>
                                Şüpheli görünme olasılığı
                            </span>

                            <strong
                                class="fake-text"
                            ></strong>
                        </div>

                        <div class="track">
                            <div
                                class="bar fake-bar"
                            ></div>
                        </div>
                    </div>

                    <div class="analysis-info">
                        <div class="info-row">
                            <span>Toplam token</span>
                            <strong
                                class="total-token-count"
                            ></strong>
                        </div>

                        <div class="info-row">
                            <span>Analiz edilen bölüm</span>
                            <strong
                                class="chunk-count"
                            ></strong>
                        </div>

                        <div
                            class="info-row overlap-row"
                        >
                            <span>Parça örtüşmesi</span>
                            <strong
                                class="overlap-count"
                            ></strong>
                        </div>

                        <div class="complete"></div>
                    </div>

                    <p class="warning">
                        Bu sonuç otomatik bir model
                        değerlendirmesidir. Kesin doğruluk
                        kontrolü yerine yardımcı bir uyarı
                        olarak kullanılmalıdır.
                    </p>
                </section>

                <section class="card error">
                    <h3>
                        Analiz yapılamadı
                    </h3>

                    <p
                        class="error-message"
                    ></p>
                </section>
            </aside>
        `;

        shadow.querySelector(
            ".subtitle"
        ).textContent = (
            backendView.ready
                ? backendView.text
                : "Model bağlantısı bulunamadı"
        );

        shadow.querySelector(
            ".article-title"
        ).textContent = article.title;

        shadow.querySelector(
            ".paragraph-count"
        ).textContent = (
            `${formatNumber(
                article.paragraphCount
            )} paragraf`
        );

        shadow.querySelector(
            ".character-count"
        ).textContent = (
            `${formatNumber(
                article.characterCount
            )} karakter`
        );

        shadow.querySelector(
            ".close"
        ).addEventListener(
            "click",
            () => {
                host.remove();
            }
        );

        document.documentElement.appendChild(
            host
        );

        return {
            host,
            shadow
        };
    }


    // ========================================================
    // SONUCU PANELE YAZ
    // ========================================================

    function renderPanelResult(
        panel,
        prediction
    ) {
        const { shadow } = panel;

        shadow.querySelector(
            ".loading"
        ).style.display = "none";

        shadow.querySelector(
            ".error"
        ).style.display = "none";

        shadow.querySelector(
            ".result"
        ).style.display = "block";

        const isReal =
            prediction.label === "real";

        const realProbability = Number(
            prediction.probabilities?.real || 0
        );

        const fakeProbability = Number(
            prediction.probabilities?.fake || 0
        );

        const badge =
            shadow.querySelector(".badge");

        badge.classList.remove(
            "real",
            "fake"
        );

        badge.classList.add(
            isReal ? "real" : "fake"
        );

        badge.textContent = (
            isReal
                ? "Güvenilir Görünüyor"
                : "Şüpheli Görünüyor"
        );

        shadow.querySelector(
            ".description"
        ).textContent = (
            isReal
                ? (
                    "Model, haber metninin eğitim "
                    + "verisindeki güvenilir haber "
                    + "örüntülerine daha yakın olduğunu "
                    + "değerlendirdi."
                )
                : (
                    "Model, haber metninde sahte veya "
                    + "değiştirilmiş haber örüntülerine "
                    + "benzer özellikler tespit etti."
                )
        );

        shadow.querySelector(
            ".confidence-value"
        ).textContent = formatPercentage(
            prediction.confidence
        );

        shadow.querySelector(
            ".real-text"
        ).textContent = formatPercentage(
            realProbability
        );

        shadow.querySelector(
            ".fake-text"
        ).textContent = formatPercentage(
            fakeProbability
        );

        shadow.querySelector(
            ".real-bar"
        ).style.width = (
            `${clampPercentage(
                realProbability
            )}%`
        );

        shadow.querySelector(
            ".fake-bar"
        ).style.width = (
            `${clampPercentage(
                fakeProbability
            )}%`
        );

        const totalTokenCount = Number(
            prediction.total_token_count
            ?? prediction.processed_token_count
            ?? 0
        );

        const chunkCount = Number(
            prediction.chunk_count ?? 1
        );

        const isChunked = (
            prediction.is_chunked === true
            || chunkCount > 1
        );

        const overlapTokens = Number(
            prediction.chunk_overlap_tokens ?? 0
        );

        shadow.querySelector(
            ".total-token-count"
        ).textContent = formatNumber(
            totalTokenCount
        );

        shadow.querySelector(
            ".chunk-count"
        ).textContent = (
            isChunked
                ? `${formatNumber(chunkCount)} bölüm`
                : "Tek parça"
        );

        const overlapRow =
            shadow.querySelector(
                ".overlap-row"
            );

        if (isChunked) {
            overlapRow.style.display = "flex";

            shadow.querySelector(
                ".overlap-count"
            ).textContent = (
                `${formatNumber(
                    overlapTokens
                )} token`
            );

        } else {
            overlapRow.style.display = "none";
        }

        const completeElement =
            shadow.querySelector(
                ".complete"
            );

        if (
            prediction.all_text_analyzed
            !== false
        ) {
            completeElement.textContent = (
                isChunked
                    ? (
                        "Haberin tamamı "
                        + `${formatNumber(chunkCount)} `
                        + "bölümde analiz edildi."
                    )
                    : (
                        "Haber metni tek parça "
                        + "olarak analiz edildi."
                    )
            );

        } else {
            completeElement.textContent = (
                "Haber metninin tamamı "
                + "analiz edilemedi."
            );
        }
    }


    // ========================================================
    // HATAYI PANELE YAZ
    // ========================================================

    function renderPanelError(
        panel,
        message
    ) {
        const { shadow } = panel;

        shadow.querySelector(
            ".loading"
        ).style.display = "none";

        shadow.querySelector(
            ".result"
        ).style.display = "none";

        const errorSection =
            shadow.querySelector(".error");

        errorSection.style.display = "block";

        shadow.querySelector(
            ".error-message"
        ).textContent = (
            message
            || "Bilinmeyen bir analiz hatası oluştu."
        );
    }


    // ========================================================
    // ANALİZİ BAŞLAT
    // ========================================================

    async function startAnalysis(article) {
        if (state.analysisInProgress) {
            return;
        }

        state.analysisInProgress = true;

        removeElementById(
            PROMPT_HOST_ID
        );

        const panel =
            createSidePanel(article);

        try {
            const response =
                await sendExtensionMessage({
                    type: "ANALYZE_ARTICLE",
                    text: article.text
                });

            if (!response?.ok) {
                throw new Error(
                    response?.error
                    || "Haber analizi başarısız oldu."
                );
            }

            renderPanelResult(
                panel,
                response.data
            );

        } catch (error) {
            renderPanelError(
                panel,

                error instanceof Error
                    ? error.message
                    : (
                        "Analiz sırasında bilinmeyen "
                        + "bir hata oluştu."
                    )
            );

        } finally {
            state.analysisInProgress = false;
        }
    }


    // ========================================================
    // OTOMATİK HABER ALGILAMA
    // ========================================================

    async function tryAutomaticDetection() {
        if (
            document.getElementById(
                PROMPT_HOST_ID
            )
            || document.getElementById(
                PANEL_HOST_ID
            )
        ) {
            return true;
        }

        if (state.promptDismissed) {
            return true;
        }

        const article = extractArticle();

        if (!article.ok) {
            return false;
        }

        const backendStatus =
            await getBackendHealth();

        const backendView = getBackendView(
            backendStatus,
            state.backendError
        );

        if (!backendView.ready) {
            return false;
        }

        state.detectedArticle = article;

        showAnalysisPrompt(
            article,
            backendStatus,
            null
        );

        return true;
    }


    async function initializeAutomaticDetection() {
        for (
            const delay
            of AUTO_DETECTION_DELAYS
        ) {
            await wait(delay);

            const detected =
                await tryAutomaticDetection();

            if (detected) {
                return;
            }
        }
    }


    // ========================================================
    // TEK SAYFA UYGULAMALARINDA URL DEĞİŞİKLİĞİ
    // ========================================================

    function resetForNewUrl() {
        state.currentUrl =
            window.location.href;

        state.detectedArticle = null;
        state.promptDismissed = false;
        state.analysisInProgress = false;

        removeElementById(
            PROMPT_HOST_ID
        );

        removeElementById(
            PANEL_HOST_ID
        );

        removeElementById(
            NOTICE_HOST_ID
        );

        initializeAutomaticDetection();
    }


    setInterval(
        () => {
            if (
                window.location.href
                !== state.currentUrl
            ) {
                resetForNewUrl();
            }
        },
        1500
    );


    // ========================================================
    // EKLENTİ SİMGESİNDEN MANUEL ÇALIŞTIRMA
    // ========================================================

    chrome.runtime.onMessage.addListener(
        (
            message,
            sender,
            sendResponse
        ) => {
            if (
                message?.type
                !== "SHOW_MANUAL_ANALYSIS_PROMPT"
            ) {
                return;
            }

            state.promptDismissed = false;

            state.backendStatus = (
                message?.backendStatus
                || null
            );

            state.backendError = (
                message?.backendError
                || null
            );

            const article = extractArticle();

            if (!article.ok) {
                showNotice(
                    article.reason
                    || (
                        "Bu sayfada yeterli haber "
                        + "içeriği bulunamadı."
                    )
                );

                sendResponse({
                    ok: false,

                    error: (
                        article.reason
                        || "Haber içeriği bulunamadı."
                    )
                });

                return;
            }

            showAnalysisPrompt(
                article,
                state.backendStatus,
                state.backendError
            );

            sendResponse({
                ok: true
            });
        }
    );


    // ========================================================
    // BAŞLAT
    // ========================================================

    initializeAutomaticDetection();
})();