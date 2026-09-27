"use strict";


/*
 * Haber sitelerine özel içerik çıkarma kuralları.
 *
 * Bir seçici bulunamazsa content.js içerisindeki
 * genel haber çıkarma sistemi çalışmaya devam eder.
 */

globalThis.HG_SITE_RULES = [
    {
        name: "TRT Haber",

        hostnames: [
            "trthaber.com",
            "www.trthaber.com"
        ],

        titleSelectors: [
            "article h1",
            ".news-title",
            ".detail-title",
            ".article-title",
            "main h1",
            "h1"
        ],

        descriptionSelectors: [
            ".news-spot",
            ".article-spot",
            ".detail-spot",
            ".summary",
            ".lead",
            "article h2"
        ],

        bodySelectors: [
            "[itemprop='articleBody']",
            ".news-content",
            ".news-detail",
            ".article-content",
            ".article-body",
            ".detail-content",
            ".detail-text",
            "article"
        ],

        excludedSelectors: [
            ".related-news",
            ".latest-news",
            ".most-read",
            ".news-gallery",
            ".social-share",
            ".advertisement",
            ".ad-container",
            ".author-box",
            ".tags"
        ]
    },

    {
        name: "Anadolu Ajansı",

        hostnames: [
            "aa.com.tr",
            "www.aa.com.tr"
        ],

        titleSelectors: [
            "article h1",
            ".detay-baslik",
            ".detail-title",
            ".news-title",
            ".article-title",
            "main h1",
            "h1"
        ],

        descriptionSelectors: [
            ".detay-spot",
            ".news-spot",
            ".article-spot",
            ".summary",
            ".lead",
            "article h2"
        ],

        bodySelectors: [
            "[itemprop='articleBody']",
            ".detay-icerik",
            ".detay-icerik-container",
            ".news-content",
            ".article-content",
            ".article-body",
            ".detail-content",
            "article"
        ],

        excludedSelectors: [
            ".related-news",
            ".related-topics",
            ".latest-news",
            ".most-read",
            ".social-share",
            ".share-area",
            ".advertisement",
            ".ad-container",
            ".subscription",
            ".whatsapp-channel",
            ".tags"
        ]
    },

    {
        name: "Sözcü",

        hostnames: [
            "sozcu.com.tr",
            "www.sozcu.com.tr"
        ],

        titleSelectors: [
            "article h1",
            ".news-title",
            ".article-title",
            ".detail-title",
            "main h1",
            "h1"
        ],

        descriptionSelectors: [
            ".news-spot",
            ".article-spot",
            ".spot",
            ".summary",
            ".lead",
            "article h2"
        ],

        bodySelectors: [
            "[itemprop='articleBody']",
            ".article-body",
            ".news-body",
            ".news-content",
            ".article-content",
            ".content-text",
            ".detail-content",
            "article"
        ],

        excludedSelectors: [
            ".related-news",
            ".suggested-news",
            ".latest-news",
            ".most-read",
            ".social-share",
            ".share-buttons",
            ".advertisement",
            ".ad-container",
            ".author-info",
            ".tags"
        ]
    },

    {
        name: "Cumhuriyet",

        hostnames: [
            "cumhuriyet.com.tr",
            "www.cumhuriyet.com.tr"
        ],

        titleSelectors: [
            "article h1",
            ".haber-baslik",
            ".news-title",
            ".article-title",
            ".detail-title",
            "main h1",
            "h1"
        ],

        descriptionSelectors: [
            ".haber-spot",
            ".news-spot",
            ".article-spot",
            ".spot",
            ".summary",
            ".lead",
            "article h2"
        ],

        bodySelectors: [
            "[itemprop='articleBody']",
            ".haberMetni",
            ".haber-metni",
            ".news-content",
            ".article-content",
            ".article-body",
            ".detail-content",
            "article"
        ],

        excludedSelectors: [
            ".related-news",
            ".suggested-news",
            ".latest-news",
            ".most-read",
            ".social-share",
            ".share-buttons",
            ".advertisement",
            ".ad-container",
            ".author-box",
            ".tags"
        ]
    },

    {
        name: "Hürriyet",

        hostnames: [
            "hurriyet.com.tr",
            "www.hurriyet.com.tr"
        ],

        titleSelectors: [
            "article h1",
            ".news-title",
            ".article-title",
            ".detail-title",
            "main h1",
            "h1"
        ],

        descriptionSelectors: [
            ".news-spot",
            ".article-spot",
            ".spot",
            ".summary",
            ".lead",
            "article h2"
        ],

        bodySelectors: [
            "[itemprop='articleBody']",
            ".news-content",
            ".article-content",
            ".article-body",
            ".detail-content",
            ".content-body",
            "article"
        ],

        excludedSelectors: [
            ".related-news",
            ".latest-news",
            ".most-read",
            ".social-share",
            ".advertisement",
            ".ad-container",
            ".author-box",
            ".tags"
        ]
    },

    {
        name: "Milliyet",

        hostnames: [
            "milliyet.com.tr",
            "www.milliyet.com.tr"
        ],

        titleSelectors: [
            "article h1",
            ".news-title",
            ".article-title",
            ".detail-title",
            "main h1",
            "h1"
        ],

        descriptionSelectors: [
            ".news-spot",
            ".article-spot",
            ".spot",
            ".summary",
            ".lead",
            "article h2"
        ],

        bodySelectors: [
            "[itemprop='articleBody']",
            ".news-content",
            ".article-content",
            ".article-body",
            ".detail-content",
            ".content-body",
            "article"
        ],

        excludedSelectors: [
            ".related-news",
            ".latest-news",
            ".most-read",
            ".social-share",
            ".advertisement",
            ".ad-container",
            ".author-box",
            ".tags"
        ]
    }
];