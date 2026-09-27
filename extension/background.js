"use strict";


// ============================================================
// BACKEND AYARLARI
// ============================================================

const API_BASE_URL = "http://127.0.0.1:8000";


// ============================================================
// EKLENTİ ROZETİ AYARLARI
// ============================================================

const BADGE_SETTINGS = {
    gpu: {
        text: "GPU",
        color: "#159957",
        title: "Model hazır — GPU kullanılıyor"
    },

    cpu: {
        text: "CPU",
        color: "#d98c10",
        title: "Model hazır — CPU kullanılıyor"
    },

    offline: {
        text: "OFF",
        color: "#cf3344",
        title: "Backend veya model bağlantısı yok"
    },

    checking: {
        text: "...",
        color: "#65758b",
        title: "Backend kontrol ediliyor"
    }
};


// ============================================================
// API HATA MESAJINI ÇIKAR
// ============================================================

function extractErrorMessage(
    payload,
    statusCode
) {
    if (typeof payload?.detail === "string") {
        return payload.detail;
    }

    if (Array.isArray(payload?.detail)) {
        return payload.detail
            .map(item => {
                return item?.msg || String(item);
            })
            .join(" ");
    }

    return (
        "Backend isteği başarısız oldu. "
        + `HTTP durum kodu: ${statusCode}`
    );
}


// ============================================================
// ZAMAN AŞIMLI JSON İSTEĞİ
// ============================================================

async function fetchJson(
    url,
    options = {},
    timeoutMilliseconds = 60000
) {
    const controller = new AbortController();

    const timeoutId = setTimeout(
        () => {
            controller.abort();
        },
        timeoutMilliseconds
    );

    try {
        const response = await fetch(
            url,
            {
                ...options,
                signal: controller.signal
            }
        );

        let payload = null;

        try {
            payload = await response.json();

        } catch (error) {
            payload = null;
        }

        if (!response.ok) {
            throw new Error(
                extractErrorMessage(
                    payload,
                    response.status
                )
            );
        }

        return payload;

    } catch (error) {
        if (error?.name === "AbortError") {
            throw new Error(
                "Backend isteği zaman aşımına uğradı."
            );
        }

        throw error;

    } finally {
        clearTimeout(timeoutId);
    }
}


// ============================================================
// ROZET HEDEFİ
// ============================================================

function createActionTarget(tabId) {
    if (Number.isInteger(tabId)) {
        return {
            tabId
        };
    }

    return {};
}


// ============================================================
// ROZETİ GÜNCELLE
// ============================================================

async function setActionBadge(
    badgeType,
    tabId = null,
    customTitle = null
) {
    const settings = (
        BADGE_SETTINGS[badgeType]
        || BADGE_SETTINGS.offline
    );

    const target = createActionTarget(
        tabId
    );

    try {
        await Promise.all([
            chrome.action.setBadgeText({
                ...target,
                text: settings.text
            }),

            chrome.action.setBadgeBackgroundColor({
                ...target,
                color: settings.color
            }),

            chrome.action.setTitle({
                ...target,

                title: (
                    customTitle
                    || settings.title
                )
            })
        ]);

    } catch (error) {
        console.warn(
            "Eklenti rozeti güncellenemedi:",
            error
        );
    }
}


// ============================================================
// HEALTH SONUCUNU ROZETE YANSIT
// ============================================================

async function updateBadgeFromHealth(
    health,
    tabId = null
) {
    const model = health?.model;

    const modelReady = (
        health?.status === "healthy"
        && model?.loaded === true
    );

    if (!modelReady) {
        await setActionBadge(
            "offline",
            tabId,
            "Backend açık fakat model hazır değil"
        );

        return;
    }

    const device = String(
        model?.device || ""
    ).toLocaleLowerCase("tr-TR");

    if (
        device === "cuda"
        || device.startsWith("cuda:")
    ) {
        const gpuName = (
            model?.gpu
            || "CUDA GPU"
        );

        await setActionBadge(
            "gpu",
            tabId,
            `Model hazır — ${gpuName}`
        );

        return;
    }

    await setActionBadge(
        "cpu",
        tabId,
        "Model hazır — CPU kullanılıyor"
    );
}


// ============================================================
// BACKEND SAĞLIK KONTROLÜ
// ============================================================

async function checkBackend(tabId = null) {
    await setActionBadge(
        "checking",
        tabId
    );

    try {
        const health = await fetchJson(
            `${API_BASE_URL}/health`,
            {
                method: "GET",

                headers: {
                    "Accept": "application/json"
                }
            },
            5000
        );

        await updateBadgeFromHealth(
            health,
            tabId
        );

        return health;

    } catch (error) {
        await setActionBadge(
            "offline",
            tabId,
            "Backend bağlantısı kurulamadı"
        );

        throw error;
    }
}


// ============================================================
// HABERİ ANALİZ ET
// ============================================================

async function analyzeArticle(text) {
    if (typeof text !== "string") {
        throw new Error(
            "Analiz edilecek haber metni geçersiz."
        );
    }

    const normalizedText = text
        .replace(/\s+/g, " ")
        .trim();

    if (!normalizedText) {
        throw new Error(
            "Analiz edilecek haber metni boş."
        );
    }

    if (normalizedText.length > 100000) {
        throw new Error(
            "Haber metni izin verilen karakter "
            + "sınırını aşıyor."
        );
    }

    return fetchJson(
        `${API_BASE_URL}/predict`,
        {
            method: "POST",

            headers: {
                "Accept": "application/json",
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: normalizedText
            })
        },
        120000
    );
}


// ============================================================
// CONTENT SCRIPT MESAJLARI
// ============================================================

chrome.runtime.onMessage.addListener(
    (
        message,
        sender,
        sendResponse
    ) => {
        if (
            sender.id
            && sender.id !== chrome.runtime.id
        ) {
            sendResponse({
                ok: false,
                error: "Yetkisiz eklenti mesajı."
            });

            return false;
        }

        const tabId = sender?.tab?.id;

        const handleMessage = async () => {
            switch (message?.type) {
                case "CHECK_BACKEND": {
                    const health = await checkBackend(
                        tabId
                    );

                    return {
                        ok: true,
                        data: health
                    };
                }

                case "ANALYZE_ARTICLE": {
                    const prediction = await analyzeArticle(
                        message?.text
                    );

                    return {
                        ok: true,
                        data: prediction
                    };
                }

                default: {
                    return {
                        ok: false,
                        error: "Bilinmeyen mesaj türü."
                    };
                }
            }
        };

        handleMessage()
            .then(sendResponse)
            .catch(error => {
                sendResponse({
                    ok: false,

                    error: (
                        error instanceof Error
                            ? error.message
                            : "Beklenmeyen backend hatası."
                    )
                });
            });

        // Asenkron sendResponse için bağlantıyı açık tutar.
        return true;
    }
);


// ============================================================
// ARAÇ ÇUBUĞU SİMGESİNE TIKLAMA
// ============================================================

chrome.action.onClicked.addListener(
    async tab => {
        if (typeof tab?.id !== "number") {
            return;
        }

        let backendStatus = null;
        let backendError = null;

        try {
            backendStatus = await checkBackend(
                tab.id
            );

        } catch (error) {
            backendError = (
                error instanceof Error
                    ? error.message
                    : "Backend bağlantısı kurulamadı."
            );
        }

        try {
            await chrome.tabs.sendMessage(
                tab.id,
                {
                    type: "SHOW_MANUAL_ANALYSIS_PROMPT",
                    backendStatus,
                    backendError
                }
            );

        } catch (error) {
            console.warn(
                "Bu sekmede haber analizi başlatılamadı:",
                error
            );
        }
    }
);


// ============================================================
// EKLENTİ YÜKLENDİĞİNDE BACKEND KONTROLÜ
// ============================================================

chrome.runtime.onInstalled.addListener(
    () => {
        checkBackend().catch(() => {
            // OFF rozeti checkBackend tarafından ayarlanır.
        });
    }
);


// ============================================================
// CHROME BAŞLADIĞINDA BACKEND KONTROLÜ
// ============================================================

chrome.runtime.onStartup.addListener(
    () => {
        checkBackend().catch(() => {
            // OFF rozeti checkBackend tarafından ayarlanır.
        });
    }
);