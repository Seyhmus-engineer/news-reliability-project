# ============================================================
# NEWS RELIABILITY FASTAPI APPLICATION
# ============================================================

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import (
    CORSMiddleware,
)

from backend.app.api.routes import router
from backend.app.services.model_service import (
    model_service,
)


# ============================================================
# LOG AYARLARI
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ============================================================
# UYGULAMA YAŞAM DÖNGÜSÜ
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """
    FastAPI açılırken modeli yalnızca bir kez yükler.
    """

    logger.info(
        "News Reliability API başlatılıyor."
    )

    try:
        model_service.load_model()

    except Exception:
        logger.exception(
            "Final ELECTRA modeli yüklenemedi."
        )

        # Model olmadan API'nin başlamasına izin verme.
        raise

    logger.info(
        "Model kullanıma hazır. API başlatıldı."
    )

    yield

    logger.info(
        "News Reliability API kapatılıyor."
    )


# ============================================================
# FASTAPI UYGULAMASI
# ============================================================

app = FastAPI(
    title="News Reliability API",
    description=(
        "Turkish ELECTRA Base kullanarak "
        "Türkçe haberlerin güvenilirliğini analiz eder."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    # Geliştirme aşamasında Chrome eklentisinin
    # yerel API'ye erişebilmesi için açık bırakıldı.
    allow_origins=["*"],

    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=["*"],
)


# ============================================================
# ROUTER
# ============================================================

app.include_router(
    router
)


# ============================================================
# PYCHARM RUN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",

        host="127.0.0.1",
        port=8000,

        # GPU modelinin iki kez yüklenmemesi için kapalı.
        reload=False,

        log_level="info",
    )