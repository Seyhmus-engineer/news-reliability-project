# ============================================================
# FASTAPI ROUTES
# ============================================================

import logging

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from backend.app.schemas.prediction import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

from backend.app.services.model_service import (
    model_service,
)


logger = logging.getLogger(__name__)


router = APIRouter()


# ============================================================
# ANA ENDPOINT
# ============================================================

@router.get("/")
def root() -> dict[str, str]:
    return {
        "application": "News Reliability API",
        "status": "running",
        "model": "Turkish ELECTRA Base",
        "documentation": "/docs",
    }


# ============================================================
# SAĞLIK KONTROLÜ
# ============================================================

@router.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    model_status = model_service.get_status()

    application_status = (
        "healthy"
        if model_service.is_loaded
        else "degraded"
    )

    return HealthResponse(
        status=application_status,
        model=model_status,
    )


# ============================================================
# HABER TAHMİNİ
# ============================================================

@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:

    if not model_service.is_loaded:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Tahmin modeli henüz kullanıma hazır değil."
            ),
        )

    try:
        prediction_result = (
            model_service.predict(
                request.text
            )
        )

        return PredictionResponse(
            **prediction_result
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=str(error),
        ) from error

    except RuntimeError as error:
        logger.exception(
            "Model tahmini sırasında çalışma hatası."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Beklenmeyen tahmin hatası."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Tahmin sırasında beklenmeyen "
                "bir sunucu hatası oluştu."
            ),
        ) from error