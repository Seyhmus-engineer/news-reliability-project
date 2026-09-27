# ============================================================
# TAHMİN İSTEK VE CEVAP ŞEMALARI
# ============================================================

from typing import Literal

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# İSTEK
# ============================================================

class PredictionRequest(BaseModel):
    """
    /predict endpoint'ine gönderilecek haber metni.
    """

    text: str = Field(
        ...,
        min_length=20,
        max_length=100_000,
        description=(
            "Sınıflandırılacak Türkçe haber metni."
        ),
    )


# ============================================================
# OLASILIKLAR
# ============================================================

class ProbabilityResponse(BaseModel):
    real: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    fake: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )


# ============================================================
# PARÇA SONUCU
# ============================================================

class ChunkPredictionResponse(BaseModel):
    chunk_index: int = Field(
        ...,
        ge=1,
    )

    token_count: int = Field(
        ...,
        ge=1,
        le=512,
    )

    effective_token_count: int = Field(
        ...,
        ge=1,
    )

    label: Literal[
        "real",
        "fake",
    ]

    label_id: int = Field(
        ...,
        ge=0,
        le=1,
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    probabilities: ProbabilityResponse


# ============================================================
# GENEL TAHMİN SONUCU
# ============================================================

class PredictionResponse(BaseModel):
    label: Literal[
        "real",
        "fake",
    ]

    label_id: int = Field(
        ...,
        ge=0,
        le=1,
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    probabilities: ProbabilityResponse

    # Önceki Chrome eklentisiyle uyumluluk için korunur.
    processed_token_count: int = Field(
        ...,
        ge=1,
    )

    maximum_token_length: int = Field(
        ...,
        ge=1,
    )

    total_token_count: int = Field(
        ...,
        ge=1,
    )

    chunk_count: int = Field(
        ...,
        ge=1,
    )

    is_chunked: bool

    all_text_analyzed: bool

    chunk_overlap_tokens: int = Field(
        ...,
        ge=0,
    )

    chunks: list[
        ChunkPredictionResponse
    ]


# ============================================================
# MODEL DURUMU
# ============================================================

class ModelStatusResponse(BaseModel):
    loaded: bool
    model_name: str
    model_path: str
    device: str
    gpu: str | None = None

    labels: dict[
        str,
        str,
    ]

    minimum_token_length: int = Field(
        ...,
        ge=1,
    )

    maximum_token_length: int = Field(
        ...,
        ge=1,
    )

    chunking_enabled: bool

    chunk_overlap_tokens: int = Field(
        ...,
        ge=0,
    )


# ============================================================
# HEALTH CEVABI
# ============================================================

class HealthResponse(BaseModel):
    status: Literal[
        "healthy",
        "degraded",
    ]

    model: ModelStatusResponse