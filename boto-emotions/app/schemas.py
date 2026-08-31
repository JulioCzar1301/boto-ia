from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EmotionFrameRequest(BaseModel):
    image: str = Field(..., description="Imagem em base64, com ou sem prefixo data:image.")
    session_id: str | None = Field(
        default=None,
        description="Sessao opcional para manter historico temporal entre frames.",
    )
    include_debug: bool = Field(
        default=False,
        description="Inclui dados internos de debug na resposta.",
    )
    use_temporal_as_final: bool = Field(
        default=False,
        description="Usa a BiLSTM temporal como decisao final quando estiver pronta.",
    )


class SessionCreateRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        description="Identificador opcional da sessao. Se omitido, a API gera um UUID.",
    )
    use_temporal_as_final: bool = Field(
        default=False,
        description="Usa a BiLSTM temporal como decisao final quando estiver pronta.",
    )


class SessionResponse(BaseModel):
    session_id: str
    frame_count: int
    created_at: float
    updated_at: float


class EmotionResponse(BaseModel):
    detected: bool
    emotion: str | None
    confidence: float
    scores: dict[str, float]
    rule_based_emotion: str | None
    rule_based_confidence: float
    temporal: dict[str, Any]
    quality: dict[str, Any]
    bbox: list[int] | None
    face_count: int
    session_id: str | None = None
    debug: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: bool
    active_sessions: int
    privacy_mode: str


class PrivacyResponse(BaseModel):
    image_storage_enabled: bool
    request_payload_logging_enabled: bool
    session_ttl_seconds: int
    max_image_base64_chars: int
    retained_session_data: list[str]
    discarded_after_inference: list[str]
