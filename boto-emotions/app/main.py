from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from app import settings
from app.schemas import (
    EmotionFrameRequest,
    EmotionResponse,
    HealthResponse,
    PrivacyResponse,
    SessionCreateRequest,
    SessionResponse,
)
from app.services.emotion import EmotionSession, emotion_service
from app.utils.image import b64_to_bgr
from config import DEVICE


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description="API FastAPI para deteccao de emocoes com ENet, MediaPipe e BiLSTM temporal.",
    version=settings.APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials="*" not in settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def session_to_response(session: EmotionSession) -> SessionResponse:
    return SessionResponse(
        session_id=session.session_id,
        frame_count=session.frame_count,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@app.on_event("startup")
def startup_event() -> None:
    emotion_service.load_models()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=DEVICE,
        models_loaded=emotion_service.loaded,
        active_sessions=emotion_service.active_sessions,
        privacy_mode="in_memory_inference_only",
    )


@app.get("/privacy", response_model=PrivacyResponse)
def privacy() -> PrivacyResponse:
    return PrivacyResponse(
        image_storage_enabled=settings.STORE_IMAGES,
        request_payload_logging_enabled=settings.LOG_REQUEST_PAYLOADS,
        session_ttl_seconds=settings.SESSION_TTL_SECONDS,
        max_image_base64_chars=settings.MAX_IMAGE_BASE64_CHARS,
        retained_session_data=[
            "estado temporal numerico",
            "scores suavizados",
            "historico curto de features",
            "buffer temporal da BiLSTM",
            "contagem e timestamps da sessao",
        ],
        discarded_after_inference=[
            "imagem recebida",
            "payload base64",
            "frame OpenCV decodificado",
        ],
    )


@app.post("/emotion/frame", response_model=EmotionResponse)
def detect_emotion(body: EmotionFrameRequest) -> dict:
    try:
        frame = b64_to_bgr(body.image)
        return emotion_service.process_frame(
            frame_bgr=frame,
            session_id=body.session_id,
            include_debug=body.include_debug,
            use_temporal_as_final=body.use_temporal_as_final,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        log.exception("Falha ao processar frame de emocao: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Falha interna ao processar frame.",
        ) from exc


@app.post("/emotion/session/start", response_model=SessionResponse)
def start_session(body: SessionCreateRequest = SessionCreateRequest()) -> SessionResponse:
    session = emotion_service.create_session(
        session_id=body.session_id,
        use_temporal_as_final=body.use_temporal_as_final,
    )
    return session_to_response(session)


@app.post("/emotion/session/{session_id}/frame", response_model=EmotionResponse)
def detect_emotion_in_session(session_id: str, body: EmotionFrameRequest) -> dict:
    try:
        frame = b64_to_bgr(body.image)
        return emotion_service.process_frame(
            frame_bgr=frame,
            session_id=session_id,
            include_debug=body.include_debug,
            use_temporal_as_final=body.use_temporal_as_final,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        log.exception("Falha ao processar frame da sessao %s: %s", session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Falha interna ao processar frame.",
        ) from exc


@app.post("/emotion/session/{session_id}/reset", response_model=SessionResponse)
def reset_session(
    session_id: str,
    body: SessionCreateRequest = SessionCreateRequest(),
) -> SessionResponse:
    session = emotion_service.reset_session(
        session_id=session_id,
        use_temporal_as_final=body.use_temporal_as_final,
    )
    return session_to_response(session)


@app.get("/emotion/session/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    session = emotion_service.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sessao nao encontrada.",
        )

    return session_to_response(session)


@app.delete("/emotion/session/{session_id}")
def delete_session(session_id: str) -> dict[str, bool]:
    removed = emotion_service.delete_session(session_id)

    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sessao nao encontrada.",
        )

    return {"removed": True}
