from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.settings import SESSION_TTL_SECONDS
from config import (
    DEVICE,
    ENET_MODEL_PATH,
    TEMPORAL_MODEL_PATH,
    USE_TEMPORAL_MODEL,
)
from models import ENetEmotionPerceiver, create_face_landmarker
from processor import EmotionFrameProcessor
from state import TemporalEmotionState
from temporal_model import TemporalEmotionPerceiver


log = logging.getLogger(__name__)


@dataclass
class EmotionSession:
    session_id: str
    processor: EmotionFrameProcessor
    created_at: float
    updated_at: float
    frame_count: int = 0


class EmotionService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._landmarker = None
        self._enet = None
        self._temporal_perceiver = None
        self._sessions: dict[str, EmotionSession] = {}
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def active_sessions(self) -> int:
        with self._lock:
            self.cleanup_expired_sessions()
            return len(self._sessions)

    def cleanup_expired_sessions(self) -> int:
        with self._lock:
            now = time.time()
            expired_ids = [
                session_id
                for session_id, session in self._sessions.items()
                if now - session.updated_at > SESSION_TTL_SECONDS
            ]

            for session_id in expired_ids:
                self._sessions.pop(session_id, None)

            if expired_ids:
                log.info("Sessoes expiradas removidas: %s", len(expired_ids))

            return len(expired_ids)

    def load_models(self) -> None:
        with self._lock:
            if self._loaded:
                return

            log.info("Carregando FaceLandmarker...")
            self._landmarker = create_face_landmarker()

            log.info("Carregando ENet: %s", ENET_MODEL_PATH)
            self._enet = ENetEmotionPerceiver(ENET_MODEL_PATH, DEVICE)

            if USE_TEMPORAL_MODEL:
                log.info("Carregando modelo temporal: %s", TEMPORAL_MODEL_PATH)
                self._temporal_perceiver = TemporalEmotionPerceiver(TEMPORAL_MODEL_PATH, DEVICE)
            else:
                log.info("Modelo temporal desativado por configuracao.")

            self._loaded = True
            log.info("Modelos de emocao carregados no dispositivo: %s", DEVICE)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_models()

    def _build_processor(self, use_temporal_as_final: bool = False) -> EmotionFrameProcessor:
        self._ensure_loaded()

        return EmotionFrameProcessor(
            landmarker=self._landmarker,
            enet=self._enet,
            temporal_state=TemporalEmotionState(),
            temporal_perceiver=self._temporal_perceiver,
            use_temporal_as_final=use_temporal_as_final,
            draw_debug_overlay=False,
            render_overlay=False,
        )

    def create_session(
        self,
        session_id: str | None = None,
        use_temporal_as_final: bool = False,
    ) -> EmotionSession:
        with self._lock:
            self.cleanup_expired_sessions()
            sid = session_id or str(uuid.uuid4())

            if sid in self._sessions:
                return self._sessions[sid]

            now = time.time()
            session = EmotionSession(
                session_id=sid,
                processor=self._build_processor(use_temporal_as_final=use_temporal_as_final),
                created_at=now,
                updated_at=now,
            )
            self._sessions[sid] = session
            log.info("Sessao de emocao criada: %s", sid)
            return session

    def get_session(self, session_id: str) -> EmotionSession | None:
        self.cleanup_expired_sessions()
        return self._sessions.get(session_id)

    def reset_session(
        self,
        session_id: str,
        use_temporal_as_final: bool = False,
    ) -> EmotionSession:
        with self._lock:
            self.cleanup_expired_sessions()
            now = time.time()
            session = EmotionSession(
                session_id=session_id,
                processor=self._build_processor(use_temporal_as_final=use_temporal_as_final),
                created_at=now,
                updated_at=now,
            )
            self._sessions[session_id] = session
            log.info("Sessao de emocao reiniciada: %s", session_id)
            return session

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            removed = self._sessions.pop(session_id, None) is not None
            if removed:
                log.info("Sessao de emocao removida: %s", session_id)
            return removed

    def process_frame(
        self,
        frame_bgr,
        session_id: str | None = None,
        include_debug: bool = False,
        use_temporal_as_final: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            self.cleanup_expired_sessions()
            if session_id:
                session = self._sessions.get(session_id)
                if session is None:
                    session = self.create_session(
                        session_id=session_id,
                        use_temporal_as_final=use_temporal_as_final,
                    )
                processor = session.processor
            else:
                session = None
                processor = self._build_processor(use_temporal_as_final=use_temporal_as_final)

            # The input frame is used only in memory for inference and is not persisted.
            _, result = processor.process_frame(frame_bgr)

            if session is not None:
                session.frame_count += 1
                session.updated_at = time.time()

            return serialize_result(
                result,
                session_id=session.session_id if session is not None else None,
                include_debug=include_debug,
            )


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    return value


def _float_scores(scores: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in (scores or {}).items()}


def serialize_result(
    result: dict[str, Any],
    session_id: str | None = None,
    include_debug: bool = False,
) -> dict[str, Any]:
    payload = {
        "detected": bool(result.get("detected", False)),
        "emotion": result.get("emotion"),
        "confidence": float(result.get("confidence", 0.0)),
        "scores": _float_scores(result.get("scores", {})),
        "rule_based_emotion": result.get("rule_based_emotion"),
        "rule_based_confidence": float(result.get("rule_based_confidence", 0.0)),
        "temporal": _json_safe(result.get("temporal", {}) or {}),
        "quality": _json_safe(result.get("quality", {}) or {}),
        "bbox": _json_safe(result.get("bbox")),
        "face_count": int(result.get("face_count", 0)),
        "session_id": session_id,
        "debug": _json_safe(result.get("debug", {}) or {}) if include_debug else None,
    }

    return payload


emotion_service = EmotionService()
