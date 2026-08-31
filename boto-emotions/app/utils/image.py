from __future__ import annotations

import base64
import binascii

import cv2
import numpy as np

from app.settings import MAX_IMAGE_BASE64_CHARS


def validate_image_payload_size(image_b64: str) -> None:
    if len(image_b64) > MAX_IMAGE_BASE64_CHARS:
        raise ValueError(
            "Imagem excede o tamanho maximo aceito para inferencia em memoria."
        )


def strip_data_uri(image_b64: str) -> str:
    if "," in image_b64 and image_b64.strip().lower().startswith("data:image"):
        return image_b64.split(",", 1)[1]
    return image_b64


def b64_to_bgr(image_b64: str) -> np.ndarray:
    validate_image_payload_size(image_b64)
    raw_b64 = strip_data_uri(image_b64).strip()

    try:
        image_bytes = base64.b64decode(raw_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Imagem base64 invalida.") from exc

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Nao foi possivel decodificar a imagem.")

    return frame
