from __future__ import annotations

import os


APP_NAME = "BotoLens Emotion API"
APP_VERSION = "0.1.0"

# Privacy defaults: images are accepted only for in-memory inference.
STORE_IMAGES = False
STORE_REQUEST_PAYLOADS = False
LOG_REQUEST_PAYLOADS = False

MAX_IMAGE_BASE64_CHARS = int(os.getenv("MAX_IMAGE_BASE64_CHARS", "2500000"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "300"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
