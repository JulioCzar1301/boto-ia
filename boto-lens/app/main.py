"""
API principal para detecção de objetos — Boto-Lens.

Endpoints:
  POST /detection/fused      [PRINCIPAL] Qwen + YOLOE paralelo → fusão IoU → verificação de crop
  POST /detection            Qwen autônomo (sem YOLOE)
  POST /detection/sys_prompt Qwen com system prompt customizado
  POST /detection/sequential Pipeline legado: YOLOE detecta → Qwen renomeia
  GET  /health               Health check

Pipeline /detection/fused (6 etapas):

  1. Paralelo    — Qwen detecta objetos + bboxes; YOLOE detecta candidatos com bboxes precisas.
  2. Foreground  — Remove ruído (< 1.5%), fundo (> 75%) e objetos periféricos pequenos.
  3. Fusão IoU   — IoU ≥ 0.15: label Qwen + bbox YOLOE (source="fused"); sem match: mantém Qwen.
  4. Verificação — Qwen recebe o crop de cada objeto e confirma/corrige label e score.
                   Score < 0.6 → objeto descartado.
  5. Sub-partes  — Descarta bboxes cujo interior (> 60%) está dentro de um bbox maior.
  6. Ranking     — score × (0.6 + 0.4 × centralidade) → top 5.
"""

import asyncio
import httpx
from fastapi import APIRouter, FastAPI

from config import IOI_THRESHOLD, TOP_K_RESULTS
from schemas import Prompt, PromptSys
from system_instruction import SYSTEM_INSTRUCTION
from models import DetectedObject
from services.yoloe import run_yoloe, get_yoloe
from services.qwen import (
    call_qwen,
    call_verify,
    call_vllm_sequential,
    parse_qwen_response,
    parse_verify_response,
    parse_qwen_refine_response,
)
from services.fusion import fuse_by_iou, serialize_detections
from utils.image import resize_base64_image, b64_to_pil, pil_to_b64
from utils.geometry import centrality, is_foreground, remove_subparts, crop_object

app = FastAPI(title="Boto-Lens", description="API de detecção de objetos com fusão Qwen + YOLOE")

# Constantes para processamento de imagens
IMAGE_MAX_SIDE = 1024  # Reduzido de 1024 para economizar tokens
IMAGE_JPEG_QUALITY = 60  # Reduzido de 82 para maior compressão
# ──────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    print("Inicializando... Carregando modelo YOLO...")
    try:
        get_yoloe()
        print("Modelo YOLO carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo YOLO: {e}")
        raise


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.post("/detection/fused")
async def detection_fused(body: Prompt) -> dict:
    image_b64 = resize_base64_image(body.image, body.size or IMAGE_MAX_SIDE, body.quality or IMAGE_JPEG_QUALITY)
    img = b64_to_pil(image_b64)
    img_size = img.size
    loop = asyncio.get_event_loop()

    # Acumulador — só é preenchido se solicitado
    token_usage = {"input_tokens": 0, "output_tokens": 0}

    def _accumulate(response: dict) -> None:
        """Extrai usage do response vLLM e acumula, se solicitado."""
        if not body.include_token_usage:
            return
        usage = response.get("usage", {})
        token_usage["input_tokens"]  += usage.get("prompt_tokens", 0)
        token_usage["output_tokens"] += usage.get("completion_tokens", 0)

    # ── Etapa 1 ────────────────────────────────────────────────────────────
    async with httpx.AsyncClient(timeout=120.0) as client:
        qwen_task  = call_qwen(client, image_b64, SYSTEM_INSTRUCTION)
        yoloe_task = loop.run_in_executor(None, run_yoloe, img)
        qwen_response, yoloe_objects = await asyncio.gather(qwen_task, yoloe_task)

    _accumulate(qwen_response)  # ← contabiliza chamada principal
    qwen_objects = parse_qwen_response(qwen_response, img_size=img_size)

    if not qwen_objects:
        print("Qwen sem objetos — fallback para YOLOE")
        fallback = [o for o in yoloe_objects if is_foreground(o)] or yoloe_objects
        fallback = sorted(
            fallback,
            key=lambda o: o.score * (0.6 + 0.4 * centrality(o.bbox)),
            reverse=True,
        )[:TOP_K_RESULTS]
        return serialize_detections(fallback, too_many=False)

    # ── Etapa 2: Filtros de foreground ─────────────────────────────────────
    qwen_filtered  = [o for o in qwen_objects  if is_foreground(o)] or qwen_objects
    yoloe_filtered = [o for o in yoloe_objects if is_foreground(o)]
    print(f"Após filtro — Qwen: {len(qwen_filtered)}, YOLOE: {len(yoloe_filtered)}")

    # ── Etapa 3: Fusão IoU ─────────────────────────────────────────────────
    fused = fuse_by_iou(qwen_filtered, yoloe_filtered, iou_threshold=IOI_THRESHOLD)
    print(f"Pós-fusão: {len(fused)} objeto(s)")

    # ── Etapa 4: Verificação de crop (paralelo) ────────────────────────────
    if body.verify_crops:
        async with httpx.AsyncClient(timeout=120.0) as client:
            verify_tasks = [
                call_verify(client, pil_to_b64(crop_object(img, obj.bbox)), obj.label)
                for obj in fused
            ]
            verify_responses = await asyncio.gather(*verify_tasks)

        for vresp in verify_responses:
            _accumulate(vresp)  # ← contabiliza cada verificação de crop

        verified: list[DetectedObject] = []
        for obj, vresp in zip(fused, verify_responses):
            label, score = parse_verify_response(vresp)
            if not label:
                print(f"Descartado na verificação: '{obj.label}' (score baixo)")
                continue
            verified.append(DetectedObject(
                label=label, score=score,
                bbox=obj.bbox, source=obj.source, yoloe_conf=obj.yoloe_conf,
            ))
        print(f"Após verificação: {len(verified)} objeto(s)")
    else:
        verified = fused

    # ── Etapa 5: Remove sub-partes ─────────────────────────────────────────
    verified = remove_subparts(verified)
    print(f"Após remoção de sub-partes: {len(verified)} objeto(s)")

    # ── Etapa 6: Ranking final ─────────────────────────────────────────────
    final = sorted(
        verified,
        key=lambda o: o.score * (0.6 + 0.4 * centrality(o.bbox)),
        reverse=True,
    )[:TOP_K_RESULTS]

    result = serialize_detections(final)

    if body.include_token_usage:
        token_usage["total_tokens"] = token_usage["input_tokens"] + token_usage["output_tokens"]
        result["token_usage"] = token_usage

    return result


@app.post("/detection")
async def detection(body: Prompt) -> dict:
    """Qwen autônomo — labels e bboxes gerados pelo Qwen, sem YOLOE."""
    image_b64 =  resize_base64_image(body.image, body.size or IMAGE_MAX_SIDE,  body.quality or IMAGE_JPEG_QUALITY)
    img = b64_to_pil(image_b64)
    async with httpx.AsyncClient(timeout=120.0) as client:
        return serialize_detections(
            parse_qwen_response(
                await call_qwen(client, image_b64, SYSTEM_INSTRUCTION),
                img_size=img.size,
            ),
            too_many=False,
        )


@app.post("/detection/sys_prompt")
async def detection_sys(body: PromptSys) -> dict:
    """Qwen com system prompt customizado pelo cliente."""
    image_b64 =  resize_base64_image(body.image, body.size or IMAGE_MAX_SIDE,  body.quality or IMAGE_JPEG_QUALITY)
    img = b64_to_pil(image_b64)
    async with httpx.AsyncClient(timeout=120.0) as client:
        return serialize_detections(
            parse_qwen_response(
                await call_qwen(client, image_b64, body.prompt),
                img_size=img.size,
            ),
            too_many=False,
        )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

