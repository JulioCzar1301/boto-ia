from fastapi import FastAPI, UploadFile, File, Form, APIRouter
from fastapi.responses import Response
import io
import torch
import tempfile
import soundfile as sf

from services.audio_service import AudioService
from services.embedding_service import EmbeddingService
from services.tts_service import TTSService


app = FastAPI()

router = APIRouter(prefix="/boto-tts")
audio_service = AudioService()
tts_service = TTSService()

embedding_service = EmbeddingService(
    tts_service.tts.synthesizer.tts_model
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/embedding")
async def generate_embedding(
    audio: UploadFile = File(...)
):
    audio_bytes = await audio.read()

    suffix = "." + audio.filename.split(".")[-1]

    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        chunks = audio_service.preprocess(tmp.name)

        embedding = embedding_service.create_embedding(chunks)

        for key, value in embedding.items():
            if hasattr(value, "cpu"):
                embedding[key] = value.cpu()

        buffer = io.BytesIO()
        torch.save(embedding, buffer)
        buffer.seek(0)

        return Response(
            content=buffer.read(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=embedding.pt"
            }
        )


@app.post("/tts")
async def tts(
    text: str = Form(...),
    embedding: UploadFile = File(...)
):
    embedding_bytes = await embedding.read()

    embedding = torch.load(
        io.BytesIO(embedding_bytes),
        map_location="cpu"
    )

    output = tts_service.infer(text, embedding)

    wav = output["wav"] if isinstance(output, dict) else output
    wav = audio_service.postprocess(wav)

    wav_buffer = io.BytesIO()

    sf.write(
        wav_buffer,
        wav,
        samplerate=24000,
        format="WAV"
    )

    return Response(
        content=wav_buffer.getvalue(),
        media_type="audio/wav"
    )
