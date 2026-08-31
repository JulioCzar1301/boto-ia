import os
import torch
import tempfile
import numpy as np
import soundfile as sf
import librosa
import whisper

TARGET_SR = 22050  # Frequência padrão do XTTS

class EmbeddingService:
    def __init__(self, tts_model):
        """
        Responsável por:
        - gerar embeddings de speaker
        - cachear embeddings
        - selecionar melhores chunks
        """
        self.tts_model = tts_model
        self.cache = {}
        self.asr_cache = {}
        self.asr = whisper.load_model("base")

        os.makedirs("embeddings", exist_ok=True)

    def fast_score(self, chunk):
        """
        Pré-filtro rápido:
        - energia * variação
        """
        return np.mean(np.abs(chunk)) * np.std(chunk)

    def score_chunk(self, chunk):
        """
        Score completo:
        - ASR
        - energia
        - pitch
        """
        if len(chunk) < TARGET_SR * 1:  # menos de 1s
            return 0.01

        energy = np.mean(np.abs(chunk))
        std = np.std(chunk)

        # energia
        if energy < 0.005:
            return 0.01

        # pitch 
        pitches, _ = librosa.piptrack(y=chunk, sr=TARGET_SR)
        pitch_var = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        # ASR
        text_score = 1

        try:
            key = hash(chunk.tobytes())

            if key in self.asr_cache:
                text = self.asr_cache[key]
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    sf.write(tmp.name, chunk, TARGET_SR)
                    result = self.asr.transcribe(tmp.name, fp16=False)
                    text = result.get("text", "").strip()
                    self.asr_cache[key] = text

            text_score = max(len(text), 1)

        except:
            text_score = 1  # fallback

        return (energy * 5) + (std * 3) + (pitch_var * 0.3) + (text_score * 0.5)


    def get_or_create(self, speaker, audio_chunks):
        """
        Gera embedding de speaker a partir de múltiplos chunks de áudio.

        Pipeline:
        1. Seleciona melhores chunks 
        2. Extrai embeddings individuais 
        3. Faz agregação (mediana)
        """
        
        if speaker in self.cache:
            return self.cache[speaker]

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        EMBED_DIR = os.path.join(BASE_DIR, "embeddings")

        os.makedirs(EMBED_DIR, exist_ok=True)

        path = os.path.join(EMBED_DIR, f"{speaker}.pt")
        
        if os.path.exists(path):
            data = torch.load(path)
            self.cache[speaker] = data
            return data

        if audio_chunks is None:
            raise ValueError(f"Embedding '{speaker}' não existe e nenhum áudio foi enviado")

        embedding = self.create_embedding(audio_chunks)
        torch.save(embedding, path)

        self.cache[speaker] = embedding
        return embedding


    def create_embedding(self, audio_chunks):
        """
        Gera embedding de speaker a partir de múltiplos chunks de áudio.

        Pipeline:
        1. Seleciona melhores chunks 
        2. Extrai embeddings individuais 
        3. Faz agregação (mediana)
        """
        
        if len(audio_chunks) == 0:
            raise ValueError("Nenhum chunk recebido")

        # PRÉ-SELEÇÃO 
        prev = sorted(
            [(self.fast_score(c), c) for c in audio_chunks],
            reverse=True
        )[:20]  # pega 20 melhores chunks


        # SELEÇÃO FINAL
        scored = sorted(
            [(self.score_chunk(c), c) for _, c in prev],
            reverse=True
        )
        
        gpt_list = []        # contexto de linguagem 
        speaker_list = []    # identidade da voz


        # EXTRAÇÃO DE EMBEDDINGS
        for _, chunk in scored:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, chunk, TARGET_SR)

                # Extrai embeddings do modelo
                gpt, speaker = self.tts_model.get_conditioning_latents(
                    audio_path=tmp.name
                )

                # Armazena embeddings individuais
                gpt_list.append(gpt)
                speaker_list.append(speaker)

        # AGREGAÇÃO
        return {
            "gpt_cond_latent": torch.median(
                torch.stack(gpt_list), dim=0
            ).values,

            "speaker_embedding": torch.median(
                torch.stack(speaker_list), dim=0
            ).values,
        }