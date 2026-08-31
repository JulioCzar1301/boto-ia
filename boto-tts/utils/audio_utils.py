import librosa
import noisereduce as nr
import numpy as np
import scipy.signal as signal
import pyloudnorm as pyln


TARGET_SR = 22050 # Frequência padrão do XTTS
TOP_DB = 30 # Threshold para detectar silêncio


# Converte áudio para o formato esperado pelo pipeline
def convert(audio_input):
    """
    Aceita:
    - path (str)
    - tupla (audio, sr)

    Retorna:
    - áudio mono, float32, TARGET_SR
    """
    if isinstance(audio_input, str):
        # Carrega direto já reamostrando
        audio, sr = librosa.load(audio_input, sr=TARGET_SR, mono=True)
    else:
        audio, sr = audio_input

        # Converte estéreo -> mono
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)

        # Resample se necessário
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

    return audio.astype(np.float32), sr


# Remove frequências muito baixas (ruído de fundo)
def highpass(audio, sr, cutoff=100):
    b, a = signal.butter(5, cutoff / (sr / 2), btype="highpass")
    return signal.lfilter(b, a, audio)


# Mantém apenas faixa de voz humana
def bandpass(audio, sr, low=120, high=7600):
    b, a = signal.butter(4, [low / (sr / 2), high / (sr / 2)], btype="band")
    return signal.lfilter(b, a, audio)


# Pipeline principal de limpeza de áudio (input)
def audio_process(audio, sr):
    """
    Pipeline:
    - conversão
    - highpass
    - redução de ruído
    - bandpass
    """
    audio, sr = convert((audio, sr))

    audio = highpass(audio, sr)

    # audio = nr.reduce_noise(
    #     y=audio,
    #     sr=sr,
    #     stationary=False,
    #     prop_decrease=0.7
    # )

    # audio = bandpass(audio, sr)

    # Remove silêncio
    # audio, _ = librosa.effects.trim(audio, top_db=TOP_DB)

    # # Normaliza amplitude
    # audio = librosa.util.normalize(audio)

    return audio


# Pós-processamento do áudio gerado (output)
def audio_process_output(audio, sr):
    """
    Ajusta:
    - ruído de baixa frequência
    - loudness (LUFS)
    """
    audio, sr = convert((audio, sr))

    audio = highpass(audio, sr, cutoff=80)

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)

    if np.isfinite(loudness):
        audio = pyln.normalize.loudness(audio, loudness, -16.0)
        
    audio = np.clip(audio, -1.0, 1.0)

    return audio


# Divide áudio em chunks
def split_audio(audio, sr, chunk_duration=5, overlap=1):
    """
    Estratégia:
    - detecta regiões com fala
    - divide em janelas com overlap
    """
    intervals = librosa.effects.split(audio, top_db=TOP_DB)

    chunk_size = int(chunk_duration * sr)
    step = int((chunk_duration - overlap) * sr)

    chunks = []

    for start, end in intervals:
        segment = audio[start:end]

        for i in range(0, len(segment), step):
            chunk = segment[i:i + chunk_size]

            if len(chunk) >= chunk_size * 0.5:
                chunks.append(chunk)

    if len(chunks) == 0:
        chunks = [audio]

    return chunks


def concatenate_audios(audio_paths):
    audios = []
    silence = np.zeros(int(0.2 * TARGET_SR))

    for path in audio_paths:
        audio, sr = convert(path)

        if sr != TARGET_SR:
            raise ValueError(f"Sample rate diferente em {path}")

        audios.append(audio)
        audios.append(silence)

    full_audio = np.concatenate(audios)

    return full_audio, TARGET_SR