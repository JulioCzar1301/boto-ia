import torch
import torchaudio
import soundfile as sf

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    XttsArgs,
    BaseDatasetConfig,
])

# Patch: força torchaudio.load a usar soundfile, evitando torchcodec
def _load_with_soundfile(filepath, *args, **kwargs):
    data, sr = sf.read(filepath, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)
    return waveform, sr

torchaudio.load = _load_with_soundfile

from TTS.api import TTS

class TTSService:
    def __init__(self):
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    def infer(self, text, embedding):
        return self.tts.synthesizer.tts_model.inference(
            text=text,
            language="pt",
            gpt_cond_latent=embedding["gpt_cond_latent"],
            speaker_embedding=embedding["speaker_embedding"],
            temperature=0.65,
            repetition_penalty=10.0,
            top_k=50,
            top_p=0.85,
            do_sample=True,
            speed=1.05,
        )