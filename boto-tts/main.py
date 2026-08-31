from services.audio_service import AudioService
from services.embedding_service import EmbeddingService
from services.tts_service import TTSService
import os

# def get_audio_files(folder):
#     return [
#         os.path.join(folder, f)
#         for f in os.listdir(folder)
#         if f.endswith((".wav", ".mp3", ".aac", ".ogg"))
#     ]
    
def get_audio_files(folder, selected_names=None):
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith((".wav", ".mp3", ".aac", ".ogg"))
    ]

    if selected_names:
        files = [
            f for f in files
            if os.path.basename(f) in selected_names
        ]

    return files

def generate_speech(text, speaker, audio_input=None):
    audio_service = AudioService()
    tts_service = TTSService()
    
    embedding_service = EmbeddingService(
        tts_service.tts.synthesizer.tts_model
    )

    if audio_input is None:
        chunks = None

    elif isinstance(audio_input, list):
        audio, sr = audio_service.concatenate(audio_input)
        chunks = audio_service.preprocess((audio, sr)) 

    else:
        chunks = audio_service.preprocess(audio_input)

    embedding = embedding_service.get_or_create(speaker, chunks)

    out = tts_service.infer(text, embedding)
    wav = out["wav"] if isinstance(out, dict) else out

    wav = audio_service.postprocess(wav)

    os.makedirs("voices", exist_ok=True)
    output_path = f"voices/{speaker}.wav"
    tts_service.tts.synthesizer.save_wav(wav, output_path)

    return output_path

if __name__ == "__main__":
    folder = "teste_larissa"
    speaker = "larissa"
    
    # selected = ["phrase_1_2.ogg",
    #             "phrase_2_2.ogg",
    #             "phrase_3_2.ogg",
    #             "phrase_4_2.ogg",
    #             "phrase_5_2.ogg"]
    
    # audios = get_audio_files(folder, selected)
    # audios.append("larissa.aac")

    output = generate_speech(
        # audio_input="teste_larissa/phrases_all.ogg",
        text="geladeira",
        speaker=speaker,
    )