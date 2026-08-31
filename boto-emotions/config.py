import torch

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MEDIAPIPE_MODEL_PATH = str(BASE_DIR / "models" / "face_landmarker.task")
ENET_MODEL_PATH = str(BASE_DIR / "models" / "enet_b0_8_best_afew.pt")
TEMPORAL_MODEL_PATH = str(BASE_DIR / "models" / "thaylor_temporal_best_omge.pt")

USE_TEMPORAL_MODEL = True
USE_TEMPORAL_AS_FINAL = False

MAX_NUM_FACES = 5



SHOW_SCORES_PANEL = True

WINDOW_SIZE = 8
SWITCH_MARGIN = 0.14
SWITCH_PERSISTENCE = 4
ALPHA = 0.25

WARN_YAW_DEG = 35
WARN_PITCH_DEG = 35
WARN_ROLL_DEG = 30

PENALTY_YAW_DEG = 48
PENALTY_PITCH_DEG = 45
PENALTY_ROLL_DEG = 42

WARN_FACE_RATIO_MIN = 0.060
WARN_FACE_RATIO_MAX = 0.420
PENALTY_FACE_RATIO_MIN = 0.040
PENALTY_FACE_RATIO_MAX = 0.520

ASYMMETRY_WARN = 0.24
ASYMMETRY_PENALTY = 0.34

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENET_IDX_TO_CLASS = {
    0: "anger",
    1: "contempt",
    2: "disgust",
    3: "fear",
    4: "happy",
    5: "neutral",
    6: "sadness",
    7: "surprise",
}

MAIN_EMOTIONS = [
    "happy",
    "sadness",
    "anger",
    "surprise",
    "neutral",
]

ACTIVE_EMOTIONS = [
    "happy",
    "sadness",
    "anger",
    "surprise",
]

EMOTION_COLORS = {
    "happy": (0, 255, 0),
    "sadness": (255, 0, 0),
    "anger": (0, 0, 255),
    "surprise": (0, 255, 255),
    "neutral": (200, 200, 200),
}
