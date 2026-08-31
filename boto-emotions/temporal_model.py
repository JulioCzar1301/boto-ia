from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


OMG_LABEL_TO_EMOTION = {
    0: "anger",
    3: "happy",
    4: "neutral",
    5: "sadness",
    6: "surprise",
}


class BiLSTMClassifier(nn.Module):
    """Mesma arquitetura usada no treinamento do checkpoint temporal."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * direction_factor, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        return self.classifier(last_output)


class TemporalEmotionPerceiver:
    """
    Carrega o .pt treinado no OMG-Emotion e gera uma predição temporal em paralelo.

    Importante:
    - Este modelo foi treinado sem a classe Surprise.
    - Ele não substitui automaticamente a saída atual do sistema.
    - Ele usa uma fila dos últimos `window_size` frames, normalmente 30.
    """

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device)

        checkpoint = torch.load(self.model_path, map_location=self.device)

        self.feature_columns: List[str] = checkpoint["feature_columns"]
        self.window_size: int = int(checkpoint["window_size"])
        self.hidden_size: int = int(checkpoint["hidden_size"])
        self.num_layers: int = int(checkpoint.get("num_layers", 1))
        self.dropout: float = float(checkpoint.get("dropout", 0.0))
        self.num_classes: int = int(checkpoint["num_classes"])

        self.scaler_mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
        self.scaler_scale[self.scaler_scale == 0] = 1.0

        self.inverse_label_mapping = checkpoint.get("inverse_label_mapping", {})
        self.index_to_emotion = self._build_index_to_emotion()

        self.model = BiLSTMClassifier(
            input_size=int(checkpoint["input_size"]),
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=True,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.feature_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)


    def clone_for_new_face(self) -> "TemporalEmotionPerceiver":
        """
        Cria uma nova instancia logica para outro rosto, reutilizando o mesmo
        modelo carregado em memoria, mas com buffer temporal independente.
        """
        clone = object.__new__(TemporalEmotionPerceiver)

        clone.model_path = self.model_path
        clone.device = self.device

        clone.feature_columns = self.feature_columns
        clone.window_size = self.window_size
        clone.hidden_size = self.hidden_size
        clone.num_layers = self.num_layers
        clone.dropout = self.dropout
        clone.num_classes = self.num_classes

        clone.scaler_mean = self.scaler_mean
        clone.scaler_scale = self.scaler_scale

        clone.inverse_label_mapping = self.inverse_label_mapping
        clone.index_to_emotion = self.index_to_emotion

        clone.model = self.model
        clone.feature_buffer = deque(maxlen=self.window_size)

        return clone

    def _build_index_to_emotion(self) -> Dict[int, str]:
        index_to_emotion: Dict[int, str] = {}

        for encoded_idx_str, original_label_str in self.inverse_label_mapping.items():
            encoded_idx = int(encoded_idx_str)
            original_label = int(original_label_str)
            index_to_emotion[encoded_idx] = OMG_LABEL_TO_EMOTION.get(
                original_label,
                str(original_label),
            )

        return index_to_emotion

    @staticmethod
    def _num(value, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, (bool, np.bool_)):
            return 1.0 if value else 0.0
        try:
            if np.isnan(value):
                return default
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return default

    def _get_feature_value(
        self,
        feature_name: str,
        enet_scores: Dict[str, float],
        mediapipe_scores: Dict[str, float],
        evidence: Dict[str, float],
        debug_info: Dict[str, float],
        quality_info: Dict[str, float],
    ) -> float:
        if feature_name.startswith("enet_"):
            key = feature_name.replace("enet_", "", 1)
            return self._num(enet_scores.get(key, 0.0))

        if feature_name.startswith("mp_"):
            key = feature_name.replace("mp_", "", 1)
            return self._num(mediapipe_scores.get(key, 0.0))

        if feature_name.startswith("evidence_"):
            key = feature_name.replace("evidence_", "", 1)
            return self._num(evidence.get(key, 0.0))

        if feature_name.startswith("debug_"):
            key = feature_name.replace("debug_", "", 1)
            return self._num(debug_info.get(key, 0.0))

        if feature_name.startswith("quality_"):
            key = feature_name.replace("quality_", "", 1)
            return self._num(quality_info.get(key, 0.0))

        return 0.0

    def build_feature_vector(
        self,
        enet_scores: Dict[str, float],
        mediapipe_scores: Dict[str, float],
        evidence: Dict[str, float],
        debug_info: Dict[str, float],
        quality_info: Dict[str, float],
    ) -> np.ndarray:
        values = [
            self._get_feature_value(
                feature_name=col,
                enet_scores=enet_scores,
                mediapipe_scores=mediapipe_scores,
                evidence=evidence,
                debug_info=debug_info,
                quality_info=quality_info,
            )
            for col in self.feature_columns
        ]

        return np.asarray(values, dtype=np.float32)

    def update(
        self,
        enet_scores: Dict[str, float],
        mediapipe_scores: Dict[str, float],
        evidence: Dict[str, float],
        debug_info: Dict[str, float],
        quality_info: Dict[str, float],
    ) -> Dict[str, object]:
        raw_features = self.build_feature_vector(
            enet_scores=enet_scores,
            mediapipe_scores=mediapipe_scores,
            evidence=evidence,
            debug_info=debug_info,
            quality_info=quality_info,
        )

        scaled_features = (raw_features - self.scaler_mean) / self.scaler_scale
        self.feature_buffer.append(scaled_features.astype(np.float32))

        if len(self.feature_buffer) < self.window_size:
            return {
                "ready": False,
                "emotion": None,
                "confidence": 0.0,
                "scores": {},
                "frames_ready": len(self.feature_buffer),
                "window_size": self.window_size,
            }

        x = np.stack(list(self.feature_buffer), axis=0)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        emotion = self.index_to_emotion.get(pred_idx, str(pred_idx))
        confidence = float(probs[pred_idx])

        scores = {
            self.index_to_emotion.get(i, str(i)): float(probs[i])
            for i in range(len(probs))
        }

        return {
            "ready": True,
            "emotion": emotion,
            "confidence": confidence,
            "scores": scores,
            "frames_ready": len(self.feature_buffer),
            "window_size": self.window_size,
        }
