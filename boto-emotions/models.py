import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import ENET_IDX_TO_CLASS, MEDIAPIPE_MODEL_PATH, MAX_NUM_FACES


class ENetEmotionPerceiver:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.img_size = 224

        self.transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        model = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(model.classifier, torch.nn.Sequential):
            self.classifier_weights = model.classifier[0].weight.detach().cpu().numpy()
            self.classifier_bias = model.classifier[0].bias.detach().cpu().numpy()
        else:
            self.classifier_weights = model.classifier.weight.detach().cpu().numpy()
            self.classifier_bias = model.classifier.bias.detach().cpu().numpy()

        model.classifier = torch.nn.Identity()
        self.model = model.to(device).eval()

    def predict(self, face_rgb):
        img = Image.fromarray(face_rgb)
        tensor = self.transforms(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor).detach().cpu().numpy()

        logits = np.dot(features, self.classifier_weights.T) + self.classifier_bias
        logits = logits[0]

        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()

        full_scores = {
            ENET_IDX_TO_CLASS[i]: float(probs[i])
            for i in range(len(probs))
        }

        reduced_scores = {
            "happy": full_scores.get("happy", 0.0),
            "sadness": full_scores.get("sadness", 0.0),
            "anger": full_scores.get("anger", 0.0),
            "surprise": full_scores.get("surprise", 0.0),
            "neutral": full_scores.get("neutral", 0.0),
        }

        return reduced_scores, full_scores


# =========================
# MEDIAPIPE FACE LANDMARKER
# =========================
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



def create_face_landmarker(model_path=MEDIAPIPE_MODEL_PATH):
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=MAX_NUM_FACES,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.6,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)
