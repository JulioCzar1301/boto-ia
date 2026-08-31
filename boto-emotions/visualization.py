import os
import cv2

from config import EMOTION_COLORS, MAIN_EMOTIONS
from utils import get_winner


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMOTION_ICON_PATHS = {
    "happy": os.path.join(BASE_DIR, "icons", "happy.png"),
    "sadness": os.path.join(BASE_DIR, "icons", "sadness.png"),
    "anger": os.path.join(BASE_DIR, "icons", "anger.png"),
    "surprise": os.path.join(BASE_DIR, "icons", "surprise.png"),
    "neutral": os.path.join(BASE_DIR, "icons", "neutral.png"),
}


def draw_text(img, text, x, y, scale=0.65, color=(0, 255, 0), thickness=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def get_emotion_color(emotion):
    return EMOTION_COLORS.get(emotion, (255, 255, 255))


def draw_emotion_icon(frame_out, emotion, bbox, size=64):
    icon_path = EMOTION_ICON_PATHS.get(emotion)

    if not icon_path:
        return

    if not os.path.exists(icon_path):
        return

    icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)

    if icon is None:
        return

    icon = cv2.resize(icon, (size, size), interpolation=cv2.INTER_AREA)

    x1p, y1p, x2p, y2p = bbox

    margin = 10
    frame_h, frame_w = frame_out.shape[:2]

    icon_x = x1p + ((x2p - x1p) // 2) - (size // 2)

    # Tenta acima da bbox
    icon_y = y1p - size - margin

    # Se não couber acima, coloca abaixo da bbox
    if icon_y < 0:
        icon_y = y2p + margin

    # Ajusta para não sair da tela
    icon_x = max(0, min(icon_x, frame_w - size))
    icon_y = max(0, min(icon_y, frame_h - size))

    frame_h, frame_w = frame_out.shape[:2]

    icon_x = max(0, min(icon_x, frame_w - size))
    icon_y = max(0, min(icon_y, frame_h - size))

    roi = frame_out[icon_y:icon_y + size, icon_x:icon_x + size]

    if roi.shape[0] != size or roi.shape[1] != size:
        return

    if icon.shape[2] == 4:
        alpha = icon[:, :, 3] / 255.0

        for c in range(3):
            roi[:, :, c] = (
                alpha * icon[:, :, c]
                + (1 - alpha) * roi[:, :, c]
            )
    else:
        roi[:, :, :] = icon[:, :, :3]


def draw_detection_overlay(
    frame_out,
    bbox,
    emotion,
    confidence,
    smoothed_scores,
    mediapipe_scores,
    enet_scores,
    fused_scores,
    quality_info,
    debug_info,
    temporal_result=None,
    rule_based_emotion=None,
    rule_based_confidence=0.0,
    show_debug_panel=False,
):
    h, _, _ = frame_out.shape
    x1p, y1p, x2p, y2p = bbox
    color = get_emotion_color(emotion)

    # =====================================================
    # OVERLAY BÁSICO — SEMPRE EXIBIDO
    # =====================================================

    cv2.rectangle(frame_out, (x1p, y1p), (x2p, y2p), color, 2)

    bbox_width = x2p - x1p
    icon_size = max(80, int(bbox_width * 0.35))

    draw_emotion_icon(
        frame_out=frame_out,
        emotion=emotion,
        bbox=bbox,
        size=icon_size
    )

    draw_text(
        frame_out,
        f"{emotion} ({confidence:.2f})",
        x1p,
        max(30, y1p - 10),
        scale=0.75,
        color=color,
        thickness=2
    )

    # Avisos de qualidade aparecem apenas no modo debug.
    if show_debug_panel and quality_info.get("quality_warning", False):
        msg = quality_info.get("quality_message", "Face pouco confiavel.")
        draw_text(
            frame_out,
            msg,
            x1p,
            min(h - 20, y2p + 25),
            scale=0.65,
            color=(0, 165, 255),
            thickness=2
        )

    # Se debug estiver desligado, para aqui.
    # Resultado: bbox + emoção + ícone.
    if not show_debug_panel:
        return

    # =====================================================
    # PAINEL DE DEBUG — OPCIONAL
    # =====================================================

    mp_winner, mp_score = get_winner(mediapipe_scores)
    enet_winner, enet_score = get_winner(enet_scores)
    fused_winner, fused_score = get_winner(fused_scores)
    final_winner, final_score = get_winner(smoothed_scores)

    painel_x = 20
    painel_y = 30
    draw_text(frame_out, "Scores finais:", painel_x, painel_y, scale=0.65)

    for i, emo in enumerate(MAIN_EMOTIONS, start=1):
        val = smoothed_scores.get(emo, 0.0)
        draw_text(frame_out, f"{emo}: {val:.2f}", painel_x, painel_y + i * 24, scale=0.58, color=get_emotion_color(emo))

    y_extra = painel_y + (len(MAIN_EMOTIONS) + 2) * 24
    temporal_result = temporal_result or {}
    temporal_ready = temporal_result.get("ready", False)
    temporal_emotion = temporal_result.get("emotion") if temporal_ready else "aquecendo"
    temporal_confidence = temporal_result.get("confidence", 0.0) if temporal_ready else 0.0

    temporal_color = (255, 255, 255)
    if temporal_ready:
        temporal_color = get_emotion_color(temporal_emotion)

    temporal_status = (
        f"BiLSTM temporal: {temporal_emotion} ({temporal_confidence:.2f})"
        if temporal_ready
        else f"BiLSTM temporal: aquecendo {temporal_result.get('frames_ready', 0)}/{temporal_result.get('window_size', 30)}"
    )

    agreement = "N/A"
    if temporal_ready and rule_based_emotion is not None:
        agreement = "SIM" if temporal_emotion == rule_based_emotion else "NAO"

    lines = [
        (f"MediaPipe sug.: {mp_winner} ({mp_score:.2f})", get_emotion_color(mp_winner)),
        (f"ENet sug.: {enet_winner} ({enet_score:.2f})", get_emotion_color(enet_winner)),
        (f"Fusao antes suav.: {fused_winner} ({fused_score:.2f})", get_emotion_color(fused_winner)),
        (f"Regra atual: {rule_based_emotion or final_winner} ({rule_based_confidence or final_score:.2f})", get_emotion_color(rule_based_emotion or final_winner)),
        (temporal_status, temporal_color),
        (f"Concordancia regra x BiLSTM: {agreement}", (255, 255, 255)),
        (
            f"Fiscal troca: {debug_info.get('temporal_switch_decision', 'not_used')} | "
            f"{debug_info.get('temporal_switch_candidate_count', 0)}/"
            f"{debug_info.get('temporal_switch_needed_persistence', 0)}",
            (255, 255, 255)
        ),
        (f"Quality warn: {quality_info.get('quality_warning', False)} | Penalty: {quality_info.get('quality_penalty_active', False)}", (255, 255, 255)),
        (f"Yaw:{quality_info.get('yaw', 0.0):.1f} Pitch:{quality_info.get('pitch', 0.0):.1f} Roll:{quality_info.get('roll', 0.0):.1f}", (255, 255, 255)),
        (f"FaceRatio:{quality_info.get('face_ratio', 0.0):.3f} Asym:{quality_info.get('face_asymmetry', 0.0):.2f}", (255, 255, 255)),
        (f"SadEv:{debug_info.get('sadness_evidence', 0.0):.2f} SadF:{debug_info.get('sadness_factor', 1.0):.2f}", (255, 255, 255)),
        (f"AngEv:{debug_info.get('anger_evidence', 0.0):.2f} AngF:{debug_info.get('anger_factor', 1.0):.2f}", (255, 255, 255)),
        (f"SadIntent:{debug_info.get('sadness_intent', 0.0):.2f} SadMouth:{debug_info.get('sad_mouth_signature', 0.0):.2f}", (255, 255, 255)),
        (f"SadFace:{debug_info.get('sad_face_pattern', 0.0):.2f} SadOverride:{debug_info.get('sadness_override_neutral', False)}", (255, 255, 255)),
        (f"AngIntent:{debug_info.get('anger_intent', 0.0):.2f} AngTens:{debug_info.get('anger_tension_signature', 0.0):.2f}", (255, 255, 255)),
        (f"SadStruct:{debug_info.get('sadness_structural', False)} AngStruct:{debug_info.get('anger_structural', False)} EyeNoise:{debug_info.get('eye_only_sadness_noise', False)}", (255, 255, 255)),
    ]

    for i, (text, text_color) in enumerate(lines):
        draw_text(frame_out, text, painel_x, y_extra + i * 22, scale=0.52 if i >= 4 else 0.55, color=text_color)