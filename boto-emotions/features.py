from utils import clamp01, get_bs, mean_or_current


def mouth_down_from_landmarks(face_landmarks):
    left_corner = face_landmarks[61]
    right_corner = face_landmarks[291]
    top_lip = face_landmarks[13]
    bottom_lip = face_landmarks[14]

    mouth_center_y = (top_lip.y + bottom_lip.y) / 2.0
    left_drop = left_corner.y - mouth_center_y
    right_drop = right_corner.y - mouth_center_y
    return clamp01(((left_drop + right_drop) / 2.0) * 18.0)


def extract_mediapipe_evidence(blend_dict, face_landmarks, feature_history):
    smile = (get_bs(blend_dict, "mouthSmileLeft") + get_bs(blend_dict, "mouthSmileRight")) / 2.0
    frown = (get_bs(blend_dict, "mouthFrownLeft") + get_bs(blend_dict, "mouthFrownRight")) / 2.0
    brow_inner_up = get_bs(blend_dict, "browInnerUp")
    brow_down = (get_bs(blend_dict, "browDownLeft") + get_bs(blend_dict, "browDownRight")) / 2.0
    eye_wide = (get_bs(blend_dict, "eyeWideLeft") + get_bs(blend_dict, "eyeWideRight")) / 2.0
    eye_squint = (get_bs(blend_dict, "eyeSquintLeft") + get_bs(blend_dict, "eyeSquintRight")) / 2.0
    cheek_squint = (get_bs(blend_dict, "cheekSquintLeft") + get_bs(blend_dict, "cheekSquintRight")) / 2.0
    nose_sneer = (get_bs(blend_dict, "noseSneerLeft") + get_bs(blend_dict, "noseSneerRight")) / 2.0
    mouth_press = (get_bs(blend_dict, "mouthPressLeft") + get_bs(blend_dict, "mouthPressRight")) / 2.0
    jaw_open = get_bs(blend_dict, "jawOpen")
    mouth_lower_down = mouth_down_from_landmarks(face_landmarks)

    delta_brow_down = max(0.0, brow_down - mean_or_current(feature_history["brow_down"], brow_down))
    delta_mouth_press = max(0.0, mouth_press - mean_or_current(feature_history["mouth_press"], mouth_press))
    delta_nose_sneer = max(0.0, nose_sneer - mean_or_current(feature_history["nose_sneer"], nose_sneer))
    delta_eye_squint = max(0.0, eye_squint - mean_or_current(feature_history["eye_squint"], eye_squint))

    happy_evidence = clamp01(0.70 * smile + 0.20 * cheek_squint - 0.15 * frown)

    sadness_eye_component = eye_squint * 0.008 if 0.07 <= eye_squint <= 0.18 else 0.0

    sad_mouth_signature = clamp01(
        0.58 * frown + 0.62 * mouth_lower_down - 0.18 * smile - 0.10 * mouth_press - 0.08 * nose_sneer
    )

    sad_brow_signature = clamp01(
        0.55 * brow_inner_up + 0.20 * frown - 0.20 * brow_down - 0.12 * nose_sneer
    )

    sadness_evidence = clamp01(
        0.58 * sad_mouth_signature + 0.22 * sad_brow_signature + 0.08 * (1.0 - eye_wide) +
        sadness_eye_component - 0.10 * mouth_press - 0.08 * nose_sneer
    )

    anger_tension_signature = clamp01(
        0.34 * delta_brow_down + 0.22 * mouth_press + 0.18 * nose_sneer + 0.14 * eye_squint +
        0.10 * delta_mouth_press + 0.08 * delta_nose_sneer + 0.06 * delta_eye_squint -
        0.22 * brow_inner_up - 0.34 * sad_mouth_signature
    )

    anger_onset = clamp01(
        0.08 * brow_down + 0.92 * anger_tension_signature - 0.25 * sad_mouth_signature -
        0.12 * brow_inner_up - 0.08 * smile
    )

    if delta_brow_down < 0.03:
        anger_onset *= 0.20
    if mouth_press < 0.08 and nose_sneer < 0.08 and eye_squint < 0.14:
        anger_onset *= 0.25
    if mouth_lower_down > 0.10 and mouth_press < 0.08 and nose_sneer < 0.08:
        anger_onset *= 0.25
    if delta_brow_down > 0.08:
        anger_onset += 0.08

    anger_evidence = clamp01(anger_onset)

    surprise_onset = clamp01(0.35 * eye_wide + 0.30 * brow_inner_up + 0.25 * jaw_open - 0.10 * brow_down)
    surprise_hold = clamp01(0.42 * eye_wide + 0.36 * brow_inner_up + 0.22 * jaw_open - 0.08 * brow_down)
    surprise_evidence = max(surprise_onset, surprise_hold)

    max_active = max(happy_evidence, sadness_evidence, anger_evidence, surprise_evidence)

    mp_scores = {
        "happy": happy_evidence,
        "sadness": sadness_evidence,
        "anger": anger_evidence,
        "surprise": surprise_evidence,
        "neutral": clamp01(1.0 - max_active),
    }

    features = {
        "brow_down": brow_down,
        "mouth_press": mouth_press,
        "nose_sneer": nose_sneer,
        "eye_squint": eye_squint,
        "frown": frown,
        "mouth_lower_down": mouth_lower_down,
    }

    return {
        **mp_scores,
        "features": features,
        "debug": {
            "smile": smile,
            "frown": frown,
            "brow_inner_up": brow_inner_up,
            "mouth_lower_down": mouth_lower_down,
            "brow_down": brow_down,
            "delta_brow_down": delta_brow_down,
            "mouth_press": mouth_press,
            "nose_sneer": nose_sneer,
            "eye_squint": eye_squint,
            "sadness_eye_component": sadness_eye_component,
            "sad_mouth_signature": sad_mouth_signature,
            "sad_brow_signature": sad_brow_signature,
            "anger_tension_signature": anger_tension_signature,
            "eye_wide": eye_wide,
            "jaw_open": jaw_open,
            "happy_evidence": happy_evidence,
            "sadness_evidence": sadness_evidence,
            "anger_evidence": anger_evidence,
            "surprise_evidence": surprise_evidence,
            "surprise_onset": surprise_onset,
            "surprise_hold": surprise_hold,
        }
    }
