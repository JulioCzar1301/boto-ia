import numpy as np

from config import WARN_YAW_DEG, WARN_PITCH_DEG, WARN_ROLL_DEG, PENALTY_YAW_DEG, PENALTY_PITCH_DEG, PENALTY_ROLL_DEG


def rotation_matrix_to_euler_angles(rotation_matrix):
    r = rotation_matrix
    sy = np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(r[2, 1], r[2, 2])
        y = np.arctan2(-r[2, 0], sy)
        z = np.arctan2(r[1, 0], r[0, 0])
    else:
        x = np.arctan2(-r[1, 2], r[1, 1])
        y = np.arctan2(-r[2, 0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)


def get_head_pose_info(result, face_index=0):
    pose_info = {
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
        "pose_warning": False,
        "pose_penalty_active": False,
        "pose_message": "",
    }

    if not result.facial_transformation_matrixes:
        return pose_info

    if face_index >= len(result.facial_transformation_matrixes):
        return pose_info

    matrix_obj = result.facial_transformation_matrixes[face_index]

    try:
        matrix = np.array(matrix_obj.data).reshape(4, 4)
    except Exception:
        try:
            matrix = np.array(matrix_obj).reshape(4, 4)
        except Exception:
            return pose_info

    pitch, yaw, roll = rotation_matrix_to_euler_angles(matrix[:3, :3])
    yaw_abs, pitch_abs, roll_abs = abs(yaw), abs(pitch), abs(roll)

    warn = yaw_abs > WARN_YAW_DEG or pitch_abs > WARN_PITCH_DEG or roll_abs > WARN_ROLL_DEG
    penalty = yaw_abs > PENALTY_YAW_DEG or pitch_abs > PENALTY_PITCH_DEG or roll_abs > PENALTY_ROLL_DEG

    message = ""
    if warn:
        if yaw_abs > WARN_YAW_DEG:
            message = "Rosto virado. Olhe mais de frente."
        elif pitch_abs > WARN_PITCH_DEG:
            message = "Rosto para cima/baixo. Ajuste a camera."
        elif roll_abs > WARN_ROLL_DEG:
            message = "Cabeca inclinada. Tente alinhar o rosto."

    pose_info.update({
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "pose_warning": warn,
        "pose_penalty_active": penalty,
        "pose_message": message,
    })
    return pose_info


# =========================
# QUALIDADE DA FACE
# =========================
from config import ASYMMETRY_PENALTY, ASYMMETRY_WARN, PENALTY_FACE_RATIO_MAX, PENALTY_FACE_RATIO_MIN, WARN_FACE_RATIO_MAX, WARN_FACE_RATIO_MIN
from utils import get_bs


def get_face_distance_info(bbox, frame_w, frame_h):
    x1, y1, x2, y2 = bbox
    face_area = max(1, (x2 - x1) * (y2 - y1))
    frame_area = max(1, frame_w * frame_h)
    face_ratio = face_area / frame_area

    too_far_warn = face_ratio < WARN_FACE_RATIO_MIN
    too_close_warn = face_ratio > WARN_FACE_RATIO_MAX
    too_far_penalty = face_ratio < PENALTY_FACE_RATIO_MIN
    too_close_penalty = face_ratio > PENALTY_FACE_RATIO_MAX

    msg = ""
    if too_far_warn:
        msg = "Rosto muito longe. Aproxime um pouco."
    elif too_close_warn:
        msg = "Rosto muito perto. Afaste um pouco."

    return {
        "face_ratio": face_ratio,
        "distance_warning": too_far_warn or too_close_warn,
        "distance_penalty_active": too_far_penalty or too_close_penalty,
        "too_far": too_far_warn,
        "too_close": too_close_warn,
        "distance_message": msg,
    }


def paired_asymmetry(blend_dict, left_name, right_name):
    return abs(get_bs(blend_dict, left_name) - get_bs(blend_dict, right_name))


def get_face_asymmetry_info(blend_dict):
    asym_values = [
        paired_asymmetry(blend_dict, "mouthSmileLeft", "mouthSmileRight"),
        paired_asymmetry(blend_dict, "mouthFrownLeft", "mouthFrownRight"),
        paired_asymmetry(blend_dict, "browDownLeft", "browDownRight"),
        paired_asymmetry(blend_dict, "eyeSquintLeft", "eyeSquintRight"),
        paired_asymmetry(blend_dict, "eyeWideLeft", "eyeWideRight"),
        paired_asymmetry(blend_dict, "cheekSquintLeft", "cheekSquintRight"),
        paired_asymmetry(blend_dict, "noseSneerLeft", "noseSneerRight"),
        paired_asymmetry(blend_dict, "mouthPressLeft", "mouthPressRight"),
    ]

    face_asymmetry = float(np.mean(asym_values))
    max_asymmetry = float(np.max(asym_values))

    asymmetry_warning = face_asymmetry > ASYMMETRY_WARN or max_asymmetry > ASYMMETRY_WARN * 1.55
    asymmetry_penalty_active = face_asymmetry > ASYMMETRY_PENALTY or max_asymmetry > ASYMMETRY_PENALTY * 1.45

    return {
        "face_asymmetry": face_asymmetry,
        "max_asymmetry": max_asymmetry,
        "asymmetry_warning": asymmetry_warning,
        "asymmetry_penalty_active": asymmetry_penalty_active,
        "asymmetry_message": "Face parcialmente pouco confiavel." if asymmetry_warning else "",
    }


def get_face_quality_info(bbox, frame_w, frame_h, blend_dict, pose_info):
    distance_info = get_face_distance_info(bbox, frame_w, frame_h)
    asymmetry_info = get_face_asymmetry_info(blend_dict)

    quality_warning = (
        pose_info.get("pose_warning", False) or
        distance_info.get("distance_warning", False) or
        asymmetry_info.get("asymmetry_warning", False)
    )

    quality_penalty_active = (
        pose_info.get("pose_penalty_active", False) or
        distance_info.get("distance_penalty_active", False) or
        asymmetry_info.get("asymmetry_penalty_active", False)
    )

    quality_message = ""
    if pose_info.get("pose_warning", False):
        quality_message = pose_info.get("pose_message", "")
    elif distance_info.get("distance_warning", False):
        quality_message = distance_info.get("distance_message", "")
    elif asymmetry_info.get("asymmetry_warning", False):
        quality_message = asymmetry_info.get("asymmetry_message", "")

    return {
        "quality_warning": quality_warning,
        "quality_penalty_active": quality_penalty_active,
        "quality_message": quality_message,
        **pose_info,
        **distance_info,
        **asymmetry_info,
    }
