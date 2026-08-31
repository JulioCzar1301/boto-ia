from config import ACTIVE_EMOTIONS
from utils import clamp01, get_winner, normalize_scores


def fuse_enet_with_mediapipe(enet_scores, mp_evidence, quality_info):
    scores = enet_scores.copy()

    happy_ev = mp_evidence["happy"]
    sad_ev = mp_evidence["sadness"]
    anger_ev = mp_evidence["anger"]
    surprise_ev = mp_evidence["surprise"]

    active_ev = {
        "happy": happy_ev,
        "sadness": sad_ev,
        "anger": anger_ev,
        "surprise": surprise_ev,
    }

    dominant_mp, dominant_ev = get_winner(active_ev)
    dbg = mp_evidence["debug"]

    sad_mouth = dbg.get("sad_mouth_signature", 0.0)
    sad_brow = dbg.get("sad_brow_signature", 0.0)
    anger_tension = dbg.get("anger_tension_signature", 0.0)

    sad_face_pattern = clamp01(
        0.45 * dbg["mouth_lower_down"] +
        0.35 * dbg["frown"] +
        0.20 * dbg.get("brow_inner_up", 0.0)
    )

    happy_factor = 0.90 + 0.45 * happy_ev
    sadness_factor = 0.92 + 0.30 * sad_ev
    anger_factor = 0.38 + 0.46 * anger_ev
    surprise_factor = 0.88 + 0.55 * surprise_ev

    sadness_intent = clamp01(
        0.30 * enet_scores["sadness"] +
        0.25 * sad_ev +
        0.25 * sad_face_pattern +
        0.12 * sad_mouth +
        0.08 * sad_brow
    )

    anger_intent = clamp01(
        0.38 * enet_scores["anger"] +
        0.36 * anger_ev +
        0.26 * anger_tension
    )

    sadness_structural = (
        sad_face_pattern > 0.075 or sad_mouth > 0.055 or sad_brow > 0.12 or
        dbg["mouth_lower_down"] > 0.045 or dbg["frown"] > 0.085 or sad_ev > 0.045 or
        (
            enet_scores["sadness"] > 0.60 and
            (sad_face_pattern > 0.035 or sad_ev > 0.035 or mp_evidence["neutral"] < 0.82)
        )
    )

    enet_strong_sad = enet_scores["sadness"] > 0.65
    weak_sad_geometry = (
        sad_face_pattern > 0.04 or sad_ev > 0.035 or
        sad_brow > 0.10 or sad_mouth > 0.035
    )
    sadness_override_neutral = (
        enet_strong_sad and weak_sad_geometry and enet_scores["anger"] < 0.45
    )

    if sadness_override_neutral:
        sadness_structural = True

    anger_structural = (
        anger_tension > 0.13 or anger_ev > 0.24 or
        (
            enet_scores["anger"] > 0.76 and mp_evidence["neutral"] < 0.68 and
            sad_mouth < 0.09 and sad_face_pattern < 0.08
        )
    )

    eye_only_sadness_noise = (
        dbg["eye_squint"] > 0.11 and sad_face_pattern < 0.055 and
        sad_mouth < 0.045 and dbg["frown"] < 0.07 and
        dbg["mouth_lower_down"] < 0.04 and sad_ev < 0.08 and
        not enet_strong_sad
    )

    blink_like_noise = (
        dbg["eye_squint"] > 0.32 and sad_face_pattern < 0.07 and
        sad_mouth < 0.055 and dbg["mouth_lower_down"] < 0.045 and
        dbg["frown"] < 0.09 and not enet_strong_sad
    )

    if eye_only_sadness_noise:
        sadness_factor *= 0.28
        scores["sadness"] *= 0.35

    if blink_like_noise:
        sadness_factor *= 0.35
        scores["sadness"] *= 0.45

    if not sadness_structural:
        sadness_factor *= 0.40
        scores["sadness"] *= 0.45

    if not anger_structural:
        anger_factor *= 0.38
        scores["anger"] *= 0.45

    if sadness_override_neutral:
        scores["sadness"] *= 1.70
        sadness_factor += 0.40
        scores["neutral"] *= 0.50
        anger_factor *= 0.45
        scores["anger"] *= 0.55

    if sad_face_pattern > 0.07 or sad_mouth > 0.055:
        sadness_factor += 0.18
        scores["sadness"] *= 1.25
        anger_factor *= 0.55
        scores["anger"] *= 0.65

    if sad_mouth > 0.07:
        anger_factor *= max(0.22, 1.0 - 1.35 * sad_mouth)
        scores["anger"] *= max(0.35, 1.0 - 0.95 * sad_mouth)

    if sadness_structural and sadness_intent > 0.16:
        sadness_factor += 0.14
        anger_factor *= 0.72

    if sadness_structural and sadness_intent > 0.24:
        sadness_factor += 0.18
        scores["sadness"] *= 1.18
        anger_factor *= 0.55
        scores["anger"] *= 0.72

    if sadness_structural and sadness_intent > 0.34:
        sadness_factor += 0.22
        scores["sadness"] *= 1.24
        anger_factor *= 0.42
        scores["anger"] *= 0.58

    if sadness_structural and enet_scores["sadness"] > 0.30:
        sadness_factor += 0.10

    if sadness_structural and enet_scores["sadness"] > 0.45:
        sadness_factor += 0.14
        scores["sadness"] *= 1.18
        anger_factor *= 0.72

    if sadness_structural and enet_scores["sadness"] > 0.65:
        sadness_factor += 0.18
        scores["sadness"] *= 1.25
        scores["neutral"] *= 0.65

    if anger_structural and anger_intent > 0.35 and sad_face_pattern < 0.07:
        anger_factor += 0.10

    if anger_structural and anger_intent > 0.50 and sad_face_pattern < 0.07:
        anger_factor += 0.12

    if mp_evidence["neutral"] > 0.78:
        if sadness_override_neutral:
            scores["neutral"] *= 0.55
        elif not sadness_structural:
            sadness_factor *= 0.45
            scores["sadness"] *= 0.55
            scores["neutral"] += 0.10

        if not anger_structural:
            anger_factor *= 0.45
            scores["anger"] *= 0.55

    if sadness_structural and sad_face_pattern > 0.07 and scores["anger"] >= scores["sadness"] * 0.70:
        scores["anger"] *= 0.50
        anger_factor *= 0.50
        sadness_factor += 0.12

    if scores["anger"] > scores["neutral"] and anger_tension < 0.11 and anger_ev < 0.18:
        anger_factor *= 0.35
        scores["anger"] *= 0.50
        scores["neutral"] += 0.10

    if dominant_ev > 0.24:
        for emo in ACTIVE_EMOTIONS:
            if emo != dominant_mp:
                if emo == "sadness" and sadness_structural and sadness_intent > 0.18:
                    scores[emo] *= 0.98
                elif emo == "anger" and sadness_structural and sad_face_pattern > 0.06:
                    scores[emo] *= 0.62
                else:
                    scores[emo] *= 0.90
        scores[dominant_mp] *= 1.10

    if dbg["jaw_open"] > 0.28 and dbg["eye_wide"] < 0.08 and dbg["brow_down"] < 0.08:
        surprise_factor *= 0.75

    if enet_scores["surprise"] > 0.25 and surprise_ev > 0.20:
        surprise_factor += 0.12

    if quality_info.get("quality_penalty_active", False):
        sadness_factor *= 0.65
        anger_factor *= 0.70
        surprise_factor *= 0.78
        scores["sadness"] *= 0.65
        scores["anger"] *= 0.70
        scores["surprise"] *= 0.78
        scores["neutral"] += 0.18

    scores["happy"] *= happy_factor
    scores["sadness"] *= sadness_factor
    scores["anger"] *= anger_factor
    scores["surprise"] *= surprise_factor

    max_active_score = max(scores[emo] for emo in ACTIVE_EMOTIONS)
    max_active_ev = max(active_ev.values())

    if sadness_override_neutral:
        scores["neutral"] *= 0.55
    elif sadness_structural and enet_scores["sadness"] > 0.55 and max_active_score < 0.50:
        scores["neutral"] *= 0.70
    elif max_active_ev < 0.16 and max_active_score < 0.42:
        scores["neutral"] += 0.14
    else:
        scores["neutral"] *= 0.92

    fused_scores = normalize_scores(scores)

    effective_sadness_intent = sadness_intent if sadness_structural else sadness_intent * 0.30
    effective_anger_intent = anger_intent if anger_structural else anger_intent * 0.35

    evidence_for_state = {
        "happy": happy_ev,
        "sadness": (
            max(sad_ev, effective_sadness_intent, sad_face_pattern, sad_mouth)
            if sadness_structural else sad_ev * 0.50
        ),
        "anger": max(anger_ev, effective_anger_intent) if anger_structural else anger_ev * 0.35,
        "surprise": surprise_ev,
        "neutral": fused_scores.get("neutral", 0.0),
    }

    if sadness_override_neutral:
        evidence_for_state["sadness"] = max(evidence_for_state["sadness"], 0.42)
        evidence_for_state["neutral"] *= 0.75

    if quality_info.get("quality_penalty_active", False):
        evidence_for_state["sadness"] *= 0.75
        evidence_for_state["anger"] *= 0.75
        evidence_for_state["surprise"] *= 0.80

    debug_info = {
        "happy_factor": happy_factor,
        "sadness_factor": sadness_factor,
        "anger_factor": anger_factor,
        "surprise_factor": surprise_factor,
        "sadness_intent": sadness_intent,
        "effective_sadness_intent": effective_sadness_intent,
        "anger_intent": anger_intent,
        "effective_anger_intent": effective_anger_intent,
        "sad_face_pattern": sad_face_pattern,
        "sadness_structural": sadness_structural,
        "anger_structural": anger_structural,
        "sadness_override_neutral": sadness_override_neutral,
        "blink_like_noise": blink_like_noise,
        "eye_only_sadness_noise": eye_only_sadness_noise,
        "dominant_mp": dominant_mp,
        "dominant_ev": dominant_ev,
        **mp_evidence["debug"],
        **quality_info,
    }

    return fused_scores, evidence_for_state, debug_info