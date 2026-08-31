import time
import math

import cv2
import mediapipe as mp

from config import MAIN_EMOTIONS
from quality import get_head_pose_info, get_face_quality_info
from fusion import fuse_enet_with_mediapipe
from features import extract_mediapipe_evidence
from utils import normalize_scores
from visualization import draw_detection_overlay, draw_text


class EmotionFrameProcessor:

    def __init__(
        self,
        landmarker,
        enet,
        temporal_state,
        temporal_perceiver=None,
        use_temporal_as_final=False,
        draw_debug_overlay=False,
        render_overlay=True,
        max_track_distance=180,
        max_missing_frames=30,
    ):
        self.landmarker = landmarker
        self.enet = enet

        # O estado recebido no main funciona como template.
        # Para múltiplos rostos, cada face_id recebe seu próprio estado temporal.
        self.temporal_state_template = temporal_state
        self.temporal_perceiver_template = temporal_perceiver

        self.use_temporal_as_final = use_temporal_as_final
        self.draw_debug_overlay = draw_debug_overlay
        self.render_overlay = render_overlay

        self.max_track_distance = max_track_distance
        self.max_missing_frames = max_missing_frames

        self.next_face_id = 1
        self.face_tracks = {}

    def _build_default_result(self):
        return {
            "emotion": None,
            "confidence": 0.0,
            "scores": {},
            "enet_scores": {},
            "mediapipe_scores": {},
            "debug": {},
            "temporal": {
                "ready": False,
                "emotion": None,
                "confidence": 0.0,
                "scores": {},
            },
            "rule_based_emotion": None,
            "rule_based_confidence": 0.0,
            "detected": False,
            "bbox": None,
            "face_count": 0,
            "faces": [],
        }

    def _new_temporal_state(self):
        return type(self.temporal_state_template)()

    def _new_temporal_perceiver(self):
        if self.temporal_perceiver_template is None:
            return None

        if hasattr(self.temporal_perceiver_template, "clone_for_new_face"):
            return self.temporal_perceiver_template.clone_for_new_face()

        # Fallback conservador: não compartilha o mesmo buffer BiLSTM entre rostos.
        return None

    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    def _bbox_diag(self, bbox):
        x1, y1, x2, y2 = bbox
        return math.hypot(x2 - x1, y2 - y1)

    def _center_distance(self, bbox_a, bbox_b):
        ax, ay = self._bbox_center(bbox_a)
        bx, by = self._bbox_center(bbox_b)
        return math.hypot(ax - bx, ay - by)

    def _create_track(self, bbox):
        face_id = self.next_face_id
        self.next_face_id += 1

        self.face_tracks[face_id] = {
            "state": self._new_temporal_state(),
            "perceiver": self._new_temporal_perceiver(),
            "bbox": bbox,
            "center": self._bbox_center(bbox),
            "missing_frames": 0,
        }

        return face_id

    def _get_bbox_from_landmarks(self, face_landmarks, frame_w, frame_h):
        xs = [lm.x for lm in face_landmarks]
        ys = [lm.y for lm in face_landmarks]

        x1 = max(0, int(min(xs) * frame_w))
        y1 = max(0, int(min(ys) * frame_h))

        x2 = min(frame_w - 1, int(max(xs) * frame_w))
        y2 = min(frame_h - 1, int(max(ys) * frame_h))

        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.20)

        x1p = max(0, x1 - pad_x)
        y1p = max(0, y1 - pad_y)

        x2p = min(frame_w - 1, x2 + pad_x)
        y2p = min(frame_h - 1, y2 + pad_y)

        return (x1p, y1p, x2p, y2p)

    def _assign_face_tracks(self, detected_faces):
        """
        Associa cada rosto detectado a um face_id persistente usando proximidade
        entre centros de bboxes. Assim cada rosto mantém seu próprio histórico,
        suavização, histerese e buffer temporal da BiLSTM.
        """
        assigned_track_ids = set()

        for face_item in detected_faces:
            bbox = face_item["bbox"]

            best_track_id = None
            best_distance = None

            for track_id, track in self.face_tracks.items():
                if track_id in assigned_track_ids:
                    continue

                distance = self._center_distance(
                    bbox,
                    track["bbox"]
                )

                adaptive_limit = max(
                    self.max_track_distance,
                    self._bbox_diag(track["bbox"]) * 0.65,
                )

                if distance <= adaptive_limit:
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_track_id = track_id

            if best_track_id is None:
                best_track_id = self._create_track(bbox)

            assigned_track_ids.add(best_track_id)

            self.face_tracks[best_track_id]["bbox"] = bbox
            self.face_tracks[best_track_id]["center"] = self._bbox_center(bbox)
            self.face_tracks[best_track_id]["missing_frames"] = 0

            face_item["face_id"] = best_track_id

        # Remove tracks que sumiram por muitos frames.
        for track_id in list(self.face_tracks.keys()):
            if track_id not in assigned_track_ids:
                self.face_tracks[track_id]["missing_frames"] += 1

                if self.face_tracks[track_id]["missing_frames"] > self.max_missing_frames:
                    del self.face_tracks[track_id]

        return detected_faces

    def process_frame(self, frame_bgr):

        frame_out = frame_bgr.copy()

        rgb = cv2.cvtColor(
            frame_out,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(time.time() * 1000)

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        h, w, _ = frame_out.shape

        result_dict = self._build_default_result()

        # =====================================================
        # SEM ROSTO DETECTADO
        # =====================================================

        if not result.face_landmarks:
            for track_id in list(self.face_tracks.keys()):
                self.face_tracks[track_id]["missing_frames"] += 1

                if self.face_tracks[track_id]["missing_frames"] > self.max_missing_frames:
                    del self.face_tracks[track_id]

            if self.render_overlay and self.draw_debug_overlay:
                draw_text(
                    frame_out,
                    "Nenhum rosto detectado",
                    20,
                    30,
                    scale=0.65
                )

            return frame_out, result_dict

        # =====================================================
        # PREPARA LISTA DE ROSTOS
        # =====================================================

        detected_faces = []

        for face_index, face_landmarks in enumerate(result.face_landmarks):
            bbox = self._get_bbox_from_landmarks(
                face_landmarks,
                frame_w=w,
                frame_h=h
            )

            x1p, y1p, x2p, y2p = bbox
            area = max(0, x2p - x1p) * max(0, y2p - y1p)

            if area <= 0:
                continue

            detected_faces.append({
                "face_index": face_index,
                "landmarks": face_landmarks,
                "bbox": bbox,
                "area": area,
            })

        if not detected_faces:
            return frame_out, result_dict

        detected_faces = self._assign_face_tracks(detected_faces)

        # Maior rosto continua sendo o resultado principal retornado no dict,
        # mas todos os rostos detectados usam o pipeline completo.
        detected_faces.sort(
            key=lambda item: item["area"],
            reverse=True
        )

        primary_face_id = detected_faces[0]["face_id"]

        # =====================================================
        # PROCESSA CADA ROSTO
        # =====================================================

        faces_results = []

        for face_item in detected_faces:

            face_id = face_item["face_id"]
            face_index = face_item["face_index"]
            face_landmarks = face_item["landmarks"]
            bbox = face_item["bbox"]

            track = self.face_tracks[face_id]
            face_temporal_state = track["state"]
            face_temporal_perceiver = track["perceiver"]

            x1p, y1p, x2p, y2p = bbox

            is_primary_face = face_id == primary_face_id

            # =================================================
            # POSE / LANDMARKS DO ROSTO ATUAL
            # =================================================

            pose_info = get_head_pose_info(
                result,
                face_index=face_index
            )

            # =================================================
            # FACE CROP
            # =================================================

            face_rgb = rgb[y1p:y2p, x1p:x2p]

            if face_rgb.size == 0:
                continue

            # =================================================
            # ENET
            # =================================================

            enet_scores, full_enet_scores = (
                self.enet.predict(face_rgb)
            )

            enet_scores = normalize_scores(
                enet_scores
            )

            # =================================================
            # INICIALIZAÇÕES
            # =================================================

            mediapipe_scores = {}

            debug_info = {}

            history_updated = False

            evidence_for_state = {
                emo: 0.0
                for emo in MAIN_EMOTIONS
            }

            quality_info = pose_info.copy()

            # =================================================
            # MEDIAPIPE BLENDSHAPES
            # =================================================

            has_blendshapes = (
                result.face_blendshapes
                and face_index < len(result.face_blendshapes)
            )

            if has_blendshapes:

                blend_dict = {
                    bs.category_name: bs.score
                    for bs in result.face_blendshapes[face_index]
                }

                quality_info = get_face_quality_info(
                    bbox,
                    w,
                    h,
                    blend_dict,
                    pose_info
                )

                mp_evidence = extract_mediapipe_evidence(
                    blend_dict,
                    face_landmarks,
                    face_temporal_state.feature_history
                )

                mediapipe_scores = {
                    "happy": mp_evidence["happy"],
                    "sadness": mp_evidence["sadness"],
                    "anger": mp_evidence["anger"],
                    "surprise": mp_evidence["surprise"],
                    "neutral": mp_evidence["neutral"],
                }

                fused_scores, evidence_for_state, debug_info = (
                    fuse_enet_with_mediapipe(
                        enet_scores,
                        mp_evidence,
                        quality_info
                    )
                )

                history_updated = (
                    face_temporal_state.update_feature_history_if_neutral(
                        features=mp_evidence["features"],
                        mediapipe_scores=mediapipe_scores,
                        enet_scores=enet_scores,
                        fused_scores=fused_scores,
                        quality_info=quality_info,
                    )
                )

            # =================================================
            # FALLBACK SOMENTE ENET
            # =================================================

            else:

                fused_scores = enet_scores

                evidence_for_state = {
                    "happy": enet_scores.get("happy", 0.0),
                    "sadness": enet_scores.get("sadness", 0.0),
                    "anger": enet_scores.get("anger", 0.0),
                    "surprise": enet_scores.get("surprise", 0.0),
                    "neutral": enet_scores.get("neutral", 0.0),
                }

                debug_info = quality_info

            # =================================================
            # SUAVIZAÇÃO TEMPORAL DO ROSTO ATUAL
            # =================================================

            smoothed_scores = (
                face_temporal_state.smooth_scores(
                    fused_scores
                )
            )

            # =================================================
            # MODELO TEMPORAL SUPERVISIONADO DO ROSTO ATUAL
            # =================================================

            temporal_result = {
                "ready": False,
                "emotion": None,
                "confidence": 0.0,
                "scores": {},
            }

            if face_temporal_perceiver is not None:
                temporal_result = face_temporal_perceiver.update(
                    enet_scores=enet_scores,
                    mediapipe_scores=mediapipe_scores,
                    evidence=evidence_for_state,
                    debug_info=debug_info,
                    quality_info=quality_info,
                )

            emotion, confidence = (
                face_temporal_state.update_emotion_state(
                    smoothed_scores,
                    evidence_for_state,
                    temporal_result=temporal_result,
                )
            )

            rule_based_emotion = emotion
            rule_based_confidence = confidence

            temporal_switch_info = getattr(
                face_temporal_state,
                "last_temporal_switch_info",
                {}
            )

            debug_info = {
                **debug_info,
                "face_id": face_id,
                "face_index": face_index,
                "is_primary_face": is_primary_face,
                "temporal_switch_ready": temporal_switch_info.get("ready", False),
                "temporal_switch_decision": temporal_switch_info.get("decision", "not_used"),
                "temporal_switch_candidate": temporal_switch_info.get("candidate"),
                "temporal_switch_previous": temporal_switch_info.get("previous"),
                "temporal_switch_needed_persistence": temporal_switch_info.get("needed_persistence"),
                "temporal_switch_candidate_count": temporal_switch_info.get("candidate_count"),
            }

            if self.use_temporal_as_final and temporal_result.get("ready", False):
                emotion = temporal_result.get("emotion", emotion)
                confidence = temporal_result.get("confidence", confidence)

            # =================================================
            # OVERLAY BÁSICO + DEBUG OPCIONAL
            # =================================================

            if self.render_overlay:
                draw_detection_overlay(
                    frame_out=frame_out,
                    bbox=bbox,
                    emotion=emotion,
                    confidence=confidence,
                    smoothed_scores=smoothed_scores,
                    mediapipe_scores=mediapipe_scores,
                    enet_scores=enet_scores,
                    fused_scores=fused_scores,
                    quality_info=quality_info,
                    debug_info=debug_info,
                    temporal_result=temporal_result,
                    rule_based_emotion=rule_based_emotion,
                    rule_based_confidence=rule_based_confidence,
                    show_debug_panel=self.draw_debug_overlay and is_primary_face,
                )

            face_result = {
                "face_id": face_id,
                "face_index": face_index,
                "is_primary": is_primary_face,

                "emotion": emotion,
                "confidence": confidence,

                "rule_based_emotion": rule_based_emotion,
                "rule_based_confidence": rule_based_confidence,
                "temporal": temporal_result,

                "scores": smoothed_scores,

                "enet_scores": enet_scores,

                "mediapipe_scores": mediapipe_scores,

                "full_enet_scores": full_enet_scores,

                "evidence": evidence_for_state,

                "debug": debug_info,

                "history_updated": history_updated,

                "quality": quality_info,

                "detected": True,

                "bbox": bbox,
            }

            faces_results.append(face_result)

        # =====================================================
        # RESULTADO FINAL
        # =====================================================

        if not faces_results:
            return frame_out, result_dict

        faces_results.sort(
            key=lambda item: item["is_primary"],
            reverse=True
        )

        primary_face = faces_results[0]

        result_dict = {
            "emotion": primary_face["emotion"],
            "confidence": primary_face["confidence"],

            "rule_based_emotion": primary_face["rule_based_emotion"],
            "rule_based_confidence": primary_face["rule_based_confidence"],
            "temporal": primary_face["temporal"],

            "scores": primary_face["scores"],

            "enet_scores": primary_face["enet_scores"],

            "mediapipe_scores": primary_face["mediapipe_scores"],

            "full_enet_scores": primary_face["full_enet_scores"],

            "evidence": primary_face["evidence"],

            "debug": primary_face["debug"],

            "history_updated": primary_face["history_updated"],

            "quality": primary_face["quality"],

            "detected": True,

            "bbox": primary_face["bbox"],

            "face_count": len(faces_results),

            "faces": faces_results,
        }

        return frame_out, result_dict
