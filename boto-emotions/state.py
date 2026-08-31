from collections import deque

from config import (
    ACTIVE_EMOTIONS,
    ALPHA,
    MAIN_EMOTIONS,
    SWITCH_MARGIN,
    SWITCH_PERSISTENCE,
    WINDOW_SIZE,
)
from utils import get_winner, normalize_scores


class TemporalEmotionState:
    def __init__(self):
        self.smooth_scores_state = {
            emo: 0.0 for emo in MAIN_EMOTIONS
        }

        self.feature_history = {
            "brow_down": deque(maxlen=WINDOW_SIZE),
            "mouth_press": deque(maxlen=WINDOW_SIZE),
            "nose_sneer": deque(maxlen=WINDOW_SIZE),
            "eye_squint": deque(maxlen=WINDOW_SIZE),
            "frown": deque(maxlen=WINDOW_SIZE),
            "mouth_lower_down": deque(maxlen=WINDOW_SIZE),
        }

        self.current_emotion = "neutral"
        self.candidate_emotion = None
        self.candidate_count = 0

    def smooth_scores(self, raw_scores):
        for emo in MAIN_EMOTIONS:
            v = raw_scores.get(emo, 0.0)

            self.smooth_scores_state[emo] = (
                ALPHA * v
                + (1 - ALPHA) * self.smooth_scores_state[emo]
            )

        return normalize_scores(
            self.smooth_scores_state.copy()
        )

    def update_feature_history_if_neutral(
        self,
        features,
        mediapipe_scores,
        enet_scores,
        fused_scores,
        quality_info,
    ):
        if quality_info.get("quality_penalty_active", False):
            return False

        fused_winner, _ = get_winner(fused_scores)

        max_enet_active = max(
            enet_scores.get(emo, 0.0)
            for emo in ACTIVE_EMOTIONS
        )

        max_mp_active = max(
            mediapipe_scores.get(emo, 0.0)
            for emo in ACTIVE_EMOTIONS
        )

        should_update = (
            mediapipe_scores.get("neutral", 0.0) > 0.45
            and enet_scores.get("neutral", 0.0) > 0.20
            and max_mp_active < 0.42
            and max_enet_active < 0.60
            and fused_winner == "neutral"
        )

        if not should_update:
            return False

        for key in self.feature_history:
            self.feature_history[key].append(features[key])

        return True

    def update_emotion_state(self, scores, evidence, temporal_result=None):
        """
        Atualiza a emoção estabilizada usando:
        1. regra híbrida atual;
        2. persistência manual;
        3. BiLSTM como fiscal de troca, quando disponível.

        Importante:
        - A BiLSTM NÃO substitui a decisão final.
        - Ela apenas confirma ou freia uma troca proposta pelo sistema híbrido.
        - Surprise ignora a BiLSTM, porque o modelo temporal foi treinado sem essa classe.
        """
        winner, winner_score = get_winner(scores)

        temporal_result = temporal_result or {}
        temporal_ready = temporal_result.get("ready", False)
        temporal_emotion = temporal_result.get("emotion", None)
        temporal_confidence = temporal_result.get("confidence", 0.0)

        self.last_temporal_switch_info = {
            "ready": temporal_ready,
            "temporal_emotion": temporal_emotion,
            "temporal_confidence": temporal_confidence,
            "candidate": winner,
            "previous": self.current_emotion,
            "decision": "not_used",
            "needed_persistence": SWITCH_PERSISTENCE,
            "candidate_count": self.candidate_count,
        }

        current_score = scores.get(
            self.current_emotion,
            0.0
        )

        current_evidence = evidence.get(
            self.current_emotion,
            0.0
        )

        winner_evidence = evidence.get(
            winner,
            0.0
        )

        active_max = max(
            scores.get("happy", 0.0),
            scores.get("sadness", 0.0),
            scores.get("anger", 0.0),
            scores.get("surprise", 0.0),
        )

        # Surprise continua sendo controlado pelo sistema híbrido atual.
        # A BiLSTM não conhece surprise, então não deve frear essa troca.
        temporal_can_judge = (
            temporal_ready
            and temporal_emotion is not None
            and winner != "surprise"
            and self.current_emotion != "surprise"
        )

        if self.current_emotion == "anger":
            anger_evidence = evidence.get("anger", 0.0)
            neutral_score = scores.get("neutral", 0.0)

            if (
                anger_evidence < 0.18
                and neutral_score > 0.28
                and active_max < 0.40
            ):
                # Se a BiLSTM ainda vê anger, segura um pouco a saída para neutral.
                if temporal_can_judge and temporal_emotion == "anger":
                    self.last_temporal_switch_info.update({
                        "decision": "blocked_anger_to_neutral",
                        "candidate": "neutral",
                        "previous": "anger",
                    })
                    return self.current_emotion, current_score

                self.current_emotion = "neutral"
                self.candidate_emotion = None
                self.candidate_count = 0

                return self.current_emotion, neutral_score

        if (
            self.current_emotion == "neutral"
            and winner == "sadness"
        ):
            if (
                winner_evidence > 0.38
                and winner_score > current_score + 0.03
            ):
                # Se a BiLSTM confirma sadness, permite a entrada rápida.
                # Se discordar, cai no fluxo normal de persistência.
                if temporal_can_judge and temporal_emotion != "sadness":
                    self.last_temporal_switch_info.update({
                        "decision": "slowed_neutral_to_sadness",
                        "candidate": "sadness",
                        "previous": "neutral",
                    })
                else:
                    self.current_emotion = "sadness"
                    self.candidate_emotion = None
                    self.candidate_count = 0

                    return self.current_emotion, winner_score

        if winner == self.current_emotion:
            self.candidate_emotion = None
            self.candidate_count = 0

            self.last_temporal_switch_info.update({
                "decision": "same_emotion",
                "candidate": winner,
                "previous": self.current_emotion,
                "candidate_count": self.candidate_count,
            })

            return self.current_emotion, winner_score

        if self.current_emotion != "neutral":
            if self.current_emotion == "anger":
                still_valid = (
                    current_evidence > 0.32
                    and current_score > 0.32
                )
            elif self.current_emotion == "sadness":
                still_valid = (
                    current_evidence > 0.24
                    and current_score > 0.24
                )
            else:
                still_valid = (
                    current_evidence > 0.22
                    and current_score > 0.25
                )

            challenger_stronger = (
                winner_evidence > current_evidence + 0.08
            )

            # Se a BiLSTM ainda concorda com a emoção anterior,
            # exige um challenger ainda mais forte.
            if temporal_can_judge and temporal_emotion == self.current_emotion:
                challenger_stronger = (
                    winner_evidence > current_evidence + 0.16
                    and winner_score > current_score + 0.08
                )

                self.last_temporal_switch_info.update({
                    "decision": "temporal_supports_previous",
                    "candidate": winner,
                    "previous": self.current_emotion,
                })

            if still_valid and not challenger_stronger:
                self.candidate_emotion = None
                self.candidate_count = 0

                return self.current_emotion, current_score

        score_advantage = (
            winner_score > current_score + SWITCH_MARGIN
        )

        evidence_advantage = (
            winner_evidence > current_evidence + 0.10
            and winner_score > current_score + 0.04
        )

        if score_advantage or evidence_advantage:
            if self.candidate_emotion == winner:
                self.candidate_count += 1
            else:
                self.candidate_emotion = winner
                self.candidate_count = 1

            needed_persistence = SWITCH_PERSISTENCE

            if winner_evidence > 0.35:
                needed_persistence = max(
                    2,
                    SWITCH_PERSISTENCE - 2
                )

            if winner == "sadness" and winner_evidence > 0.38:
                needed_persistence = 1

            if winner == "neutral" and current_evidence < 0.20:
                needed_persistence = max(
                    2,
                    SWITCH_PERSISTENCE - 2
                )

            # =====================================================
            # BiLSTM como fiscal de troca
            # =====================================================
            if temporal_can_judge:
                if temporal_emotion == winner:
                    # A janela recente parece com a emoção candidata:
                    # aceita a troca mais cedo.
                    needed_persistence = max(1, needed_persistence - 2)
                    self.last_temporal_switch_info["decision"] = "confirmed_candidate"

                elif temporal_emotion == self.current_emotion:
                    # A janela recente ainda parece com a emoção anterior:
                    # segura a troca por mais tempo.
                    needed_persistence = needed_persistence + 2
                    self.last_temporal_switch_info["decision"] = "blocked_or_slowed"

                else:
                    # A BiLSTM não confirma nem a atual nem a candidata:
                    # mantém cautela leve.
                    needed_persistence = needed_persistence + 1
                    self.last_temporal_switch_info["decision"] = "uncertain"

            else:
                if winner == "surprise":
                    self.last_temporal_switch_info["decision"] = "ignored_for_surprise"
                else:
                    self.last_temporal_switch_info["decision"] = "not_ready"

            self.last_temporal_switch_info.update({
                "needed_persistence": needed_persistence,
                "candidate_count": self.candidate_count,
                "candidate": winner,
                "previous": self.current_emotion,
            })

            if self.candidate_count >= needed_persistence:
                self.current_emotion = winner
                self.candidate_emotion = None
                self.candidate_count = 0

                self.last_temporal_switch_info.update({
                    "decision": "switch_applied",
                    "candidate_count": self.candidate_count,
                })

        else:
            self.candidate_emotion = None
            self.candidate_count = 0

            self.last_temporal_switch_info.update({
                "decision": "no_advantage",
                "candidate": winner,
                "previous": self.current_emotion,
                "candidate_count": self.candidate_count,
            })

        return (
            self.current_emotion,
            scores.get(self.current_emotion, 0.0)
        )
