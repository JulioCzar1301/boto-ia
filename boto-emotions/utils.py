from config import MAIN_EMOTIONS


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def get_bs(blend_dict, name):
    return blend_dict.get(name, 0.0)


def get_winner(scores):
    if not scores:
        return "Nenhum", 0.0
    emotion, score = max(scores.items(), key=lambda x: x[1])
    return emotion, score


def normalize_scores(scores):
    scores = {k: max(0.0, float(v)) for k, v in scores.items()}
    total = sum(scores.values())

    if total <= 0:
        return {emo: 1.0 / len(MAIN_EMOTIONS) for emo in MAIN_EMOTIONS}

    return {k: v / total for k, v in scores.items()}


def mean_or_current(history, current_value):
    if len(history) == 0:
        return current_value
    return sum(history) / len(history)
