# liveness_fusion.py
from .blink_detector import get_vision_score
from .pulse_liveness import get_pulse_score
from .audio_liveness import get_audio_score
from .behavior_liveness import get_behavior_score

def run_liveness():
    vision_score = get_vision_score()
    pulse_score = get_pulse_score()
    audio_score = get_audio_score()
    behavior_score = get_behavior_score()

    liveness_score = (
        0.35 * vision_score +
        0.30 * pulse_score +
        0.20 * audio_score +
        0.15 * behavior_score
    )

    if liveness_score >= 0.8:
        decision = "LIVE HUMAN"
    elif liveness_score >= 0.5:
        decision = "UNCERTAIN"
    else:
        decision = "FAKE / NON-LIVE"

    return liveness_score, decision

if __name__ == "__main__":
    score, decision = run_liveness()
    print(score, decision)