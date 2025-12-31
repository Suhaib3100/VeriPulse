from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class PhysioFeatures:
    bpm_mean: float
    bpm_std: float
    snr_mean: float
    snr_std: float
    cross_roi_corr_mean: float
    ibi_cv: float
    roi_features: Dict[str, Any] = None

@dataclass
class ActiveChallengeResult:
    challenge_type: str
    requested_pattern: str
    detected_pattern: str
    timing_ok: bool
    geometry_ok: bool
    score: float

@dataclass
class LivenessResult:
    physio_score: float
    active_score: float
    final_score: float
    level: str  # "HIGH", "MEDIUM", "LOW"
    reasons: List[str]
    debug: Dict[str, Any]

def score_physiological_liveness(features: PhysioFeatures) -> tuple[float, List[str]]:
    reasons = []
    score = 0.5
    
    if 45 <= features.bpm_mean <= 120:
        reasons.append("BPM in physiological range")
        score += 0.2
    else:
        reasons.append("BPM out of range")
    
    if features.snr_mean >= 6.0:
        reasons.append("Good SNR")
        score += 0.2
    else:
        reasons.append("Low SNR")
    
    if features.cross_roi_corr_mean >= 0.7:
        reasons.append("Good ROI consistency")
        score += 0.2
    else:
        reasons.append("Poor ROI consistency")
    
    if features.ibi_cv <= 0.15:
        reasons.append("Stable heartbeat")
        score += 0.2
    
    return min(score, 1.0), reasons

def score_active_liveness(challenges: List[ActiveChallengeResult]) -> tuple[float, List[str]]:
    if not challenges:
        return 0.5, ["No active challenge performed"]
    
    scores = [c.score for c in challenges]
    reasons = [f"{c.challenge_type}: {'PASS' if c.timing_ok and c.geometry_ok else 'FAIL'}" for c in challenges]
    return np.mean(scores), reasons

def fuse_liveness_scores(physio_score: float, active_score: float, w_physio: float = 0.5, w_active: float = 0.5) -> float:
    return w_physio * physio_score + w_active * active_score

def compute_liveness_result(
    physio_features: PhysioFeatures,
    challenges: List[ActiveChallengeResult] = None,
    w_physio: float = 0.5,
    w_active: float = 0.5
) -> LivenessResult:
    challenges = challenges or []
    
    physio_score, physio_reasons = score_physiological_liveness(physio_features)
    active_score, active_reasons = score_active_liveness(challenges)
    final_score = fuse_liveness_scores(physio_score, active_score, w_physio, w_active)
    
    if final_score >= 0.7:
        level = "HIGH"
    elif final_score >= 0.4:
        level = "MEDIUM"
    else:
        level = "LOW"
    
    return LivenessResult(
        physio_score=physio_score,
        active_score=active_score,
        final_score=final_score,
        level=level,
        reasons=physio_reasons + active_reasons,
        debug={
            "bpm": physio_features.bpm_mean,
            "physio_bpm": physio_features.bpm_mean,
            "physio_snr": physio_features.snr_mean
        }
    )

if __name__ == "__main__":
    features = PhysioFeatures(75.0, 2.1, 8.2, 1.3, 0.85, 0.12)
    result = compute_liveness_result(features)
    print(result)
