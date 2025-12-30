import numpy as np
from .quality_metrics import compute_quality_metrics

def rppg_features(rppg_signal, fps):
    snr, heart_rate = compute_quality_metrics(rppg_signal, fps)

    confidence = np.clip(snr * 2.0, 0.0, 1.0)

    return {
        "snr": float(snr),
        "heart_rate": float(heart_rate),
        "confidence": float(confidence)
    }
