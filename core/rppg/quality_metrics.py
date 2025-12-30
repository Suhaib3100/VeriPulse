import numpy as np
from scipy.signal import welch

def compute_quality_metrics(rppg_signal, fps):
    freqs, power = welch(rppg_signal, fs=fps)

    hr_band = (freqs >= 0.7) & (freqs <= 4.0)
    band_power = power[hr_band]
    band_freqs = freqs[hr_band]

    peak_idx = np.argmax(band_power)
    peak_freq = band_freqs[peak_idx]
    peak_power = band_power[peak_idx]

    snr = peak_power / (np.sum(band_power) + 1e-6)
    heart_rate = peak_freq * 60.0

    return snr, heart_rate
