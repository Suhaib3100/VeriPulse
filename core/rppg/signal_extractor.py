import numpy as np
from .filters import bandpass_filter

def extract_rppg_signal(frames, roi_func, fps):
    rgb_series = []

    for frame in frames:
        roi = roi_func(frame)
        mean_rgb = roi.mean(axis=(0, 1))
        rgb_series.append(mean_rgb)

    rgb_series = np.asarray(rgb_series)
    R, G, B = rgb_series[:, 0], rgb_series[:, 1], rgb_series[:, 2]

    # POS method
    X = G - B
    Y = G + B - 2 * R
    alpha = np.std(X) / (np.std(Y) + 1e-6)
    rppg_raw = X + alpha * Y

    return bandpass_filter(rppg_raw, fps)
