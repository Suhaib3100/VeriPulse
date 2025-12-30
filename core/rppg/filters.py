# filters.py
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fps, low=0.7, high=4.0, order=3):
    b, a = butter(order, [low, high], btype="bandpass", fs=fps)
    return filtfilt(b, a, signal)
