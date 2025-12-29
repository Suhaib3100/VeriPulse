# pulse_liveness.py
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import time

def bandpass(signal, fs):
    b, a = butter(3, [0.8 / (fs / 2), 2.0 / (fs / 2)], btype='band')
    return filtfilt(b, a, signal)

def get_pulse_score(duration=15, fs=30):
    cap = cv2.VideoCapture(0)
    signals = []
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        roi = frame[int(h*0.3):int(h*0.6), int(w*0.3):int(w*0.6)]
        mean_rgb = np.mean(roi.reshape(-1, 3), axis=0)
        signals.append(mean_rgb)

        time.sleep(1 / fs)

    cap.release()

    signals = np.array(signals)
    green_signal = signals[:, 1]
    filtered = bandpass(green_signal, fs)
    peaks, _ = find_peaks(filtered, distance=fs*0.5)

    bpm = len(peaks) * (60 / duration)

    if 60 <= bpm <= 120:
        pulse_score = 1.0
    elif 50 <= bpm <= 140:
        pulse_score = 0.6
    else:
        pulse_score = 0.0

    return float(np.clip(pulse_score, 0, 1))

if __name__ == "__main__":
    print(get_pulse_score())