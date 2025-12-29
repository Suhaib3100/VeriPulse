# audio_liveness.py
import numpy as np
import librosa
import pyaudio
import time

RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        frames.append(stream.read(CHUNK))

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
    return audio

def get_audio_score():
    audio = record_audio()
    y = audio / np.max(np.abs(audio) + 1e-6)

    mfcc = librosa.feature.mfcc(y=y, sr=RATE, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=RATE)

    mfcc_var = np.mean(np.var(mfcc, axis=1))
    centroid_var = np.var(centroid)

    if mfcc_var > 50 and centroid_var > 1000:
        score = 1.0
    elif mfcc_var > 20:
        score = 0.6
    else:
        score = 0.0

    return float(np.clip(score, 0, 1))

if __name__ == "__main__":
    print(get_audio_score())
