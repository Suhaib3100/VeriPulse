import numpy as np
import librosa

class AudioLivenessDetector:
    def __init__(self, rate=16000, chunk_duration=5):
        self.rate = rate
        self.target_samples = rate * chunk_duration
        self.audio_buffer = np.array([], dtype=np.float32)

    def process_chunk(self, audio_bytes: bytes):
        """
        Process incoming raw audio bytes (16-bit PCM).
        """
        # Convert bytes to float32 array
        # Assuming 16-bit PCM input
        chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        
        # Normalize to [-1, 1]
        chunk = chunk / 32768.0
        
        self.audio_buffer = np.concatenate((self.audio_buffer, chunk))
        
        # Keep only the last N seconds
        if len(self.audio_buffer) > self.target_samples:
            self.audio_buffer = self.audio_buffer[-self.target_samples:]

    def analyze(self):
        """
        Analyze the current buffer for liveness.
        Returns a score between 0.0 (fake) and 1.0 (real).
        """
        if len(self.audio_buffer) < self.rate: # Need at least 1 second
            return 0.5 # Uncertain
            
        y = self.audio_buffer
        
        # Feature Extraction
        try:
            # MFCCs (Mel-frequency cepstral coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=self.rate, n_mfcc=13)
            
            # Spectral Centroid
            centroid = librosa.feature.spectral_centroid(y=y, sr=self.rate)
            
            # Spectral Flatness (Tonality vs Noise)
            flatness = librosa.feature.spectral_flatness(y=y)
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            
            # Spectral Contrast (Peaks vs Valleys)
            contrast = librosa.feature.spectral_contrast(y=y, sr=self.rate)
            
            # Variances
            mfcc_var = np.mean(np.var(mfcc, axis=1))
            centroid_var = np.var(centroid)
            flatness_mean = np.mean(flatness)
            contrast_mean = np.mean(contrast)
            
            score = 0.0
            
            # 1. Dynamic Range (Stricter)
            # Real speech has high variance in timbre. AI is often smoother.
            if mfcc_var > 150: score += 0.3
            
            # 2. Frequency Distribution
            if centroid_var > 5000: score += 0.2
            
            # 3. Spectral Contrast (Richness of sound)
            # Low contrast = muddy/synthetic. High = clear/natural.
            if contrast_mean > 20: score += 0.2
            
            # 4. Natural Noise/Breathiness
            # Too flat = robotic. Too noisy = bad mic.
            if 0.001 < flatness_mean < 0.05: score += 0.2
            
            # 5. Zero Crossing (Sibilance)
            if np.mean(zcr) > 0.05: score += 0.1
            
            # Penalty for "Too Perfect" (Low variance in contrast/flatness)
            if np.var(flatness) < 0.0001: score -= 0.3
            
            # 5. Duration Bonus
            if len(y) > self.rate * 2: score += 0.1 
            
            return float(np.clip(score, 0, 1))
            
        except Exception as e:
            print(f"Audio analysis error: {e}")
            return 0.0

    def clear(self):
        self.audio_buffer = np.array([], dtype=np.float32)