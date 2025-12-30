import cv2
import numpy as np
import time
from core.rppg.signal_extractor import SignalExtractor
from core.rppg.filters import BandpassFilter
from core.rppg.features import FeatureExtractor
from core.rppg.quality_metrics import QualityAnalyzer

class PulseLiveness:
    def __init__(self, fs=30):
        self.fs = fs
        self.signal_extractor = SignalExtractor()
        self.filter = BandpassFilter(fs=fs)
        self.feature_extractor = FeatureExtractor(fs=fs)
        self.quality_analyzer = QualityAnalyzer()

    def evaluate(self, frames, rois=None):
        """
        Evaluate pulse liveness from a sequence of frames.
        
        Args:
            frames: List of frames (numpy arrays).
            rois: List of ROIs (x, y, w, h). If None, uses a central crop.
            
        Returns:
            dict: Liveness score and metadata.
        """
        if not frames:
            return {"score": 0.0, "details": "No frames provided"}

        # If no ROIs provided, define a central ROI
        if rois is None:
            h, w = frames[0].shape[:2]
            # Central region (approx forehead/face center)
            rois = [(int(w*0.3), int(h*0.2), int(w*0.4), int(h*0.4))]

        # Extract signals for each ROI
        roi_signals = []
        for (x, y, rw, rh) in rois:
            # Ensure ROI is within bounds
            x, y = max(0, x), max(0, y)
            rw = min(rw, frames[0].shape[1] - x)
            rh = min(rh, frames[0].shape[0] - y)
            
            roi_frames = [f[y:y+rh, x:x+rw] for f in frames]
            raw_signal = self.signal_extractor.extract(roi_frames)
            filtered_signal = self.filter.apply(raw_signal)
            roi_signals.append(filtered_signal)

        if not roi_signals:
             return {"score": 0.0, "details": "No signals extracted"}

        # Compute features for the primary ROI (first one)
        primary_signal = roi_signals[0]
        features = self.feature_extractor.extract(primary_signal)
        
        # Compute consistency if multiple ROIs
        consistency = 1.0
        if len(roi_signals) > 1:
            # Correlation between first and others
            correlations = []
            for i in range(1, len(roi_signals)):
                # Normalize
                s1 = (primary_signal - np.mean(primary_signal))
                s2 = (roi_signals[i] - np.mean(roi_signals[i]))
                
                std1 = np.std(s1)
                std2 = np.std(s2)
                
                if std1 > 0 and std2 > 0:
                    s1 /= std1
                    s2 /= std2
                    # Ensure same length
                    min_len = min(len(s1), len(s2))
                    corr = np.corrcoef(s1[:min_len], s2[:min_len])[0, 1]
                    correlations.append(corr)
            
            if correlations:
                consistency = np.mean(correlations)
                # Clip negative correlations to 0 for scoring
                consistency = max(0.0, consistency)

        features['consistency'] = consistency
        
        # Compute final score
        quality_score = self.quality_analyzer.analyze(primary_signal, features)
        
        # Combine quality with consistency
        # If consistency is low, it might be noise or fake
        # We weight consistency into the final score
        final_score = quality_score * (0.7 + 0.3 * consistency)
        
        return {
            "score": float(np.clip(final_score, 0, 1)),
            "details": features
        }

def get_pulse_score(duration=15, fs=30):
    """
    Capture video from webcam and compute pulse score.
    This is a demo function.
    """
    cap = cv2.VideoCapture(0)
    frames = []
    start = time.time()
    
    print(f"Capturing for {duration} seconds...")
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)
        # Limit frame rate capture if needed, but usually read() blocks
        # time.sleep(1/fs) 

    cap.release()
    print(f"Captured {len(frames)} frames.")
    
    if not frames:
        return 0.0

    liveness = PulseLiveness(fs=fs)
    
    # Define multiple ROIs for consistency check
    h, w = frames[0].shape[:2]
    rois = [
        (int(w*0.3), int(h*0.2), int(w*0.4), int(h*0.2)), # Forehead
        (int(w*0.2), int(h*0.5), int(w*0.2), int(h*0.2)), # Left Cheek
        (int(w*0.6), int(h*0.5), int(w*0.2), int(h*0.2))  # Right Cheek
    ]
    
    result = liveness.evaluate(frames, rois=rois)
    print(f"Result: {result}")
    return result["score"]

if __name__ == "__main__":
    print(get_pulse_score())