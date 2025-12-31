"""rPPG signal extraction from facial ROI."""

import numpy as np

class SignalExtractor:
    def __init__(self):
        pass
    
    def extract(self, roi_frames, method='green'):
        """
        Extract blood volume pulse signal.
        
        Args:
            roi_frames: List or array of frames (N, H, W, 3).
            method: 'green' or 'pos'.
            
        Returns:
            np.array: 1D rPPG signal.
        """
        # 1. Spatial Averaging
        means = []
        for frame in roi_frames:
            if frame is None or frame.size == 0:
                means.append([0, 0, 0])
                continue
            
            if len(frame.shape) == 3:
                # BGR to RGB if needed? 
                # Assuming input is BGR (OpenCV default), so index 0=B, 1=G, 2=R
                # POS expects RGB usually, but the relative changes matter.
                # Let's stick to BGR order for consistency with OpenCV.
                mean_bgr = np.mean(frame, axis=(0, 1))
                means.append(mean_bgr)
            else:
                means.append([np.mean(frame)]*3)
        
        means = np.array(means) # (N, 3)
        
        if method == 'green':
            return means[:, 1] # Green channel
            
        elif method == 'pos':
            return self._pos(means)
            
        else:
            raise ValueError(f"Unknown method: {method}")

    def _pos(self, signals):
        """
        Plane-Orthogonal-to-Skin (POS) algorithm.
        Args:
            signals: (N, 3) array of BGR means.
        Returns:
            (N,) rPPG signal.
        """
        # Sliding window approach is often used, but for batch processing:
        # 1. Temporal Normalization
        # Divide by mean to get normalized color variations
        # Avoid division by zero
        mean_color = np.mean(signals, axis=0)
        if np.any(mean_color == 0):
            return np.zeros(len(signals))
            
        norm_signals = signals / mean_color # Cn
        
        # 2. Projection
        # POS uses a projection matrix. 
        # Assuming BGR input: B=0, G=1, R=2
        # Standard POS uses RGB. Let's map BGR to RGB indices.
        # R is index 2, G is index 1, B is index 0.
        
        # S1 = G - B
        # S2 = G + B - 2R
        
        # Using BGR indices:
        # S1 = signals[:, 1] - signals[:, 0]
        # S2 = signals[:, 1] + signals[:, 0] - 2 * signals[:, 2]
        
        s1 = norm_signals[:, 1] - norm_signals[:, 0]
        s2 = norm_signals[:, 1] + norm_signals[:, 0] - 2 * norm_signals[:, 2]
        
        # 3. Alpha Tuning
        # H = S1 + alpha * S2
        # alpha = std(S1) / std(S2)
        
        std1 = np.std(s1)
        std2 = np.std(s2)
        
        if std2 == 0:
            return s1
            
        alpha = std1 / std2
        h = s1 + alpha * s2
        
        return h
