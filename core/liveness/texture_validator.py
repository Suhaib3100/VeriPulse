import cv2
import numpy as np

class TextureValidator:
    def __init__(self):
        self.scores = []

    def process(self, frame, face_bbox=None):
        """
        Analyze texture for smoothness (AI artifact) vs natural texture.
        If face_bbox is None, analyzes the entire frame.
        """
        try:
            if face_bbox:
                x, y, w, h = face_bbox
                # Extract face ROI
                roi = frame[y:y+h, x:x+w]
            else:
                # Analyze full frame
                roi = frame

            if roi.size == 0:
                return 0.0

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Laplacian Variance (Focus Measure)
            # High variance = sharp texture (pores, hair, wrinkles, background details)
            # Low variance = smooth/blurry (AI generation, filters, or out of focus)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            self.scores.append(variance)
            if len(self.scores) > 30:
                self.scores.pop(0)
                
            return variance
        except Exception:
            return 0.0

    def get_score(self):
        if not self.scores:
            return 0.0
            
        avg_variance = np.mean(self.scores)
        
        # Thresholds calibrated for standard 480p/720p webcams
        # < 10: Very blurry / Smooth (Likely AI or bad camera)
        # 10 - 50: Soft / Normal Webcam
        # > 50: Sharp (High Def Real texture)
        
        if avg_variance > 50:
            return 1.0
        elif avg_variance < 10:
            return 0.2 # Suspiciously smooth
        else:
            # Linear interpolation between 10 and 50
            return 0.2 + (0.8 * (avg_variance - 10) / 40)
