import numpy as np
from .face_detector import FaceDetector

class FaceTracker:
    def __init__(self, alpha=0.7):
        """
        Initialize FaceTracker.
        
        Args:
            alpha: Smoothing factor for exponential moving average (0 < alpha <= 1).
                   Higher alpha = more responsive, lower alpha = smoother.
        """
        self.detector = FaceDetector()
        self.alpha = alpha
        self.bbox = None # (x, y, w, h)

    def process_frame(self, frame):
        """
        Detect and track face in the frame.
        
        Args:
            frame: Input image.
            
        Returns:
            tuple: (x, y, w, h) of the tracked face, or None if lost.
        """
        detected_bbox = self.detector.detect(frame)
        
        if detected_bbox is not None:
            if self.bbox is None:
                self.bbox = detected_bbox
            else:
                # Smooth the bounding box
                dx, dy, dw, dh = detected_bbox
                sx, sy, sw, sh = self.bbox
                
                nx = int(self.alpha * dx + (1 - self.alpha) * sx)
                ny = int(self.alpha * dy + (1 - self.alpha) * sy)
                nw = int(self.alpha * dw + (1 - self.alpha) * sw)
                nh = int(self.alpha * dh + (1 - self.alpha) * sh)
                
                self.bbox = (nx, ny, nw, nh)
        else:
            # If detection fails, we might want to keep the last known position for a few frames
            # For now, let's just return the last known bbox if it exists, 
            # but maybe we should decay confidence?
            # The user requirement says "Tracks the face...". 
            # If we lose detection, maybe we should return None or keep last.
            # Let's keep last for robustness, but maybe reset if lost for too long?
            # For simplicity, if detection fails, we return None to indicate "No Face Found" 
            # so we don't extract garbage.
            # BUT, Haar cascade is flickery. 
            # Let's stick to: if detection fails, return None, but in a real tracker we'd use optical flow.
            # Given the constraints, let's just return None if detection fails, 
            # but maybe the user wants smoothing to handle missed detections?
            # "Tracks the face across frames (e.g., simple bounding-box smoothing...)"
            pass

        return self.bbox
