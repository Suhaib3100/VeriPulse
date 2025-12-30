import cv2
import time
import numpy as np

class VideoReader:
    def __init__(self, source=0, target_fps=30):
        """
        Initialize VideoReader.
        
        Args:
            source: Webcam index (int) or video file path (str).
            target_fps: Target FPS to yield frames at. If None, uses source FPS.
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
            
        self.target_fps = target_fps
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.source_fps <= 0:
            self.source_fps = 30.0 # Default fallback
            
        self.frame_interval = 1.0 / target_fps if target_fps else 1.0 / self.source_fps
        self.last_frame_time = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise StopIteration
                
            current_time = time.time()
            # Rate limiting
            if self.target_fps:
                elapsed = current_time - self.last_frame_time
                if elapsed < self.frame_interval:
                    # Sleep to save CPU
                    time.sleep(self.frame_interval - elapsed)
                    current_time = time.time()
                
            self.last_frame_time = current_time
            
            # Timestamp
            # For files, use CAP_PROP_POS_MSEC
            # For webcam, use relative time
            if self.source_fps > 0 and self.source_fps < 1000: # Likely a file
                 timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:
                 timestamp = time.time()
            
            return frame, timestamp

    def release(self):
        self.cap.release()
